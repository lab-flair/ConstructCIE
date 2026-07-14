"""Few-shot job bucketing for SLURM script generation.

Bin-pack a flat list of few-shot `k` values into load-balanced buckets whose
total weight fits the SLURM time budget for the (task, model). `run.py`
emits one set of SLURM scripts per bucket; buckets are independent and the
user submits them separately.

Weights drive ONLY the partition decision — every bucket still uses the
same SLURM --time as today (the original `get_duration`).

Config keys (resolved through the caller's `get_config_fn`, which honors
the same task→model→default lookup chain as the rest of run.py):
  few_shot_buckets   — [min_buckets, max_buckets]. The natural FFD count
                       is clamped into this range. `[1, 1]` means "no
                       bucketing" — every shot lands in one bucket. When
                       min and max are equal, that exact count is forced.
  few_shot_weights   — {str(shot): float} per-shot weights. Missing shots
                       default to 0.0 via the config's own `default` field,
                       so an absent config equals "no bucketing".
"""


def parse_slurm_time_to_hours(time_str):
    """'08:30:00' → 8.5. Accepts HH:MM:SS produced by run.get_duration."""
    parts = str(time_str).strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS, got {time_str!r}")
    h, m, s = (int(p) for p in parts)
    return h + m / 60.0 + s / 3600.0


def _ffd_pack(weighted_shots, capacity):
    """First-Fit-Decreasing bin-pack. `weighted_shots` is [(shot, weight), ...]
    already sorted by weight desc. Each bucket stays ≤ capacity unless a
    single item already exceeds it (then it goes alone)."""
    buckets = []
    for s, w in weighted_shots:
        placed = False
        for b in buckets:
            if b["weight"] + w <= capacity:
                b["shots"].append(s)
                b["weight"] += w
                placed = True
                break
        if not placed:
            buckets.append({"shots": [s], "weight": w})
    return buckets


def _lpt_pack(weighted_shots, n_buckets):
    """Greedy LPT into exactly `n_buckets` buckets, ignoring capacity.
    `weighted_shots` already sorted by weight desc.

    Tiebreaker: when two buckets have equal total weight, the one with
    fewer items wins. That keeps shots round-robin-distributed in the
    degenerate all-zero-weight case so we don't return empty buckets.
    """
    buckets = [{"shots": [], "weight": 0.0} for _ in range(n_buckets)]
    for s, w in weighted_shots:
        b = min(buckets, key=lambda b: (b["weight"], len(b["shots"])))
        b["shots"].append(s)
        b["weight"] += w
    return buckets


def _resolve_bucket_bounds(get_config_fn, task, model_key, n_shots):
    """Read `few_shot_buckets` → (min_buckets, max_buckets), clamped to >= 1
    and <= n_shots. Missing config falls through to (1, 1) (no bucketing)."""
    try:
        raw = get_config_fn("few_shot_buckets", model=model_key, task=task)
    except KeyError:
        return 1, 1
    if raw is None:
        return 1, 1
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lo, hi = int(raw[0]), int(raw[1])
    elif isinstance(raw, int):
        lo = hi = int(raw)
    else:
        raise ValueError(
            f"few_shot_buckets for task={task!r}, model={model_key!r} must be "
            f"[min, max] or int, got {raw!r}")
    lo = max(1, min(lo, n_shots))
    hi = max(lo, min(hi, n_shots))
    return lo, hi


def bucket_shots_balanced(shots, task, model_key, get_config_fn,
                          capacity_hours, logger=None):
    """Partition `shots` into load-balanced buckets for one (task, model).

    Internally resolves config via `get_config_fn` (typically
    `run.get_global_config`), which is expected to honor the
    task→model→default lookup chain and raise `KeyError` when a field has
    no value at any level.

    Strategy:
      1. Read [min_buckets, max_buckets] from `few_shot_buckets`. If max==1
         we short-circuit to a single bucket — bucketing is off.
      2. Read per-shot weights from `few_shot_weights`; missing entries get
         weight 0.0 (so absent config also collapses to one bucket).
      3. First-Fit-Decreasing with capacity=`capacity_hours`. This is the
         minimum bucket count that keeps every bucket within the SLURM time
         budget (assuming no single shot exceeds capacity).
      4. Clamp into [min_buckets, max_buckets]:
           * count < min  → re-pack via LPT into exactly min buckets
                            (gives more parallelism than needed for time).
           * count > max  → re-pack via LPT into exactly max buckets
                            (some buckets will exceed capacity; warned).

    Returns a list of non-empty buckets, each a list of shots in the
    original input order.
    """
    if not shots:
        return []
    if capacity_hours <= 0:
        raise ValueError(f"capacity_hours must be > 0, got {capacity_hours!r}")

    min_buckets, max_buckets = _resolve_bucket_bounds(
        get_config_fn, task, model_key, len(shots))
    if max_buckets <= 1:
        return [list(shots)]

    # Per-shot weight lookup. Pushes the default through the existing
    # task→model→default chain so the helper inherits whatever
    # `few_shot_weights.default` is set to (typically 0) rather than
    # hardcoding a fallback here.
    def _weight_for(s):
        try:
            v = get_config_fn(
                "few_shot_weights", model=model_key, task=task, k=s)
        except KeyError:
            return 0.0
        if v is None:
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    weighted = [(s, _weight_for(s)) for s in shots]
    indexed = sorted(weighted, key=lambda kv: -kv[1])

    if logger is not None:
        for s, w in indexed:
            if w > capacity_hours:
                logger.warning(
                    "shot k=%s has weight %.2fh > capacity %.2fh; its bucket "
                    "may hit the wall-clock limit", s, w, capacity_hours)

    buckets = _ffd_pack(indexed, capacity_hours)

    if len(buckets) < min_buckets:
        if logger is not None:
            logger.info(
                "FFD produced %d bucket(s) at capacity %.2fh; expanding to "
                "few_shot_buckets minimum=%d via LPT",
                len(buckets), capacity_hours, min_buckets)
        buckets = _lpt_pack(indexed, min_buckets)
    elif len(buckets) > max_buckets:
        if logger is not None:
            logger.warning(
                "FFD needs %d bucket(s) at capacity %.2fh, but "
                "few_shot_buckets maximum=%d — falling back to LPT with %d "
                "buckets; some buckets will exceed capacity",
                len(buckets), capacity_hours, max_buckets, max_buckets)
        buckets = _lpt_pack(indexed, max_buckets)
        if logger is not None:
            for i, b in enumerate(buckets, 1):
                if b["weight"] > capacity_hours:
                    logger.warning(
                        "  bucket %d total %.2fh > capacity %.2fh",
                        i, b["weight"], capacity_hours)

    pos = {s: i for i, s in enumerate(shots)}
    for b in buckets:
        b["shots"].sort(key=lambda s: pos[s])
    # Sort buckets themselves by their first (lowest) shot so emitted scripts
    # go b1=lowest-k…bN=highest-k. Keeps filenames predictable across runs and
    # makes the master submit.sh launch lighter buckets first.
    non_empty = [b for b in buckets if b["shots"]]
    non_empty.sort(key=lambda b: pos[b["shots"][0]])
    return [b["shots"] for b in non_empty]
