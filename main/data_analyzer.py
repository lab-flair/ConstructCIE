"""Analysis-mode renderer for TextEE.

`render_analysis_tables` first splits the (model, task, dataset, kshot) ->
preds+golds pool populated by `progress_analyzer.check_combo` apart by
dataset (see `progress_analyzer.split_preds_golds_by_dataset`), then builds
three views per dataset from the resulting (model, task, kshot) -> preds+golds
slice:

  1. TASK-LEVEL F1        — rows: all models. cols: best F1 across k-shots for
                            each of {exact (sim=1.0), soft (sim=0.8), keyword}.
  2. FEW-SHOT SCALING ×3  — rows: LLM models. cols: k-shot sizes. F1 cells.
                            One table per metric (exact / soft / keyword).
  3. FIELD-LEVEL HEATMAP×3 — rows: factors (with sub-factor indent). cols:
                            models. cell: that model's factor F1 at the k-shot
                            that maximized its TOTAL F1 ("best operating point"
                            per model). One table per metric.

All three are emitted twice — plain-text (.log) and LaTeX (.tex). The text
form mirrors `print_score_comparison`'s grid style so the file stays grep/
diff-friendly; the LaTeX form uses booktabs + resizebox so it drops straight
into a paper.
"""

import logging
import os
import re
import io
import contextlib
import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score collection
# ---------------------------------------------------------------------------

def _collect_skip(preds_golds, skip_unions, base_skip_by_combo):
    """Per-(task, shot) skip union across datasets — mirrors what
    `render_cross_model_comparison` already does. Returns a dict keyed by
    (task, shot) -> set of column keys to skip."""
    cross_model_skip = {}
    if skip_unions:
        for (task_k, _ds_k, shot_k), entry in skip_unions.items():
            if not isinstance(entry, dict):
                entry = {"aggregate": entry or []}
            cross_model_skip.setdefault((task_k, shot_k), set()).update(
                entry.get("aggregate") or [])
    return cross_model_skip


def _score_one_combo(args):
    """Process-pool worker: runs all three rescorings (exact / soft / keyword)
    for one (model, task, shot) combo, both on the pooled preds+golds AND
    on each split separately. Defined at module level so it pickles cleanly
    under `spawn` on Windows / macOS.

    `factors.overall` is stripped of every classification factor from the
    key map (`task.classification.classes` — threaded in as `strict_keys`,
    see `_build_all_scores`), for every task, not just E2. Classification
    factors are closed-vocabulary labels, not extraction targets, so
    `overall` stays a consistent, extraction-only measure regardless of
    whether a given (model, shot) happened to skip_column a factor out
    beforehand — deterministic instead of depending on few-shot pool
    coverage. `per_factor.<key>` is left intact for each of them, so the
    standalone per-factor views (accident_type, etc.) still surface.

    `args` is a tuple — see `_build_all_scores` for the layout."""
    (key, preds, golds, per_split, skip_columns,
     strict_keys, norm, has_kw) = args
    from TextEE.scorer import (compute_AC_scores, compute_AC_keyword_scores,
                               exclude_factor_from_overall)

    def _score_pair(ps, gs):
        e = compute_AC_scores(ps, gs, sim_threshold=1.0,
                              skip_columns=skip_columns,
                              norm=norm, strict_keys=strict_keys)
        s = compute_AC_scores(ps, gs, sim_threshold=0.8,
                              skip_columns=skip_columns,
                              norm=norm, strict_keys=strict_keys)
        k = None
        if has_kw:
            k = compute_AC_keyword_scores(ps, gs,
                                           skip_columns=skip_columns,
                                           strict_keys=strict_keys)
        for cls_key in strict_keys:
            exclude_factor_from_overall(e, cls_key)
            exclude_factor_from_overall(s, cls_key)
            if k is not None:
                exclude_factor_from_overall(k, cls_key)
        return e, s, k

    sc_exact, sc_soft, sc_kw = _score_pair(preds, golds)
    # Per-split scoring — used by the analysis tables that report F1
    # sum-averaged across the 5 cross-validation splits (NOT pooled
    # micro-F1). Empty list when caller didn't populate per_split.
    per_split_scores = []
    for sp in per_split or []:
        sp_e, sp_s, sp_k = _score_pair(sp["preds"], sp["golds"])
        per_split_scores.append({
            "name": sp["name"],
            "exact": sp_e, "soft": sp_s, "kw": sp_k,
        })
    return key, sc_exact, sc_soft, sc_kw, per_split_scores


def _build_all_scores(preds_golds, skip_unions, structure, classes_map,
                      norm=False, max_workers=None):
    """Compute three parallel score maps from the pooled preds+golds.

    Returns (scores_exact, scores_soft, scores_keyword, is_supervised_map),
    each scores dict keyed by (model_display, task, shot_str) -> the same
    shape `compute_AC_scores` returns. `compute_AC_keyword_scores` is only
    invoked for entries whose gold pool carries `*_keywords` annotations.

    Combos are scored in a `ProcessPoolExecutor` — exact + soft + keyword
    for one combo run sequentially in one worker (amortizes the pickle cost
    of preds+golds), and different combos run in parallel. Falls back to
    serial when there's only one combo or `max_workers <= 1`."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # Lazy imports — kept out of worker payload, run only in the main process.
    from main.progress_analyzer import (_supervised_classification_skip,
                                        _has_keywords)

    strict_keys = list(classes_map.keys()) if classes_map else []
    cross_model_skip = _collect_skip(preds_golds, skip_unions, None)

    # Build the per-combo argument tuples in the main process so workers
    # don't need to re-import progress_analyzer or recompute skip vectors.
    args_list = []
    is_supervised_map = {}
    for (mk, task_, shot), entry in preds_golds.items():
        preds, golds = entry["preds"], entry["golds"]
        if len(preds) != len(golds):
            continue
        if entry.get("is_supervised"):
            skip_for_this = list(_supervised_classification_skip(
                classes_map, preds))
        elif (task_, shot) in cross_model_skip:
            skip_for_this = list(cross_model_skip[(task_, shot)])
        else:
            skip_for_this = list(entry.get("skip_columns") or [])
        skip_for_this = sorted(
            set(skip_for_this) | set(entry.get("base_skip") or []))

        key = (mk, task_, shot)
        per_split = entry.get("per_split") or []
        args_list.append((key, preds, golds, per_split, skip_for_this,
                          strict_keys, norm, _has_keywords(golds)))
        is_supervised_map[mk] = bool(entry.get("is_supervised"))

    scores_exact, scores_soft, scores_kw = {}, {}, {}
    # Per-split scores keyed the same as pooled — used by analysis tb1/tb2
    # for split-averaged F1. Each value is a list of per-split score dicts:
    #   [{"name": "split1", "exact": <score>, "soft": <score>, "kw": <score>}, …]
    scores_per_split = {}
    if not args_list:
        return (scores_exact, scores_soft, scores_kw, scores_per_split,
                is_supervised_map)

    # Default worker pool: cap at cpu_count() and the actual combo count.
    # Cap at 8 by default — diminishing returns past that for compute-bound
    # numpy+scipy work, and pickle/IPC overhead grows with worker count.
    if max_workers is None:
        max_workers = min(mp.cpu_count() or 1, len(args_list), 8)

    if max_workers <= 1 or len(args_list) == 1:
        # Serial fallback — avoids the ProcessPoolExecutor spin-up cost when
        # there's nothing to parallelize.
        results = (_score_one_combo(a) for a in args_list)
    else:
        logger.info("\U0001f9ee Rescoring %d combo(s) on %d workers...",
                    len(args_list), max_workers)
        # chunksize tuned so each worker takes ~equal batches; small enough
        # that progress is visible even on short runs.
        chunksize = max(1, len(args_list) // (max_workers * 4))
        pool = ProcessPoolExecutor(max_workers=max_workers)
        try:
            results = list(pool.map(_score_one_combo, args_list,
                                     chunksize=chunksize))
        finally:
            pool.shutdown(wait=True)

    for key, sc_e, sc_s, sc_k, sp_list in results:
        scores_exact[key] = sc_e
        scores_soft[key] = sc_s
        if sc_k is not None:
            scores_kw[key] = sc_k
        if sp_list:
            scores_per_split[key] = sp_list

    return (scores_exact, scores_soft, scores_kw, scores_per_split,
            is_supervised_map)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _natural_key(s):
    return tuple(int(t) if t.isdigit() else t.lower()
                 for t in re.split(r'(\d+)', s))


def _f1(node):
    """Pull a numeric F1 (or None) out of an overall/per_factor node."""
    if not node:
        return None
    v = node.get("f1")
    if v is None or isinstance(v, str):
        return None
    return float(v)


def _scope_f1(score_dict, key):
    """Overall F1 for a (model, task, shot) score entry."""
    sc = score_dict.get(key)
    if not sc:
        return None
    return _f1(sc.get("factors", {}).get("overall"))


def _factor_f1(score_dict, key, factor):
    """Per-factor overall F1 for a (model, task, shot) score entry."""
    sc = score_dict.get(key)
    if not sc:
        return None
    return _f1(((sc.get("factors", {}).get("per_factor") or {})
                .get(factor) or {}).get("overall"))


def _fmt_f1(v):
    """%.1f or `--` for missing. One decimal is the right resolution for
    F1 reported as percent — keeps each cell to four characters so the
    tabular stays compact."""
    return "--" if v is None else f"{v:.1f}"


def _esc_latex(s):
    return (str(s)
            .replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#")
            .replace("$", r"\$"))


def _title(s):
    return " ".join(p.capitalize() for p in str(s).split("_"))


# Short display names for factors / sub-factors — keep tables compact.
# Keys are the canonical factor keys; values are the labels rendered in
# every analysis table (text + LaTeX). Unknown keys fall back to `_title`.
_FACTOR_SHORTNAMES = {
    # Standalone leaves
    "accident_type":                  "Acc. Type",
    "object_involved":                "Obj. Inv.",
    # Mains
    "working_circumstances":          "Work. Circ.",
    "managerial_factors":             "Mgmt",
    "working_condition_factors":      "Cond.",
    "equipment_factors":              "Equip.",
    "behavioral_factors":             "Behav.",
    "consequences":                   "Conseq.",
    # Subs
    "construction_trade":             "Constr. Trade",
    "failure_of_hazard_management":   "Hazard Mgmt",
    "deficiency_in_safety_training":  "Safety Train.",
    "weather_condition":              "Weather",
    "workspace_condition":            "Workspace",
    "protective_equipment_condition": "Prot. Equip.",
    "work_equipment_condition":       "Work Equip.",
    "inattentive_behavior":           "Inatt. Behav.",
    "noncompliant_behavior":          "Noncomp.",
    "severity":                       "Severity",
    "affected_body_part":             "Body Part",
}


def _short(fk):
    """Compact display label for a factor key. Falls back to title-cased
    underscore-split when the key isn't in the explicit shortname map."""
    return _FACTOR_SHORTNAMES.get(fk, _title(fk))


def _collect_factor_order(scores, structure, force_factors=None):
    """Compute factor ordering identical to the cross-model table:
    main → its subs (per `structure`), then leftovers alphabetical,
    with `accident_type` hoisted to the front.

    `force_factors` is a set of factor keys to keep in the row order even
    when this task's scores don't carry a real F1 for them (cells render as
    "--"). Callers use this to keep `accident_type` aligned across tasks
    when the analysis scope includes at least one task that scores it
    (e.g. mixed ACH + E2ACH). When EVERY task in scope skips a factor,
    it gets dropped entirely — no point in a uniform "--" row."""
    all_factors = set()
    for sc in scores.values():
        all_factors.update((sc.get("factors", {}).get("per_factor") or {}))

    def _has_real(fk):
        for sc in scores.values():
            v = ((sc.get("factors", {}).get("per_factor") or {})
                 .get(fk) or {}).get("overall", {}).get("f1")
            if v is not None and not isinstance(v, str):
                return True
        return False
    all_factors = {f for f in all_factors if _has_real(f)}
    if force_factors:
        all_factors = all_factors | set(force_factors)

    main_to_subs = {m: list((structure or {}).get(m, []))
                    for m in (structure or {}).get("accident_report", [])}
    is_main = set(main_to_subs)
    is_sub = {s for subs in main_to_subs.values() for s in subs}

    ordered = []
    for main, subs in main_to_subs.items():
        if main in all_factors:
            ordered.append(main)
        for sub in subs:
            if sub in all_factors:
                ordered.append(sub)
    leftovers = sorted(f for f in all_factors if f not in is_main and f not in is_sub)
    ordered.extend(leftovers)
    if "accident_type" in ordered:
        ordered.remove("accident_type")
        ordered.insert(0, "accident_type")
    return ordered, is_main, is_sub


def _decorate(name, is_main, is_sub):
    """Display label for the text-mode heatmap row. Uses the short-name map
    for compactness; mains rendered upper-case, subs prefixed with ' ↳'."""
    label = _short(name)
    if name in is_main:
        return label.upper()
    if name in is_sub:
        return " ↳" + label
    return label


def _best_per_metric(score_dict, models, task, shots):
    """For each model, the best (k, F1) across shots. Returns
    dict[model] -> (best_k, best_f1) with None when the model has no entry."""
    out = {}
    for m in models:
        best = None
        for k in shots:
            v = _scope_f1(score_dict, (m, task, k))
            if v is None:
                continue
            if best is None or v > best[1]:
                best = (k, v)
        out[m] = best
    return out


# ---------------------------------------------------------------------------
# TABLES 1 + 2 — Per-(model, kshot) factor F1, split-averaged.
#   Rows: (model, kshot) pairs, supervised models grouped first.
#   Cols: factor × 3 metric sub-cols (Exact, Soft, Keyword).
#   Cell value: F1 sum-averaged over the 5 cross-validation splits (and
#               across all AC tasks in scope). Missing values are skipped
#               from the mean, not counted as zero.
#   Table 1 covers CAUSAL factors (mains + standalone leaves);
#   Table 2 covers SUB-CAUSAL factors (subs of mains).
# ---------------------------------------------------------------------------

# Default k-shot rows shown in tb1/tb2 — keeps the table compact even when
# the run pooled every shot. Intersected with what's actually present.
_ANALYSIS_KSHOTS = ["0", "5", "30"]
# Metric sub-column ordering for tb1/tb2. Short labels keep the table from
# overflowing — full metric names live in the caption.
_TB_METRICS = [("exact", "E"), ("soft", "S"), ("kw", "K")]


def _split_avg_factor_f1(per_split_list, factor, metric_key):
    """Mean of F1 across a single combo's splits for one (factor, metric).

    Per-split entries with no real F1 (NaN string / missing factor /
    skip_columns) are dropped from the mean — they're not counted as 0.
    Returns None when no split has a real value."""
    vals = []
    for sp in per_split_list:
        sc = sp.get(metric_key)
        if not sc:
            continue
        node = ((sc.get("factors", {}).get("per_factor") or {})
                .get(factor) or {}).get("overall", {})
        v = node.get("f1")
        if v is None or isinstance(v, str):
            continue
        vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _mean_factor_f1(scores_per_split, model, shot, factor, metric_key,
                     tasks):
    """Sum-average F1 across (5 splits × N AC tasks in scope) for the given
    (model, kshot, factor, metric). Equivalent to flattening all per-split
    F1s from every AC task and taking the arithmetic mean — matches what
    the user calls 'sum-average' (NOT micro / macro)."""
    vals = []
    for task in tasks:
        per_split_list = scores_per_split.get((model, task, shot))
        if not per_split_list:
            continue
        for sp in per_split_list:
            sc = sp.get(metric_key)
            if not sc:
                continue
            node = ((sc.get("factors", {}).get("per_factor") or {})
                    .get(factor) or {}).get("overall", {})
            v = node.get("f1")
            if v is None or isinstance(v, str):
                continue
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _mean_overall_f1(scores_per_split, model, shot, metric_key, tasks):
    """Sum-average OVERALL F1 (i.e. factors.overall) across (splits × tasks)
    for the given (model, kshot, metric). Same averaging convention as
    `_mean_factor_f1` — used by the scaling chart so its lines match the
    table cell values."""
    vals = []
    for task in tasks:
        per_split_list = scores_per_split.get((model, task, shot))
        if not per_split_list:
            continue
        for sp in per_split_list:
            sc = sp.get(metric_key)
            if not sc:
                continue
            v = (sc.get("factors", {}).get("overall") or {}).get("f1")
            if v is None or isinstance(v, str):
                continue
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _split_main_and_sub_factors(structure, all_factors):
    """Return (causal_factors, sub_factors). Causal = mains from structure
    + leftovers (not in any main/sub list). Sub = items in any sub list."""
    main_to_subs = {m: list((structure or {}).get(m, []))
                    for m in (structure or {}).get("accident_report", [])}
    mains = list(main_to_subs.keys())
    sub_set = {s for subs in main_to_subs.values() for s in subs}
    main_set = set(mains)
    leftovers = sorted(f for f in all_factors
                       if f not in main_set and f not in sub_set)
    # Causal factors: mains in structure order + leftovers (alpha).
    # Filter to what actually appears in scored data.
    causal = [m for m in mains if m in all_factors] + leftovers
    # Sub-causal factors: subs in structure order (per their parent main).
    sub = []
    for main in mains:
        for s in main_to_subs[main]:
            if s in all_factors:
                sub.append(s)
    return causal, sub


def _row_sort_key(model, is_sup):
    """Supervised models first, then natural-sort by model name."""
    return (0 if is_sup else 1, _natural_key(model))


# Variant suffixes (e.g. "_c1", "_c2") get appended by check_combo to keep
# multi-variant runs apart in the cross-model pool. The analysis table only
# shows one variant's column per model, so strip the tag for display.
_VARIANT_SUFFIX_RE = re.compile(r"_c\d+$")


def _display_model(m, model_aliases=None):
    """Render a model's row label: alias if known, otherwise the pool key
    with any `_c<N>` variant tag stripped."""
    if model_aliases and m in model_aliases:
        return model_aliases[m]
    return _VARIANT_SUFFIX_RE.sub("", m)


def _display_k(k, is_supervised):
    """Render the `k` cell. Supervised models train on the full training
    set (no few-shot), so report `-` instead of the literal 0."""
    return "-" if is_supervised else str(k)


def _chunk_factors(factors, n_chunks=2):
    """Split factor list into `n_chunks` ~equal-size chunks. Returns a
    single-chunk list when len(factors) <= 1 or n_chunks <= 1."""
    import math
    n = len(factors)
    if n <= 1 or n_chunks <= 1:
        return [factors]
    per = math.ceil(n / n_chunks)
    chunks = [factors[i:i + per] for i in range(0, n, per)]
    return [c for c in chunks if c]


def _metrics_for_factor(f, classification_keys):
    """[E] only for a classification factor (soft/keyword match aren't
    meaningful for a discrete label), else the full [E, S, K]."""
    if f in (classification_keys or ()):
        return _TB_METRICS[:1]
    return _TB_METRICS


def _print_factor_table_text(title, factors, scores_per_split, models,
                              shots, sup_map, tasks, structure, n_chunks=1,
                              model_aliases=None, classification_keys=None):
    if not factors or not models or not tasks:
        return
    # Build (model, kshot) rows in display order. Supervised models train
    # once on the full set — k is irrelevant for them, so collapse to a
    # single row per supervised model regardless of how many shots were
    # pooled.
    row_keys = []
    for m in sorted(models, key=lambda mm: _row_sort_key(mm, sup_map.get(mm))):
        is_sup = sup_map.get(m)
        for k in shots:
            if any((m, t, k) in scores_per_split for t in tasks):
                row_keys.append((m, k))
                if is_sup:
                    break  # one row per supervised model
    if not row_keys:
        return

    chunks = _chunk_factors(factors, n_chunks=n_chunks)
    def _disp(m): return _display_model(m, model_aliases)
    def _kshow(m, k): return _display_k(k, sup_map.get(m))
    name_w = max(len("model"),
                 max(len(_disp(m)) for m, _ in row_keys))
    metric_w = 6

    def _block_w(f):
        n = len(_metrics_for_factor(f, classification_keys))
        return metric_w * n + 2 * (n - 1)  # n cells + (n-1) inter-cell gaps

    print(f"\n  {title}")
    for ci, chunk in enumerate(chunks):
        header1 = (
            "model".ljust(name_w) + "  k  "
            + "  ".join(_short(f).center(_block_w(f)) for f in chunk))
        header2 = (
            " " * name_w + "     "
            + "  ".join(
                "  ".join(lbl.center(metric_w)
                          for _, lbl in _metrics_for_factor(f, classification_keys))
                for f in chunk))
        line = "-" * max(len(header1), len(header2))
        if ci > 0:
            print()  # visual gap between sub-tables
        print("  " + line)
        print("  " + header1)
        print("  " + header2)
        print("  " + line)
        prev_model = None
        for (m, k) in row_keys:
            if prev_model is not None and m != prev_model:
                print("  " + "-" * len(header1))
            blocks = []
            for f in chunk:
                cells = []
                for metric_key, _ in _metrics_for_factor(f, classification_keys):
                    v = _mean_factor_f1(scores_per_split, m, k, f,
                                         metric_key, tasks)
                    cells.append(_fmt_f1(v).rjust(metric_w))
                blocks.append("  ".join(cells))
            row = (_disp(m).ljust(name_w) + " "
                   + _kshow(m, k).rjust(3) + "  "
                   + "  ".join(blocks))
            print("  " + row)
            prev_model = m
        print("  " + line)


def _tex_bold(s):
    return r"\textbf{" + s + "}"


def _print_factor_table_latex(title, label_suffix, factors, scores_per_split,
                               models, shots, sup_map, tasks, structure,
                               caption, n_chunks=1, model_aliases=None,
                               classification_keys=None):
    if not factors or not models or not tasks:
        return
    row_keys = []
    for m in sorted(models, key=lambda mm: _row_sort_key(mm, sup_map.get(mm))):
        is_sup = sup_map.get(m)
        for k in shots:
            if any((m, t, k) in scores_per_split for t in tasks):
                row_keys.append((m, k))
                if is_sup:
                    break  # one row per supervised model — k is irrelevant
    if not row_keys:
        return
    chunks = _chunk_factors(factors, n_chunks=n_chunks)
    def _disp(m): return _display_model(m, model_aliases)
    def _kshow(m, k): return _display_k(k, sup_map.get(m))

    def _emit_chunk_tabular(chunk):
        factor_metrics = [_metrics_for_factor(f, classification_keys) for f in chunk]
        col_max = {}
        for f, metrics in zip(chunk, factor_metrics):
            for metric_key, _ in metrics:
                vals = [_mean_factor_f1(scores_per_split, m, k, f, metric_key, tasks)
                        for m, k in row_keys]
                valid = [v for v in vals if v is not None]
                col_max[(f, metric_key)] = max(valid) if valid else None
        print(r"\resizebox{\textwidth}{!}{%")
        print(r"\begin{tabular}{l c "
              + " ".join("c" * len(metrics) for metrics in factor_metrics) + "}")
        print(r"\toprule")
        print(r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{$k$}}")
        for f, metrics in zip(chunk, factor_metrics):
            print(r"& \multicolumn{" + str(len(metrics)) + r"}{c}{\textbf{"
                  + _esc_latex(_short(f)) + "}}")
        print(r" \\")
        col = 3
        for metrics in factor_metrics:
            lo, hi = col, col + len(metrics) - 1
            print(r"\cmidrule(lr){" + f"{lo}-{hi}" + "}")
            col = hi + 1
        print(r" &  "
              + " ".join("& " + " & ".join(lbl for _, lbl in metrics)
                          for metrics in factor_metrics)
              + r" \\")
        print(r"\midrule")
        prev_model = None
        for (m, k) in row_keys:
            if prev_model is not None and m != prev_model:
                print(r"\midrule")
            cells = [_esc_latex(_disp(m)), _kshow(m, k)]
            for f, metrics in zip(chunk, factor_metrics):
                for metric_key, _ in metrics:
                    v = _mean_factor_f1(scores_per_split, m, k, f,
                                         metric_key, tasks)
                    fmt = _fmt_f1(v)
                    mx = col_max.get((f, metric_key))
                    if v is not None and mx is not None and v == mx:
                        fmt = _tex_bold(fmt)
                    cells.append(fmt)
            print(" & ".join(cells) + r" \\")
            prev_model = m
        print(r"\bottomrule")
        print(r"\end{tabular}%")
        print(r"}")

    print()
    print(f"% {title}")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{2pt}")
    print(r"\renewcommand{\arraystretch}{0.95}")
    for ci, chunk in enumerate(chunks):
        if ci > 0:
            print()
            print(r"\vspace{0.35em}")
            print()
        _emit_chunk_tabular(chunk)
    print(r"\caption{" + caption + "}")
    print(r"\label{tab:analysis-" + label_suffix + "}")
    print(r"\end{table*}")


# ---------------------------------------------------------------------------
# TABLES 3 + 4 — strategy comparisons (Joint vs Individual).
#   Table 3: JHE vs IHE overall.
#   Table 4: Joint (JHE) vs Individual (IHE). LLMs only — supervised models
#            train on one strategy per checkpoint.
# Both use the same row layout as tb1/tb2 (supervised collapsed to one
# row with k="-"). Cells are F1 sum-averaged across the 5 splits.
# ---------------------------------------------------------------------------

_E2_TASKS_PAIR = ("JHE", "IHE")


def _filter_tasks(tasks, allowed):
    return [t for t in tasks if t in allowed]


def _comparison_row_keys(scores_per_split, models, shots, sup_map, tasks,
                          llm_only=False):
    """Build (model, kshot) rows for a comparison table. Supervised models
    collapse to one row regardless of how many shots were pooled (since
    k is meaningless for them). When `llm_only`, supervised models are
    dropped entirely (used by the strategy table)."""
    row_keys = []
    candidates = sorted(models, key=lambda mm: _row_sort_key(mm, sup_map.get(mm)))
    for m in candidates:
        is_sup = sup_map.get(m)
        if llm_only and is_sup:
            continue
        for k in shots:
            if any((m, t, k) in scores_per_split for t in tasks):
                row_keys.append((m, k))
                if is_sup:
                    break
    return row_keys


def _print_setting_comparison_text(title, scores_per_split, models, shots,
                                     sup_map, tasks, model_aliases=None):
    e2_tasks = _filter_tasks(tasks, _E2_TASKS_PAIR)
    if not e2_tasks:
        return
    row_keys = _comparison_row_keys(scores_per_split, models, shots, sup_map,
                                     e2_tasks)
    if not row_keys:
        return
    def _disp(m): return _display_model(m, model_aliases)
    def _kshow(m, k): return _display_k(k, sup_map.get(m))
    name_w = max(len("model"), max(len(_disp(m)) for m, _ in row_keys))
    cell_w = 6
    block_w = cell_w * 3 + 2 * 2  # 3 metric cells + 2 inter-gaps

    header1 = (
        "model".ljust(name_w) + "  k  "
        + "Hierarchical Extraction".center(block_w))
    header2 = (
        " " * name_w + "     "
        + "  ".join(lbl.center(cell_w) for _, lbl in _TB_METRICS))
    line = "-" * max(len(header1), len(header2))

    print(f"\n  {title}")
    print("  " + line)
    print("  " + header1)
    print("  " + header2)
    print("  " + line)

    prev_model = None
    for (m, k) in row_keys:
        if prev_model is not None and m != prev_model:
            print("  " + "-" * len(header1))
        cells = []
        for metric_key, _ in _TB_METRICS:
            v = _mean_overall_f1(scores_per_split, m, k, metric_key, e2_tasks)
            cells.append(_fmt_f1(v).rjust(cell_w))
        row = (_disp(m).ljust(name_w) + " "
               + _kshow(m, k).rjust(3) + "  "
               + "  ".join(cells))
        print("  " + row)
        prev_model = m
    print("  " + line)


def _print_setting_comparison_latex(label_suffix, scores_per_split, models,
                                      shots, sup_map, tasks, caption,
                                      model_aliases=None):
    e2_tasks = _filter_tasks(tasks, _E2_TASKS_PAIR)
    if not e2_tasks:
        return
    row_keys = _comparison_row_keys(scores_per_split, models, shots, sup_map,
                                     e2_tasks)
    if not row_keys:
        return
    def _disp(m): return _display_model(m, model_aliases)
    def _kshow(m, k): return _display_k(k, sup_map.get(m))

    print()
    print("% Setting comparison — Hierarchical Extraction (JHE / IHE)")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{l c ccc}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{\textbf{Model}}")
    print(r"& \multirow{2}{*}{\textbf{$k$}}")
    print(r"& \multicolumn{3}{c}{\textbf{Hierarchical Extraction}} \\")
    print(r"\cmidrule(lr){3-5}")
    print(r"&  & \textbf{E} & \textbf{S} & \textbf{K} \\")
    print(r"\midrule")

    col_max_setting = {}
    for metric_key, _ in _TB_METRICS:
        vals = [_mean_overall_f1(scores_per_split, m, k, metric_key, e2_tasks)
                for m, k in row_keys]
        valid = [v for v in vals if v is not None]
        col_max_setting[metric_key] = max(valid) if valid else None

    prev_model = None
    for (m, k) in row_keys:
        if prev_model is not None and m != prev_model:
            print(r"\midrule")
        cells = [_esc_latex(_disp(m)), _kshow(m, k)]
        for metric_key, _ in _TB_METRICS:
            v = _mean_overall_f1(scores_per_split, m, k, metric_key, e2_tasks)
            fmt = _fmt_f1(v)
            mx = col_max_setting.get(metric_key)
            if v is not None and mx is not None and v == mx:
                fmt = _tex_bold(fmt)
            cells.append(fmt)
        print(" & ".join(cells) + r" \\")
        prev_model = m
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{" + caption + "}")
    print(r"\label{tab:analysis-" + label_suffix + "}")
    print(r"\end{table*}")


_JOINT_TASKS = ("JHE",)
_INDIVIDUAL_TASKS = ("IHE",)


def _print_strategy_comparison_text(title, scores_per_split, models, shots,
                                      sup_map, tasks, model_aliases=None):
    """Joint = F1(JHE); Individual = F1(IHE).
    accident_type is already stripped from E2 overalls by _score_one_combo.

    Supervised models are excluded — they only train on Joint (JHE),
    so Individual cells would be uniformly '--' and the comparison
    isn't meaningful."""
    joint_tasks = _filter_tasks(tasks, _JOINT_TASKS)
    indiv_tasks = _filter_tasks(tasks, _INDIVIDUAL_TASKS)
    if not joint_tasks and not indiv_tasks:
        return
    row_keys = _comparison_row_keys(scores_per_split, models, shots, sup_map,
                                     joint_tasks + indiv_tasks, llm_only=True)
    if not row_keys:
        return
    def _disp(m): return _display_model(m, model_aliases)
    name_w = max(len("model"), max(len(_disp(m)) for m, _ in row_keys))
    cell_w = 6
    block_w = cell_w * 3 + 2 * 2

    header1 = (
        "model".ljust(name_w) + "  k  "
        + "Joint".center(block_w) + "  "
        + "Individual".center(block_w))
    header2 = (
        " " * name_w + "     "
        + "  ".join(lbl.center(cell_w) for _, lbl in _TB_METRICS) + "  "
        + "  ".join(lbl.center(cell_w) for _, lbl in _TB_METRICS))
    line = "-" * max(len(header1), len(header2))

    print(f"\n  {title}")
    print("  " + line)
    print("  " + header1)
    print("  " + header2)
    print("  " + line)

    prev_model = None
    for (m, k) in row_keys:
        if prev_model is not None and m != prev_model:
            print("  " + "-" * len(header1))
        cells_j, cells_i = [], []
        for metric_key, _ in _TB_METRICS:
            v_j = (_mean_overall_f1(scores_per_split, m, k, metric_key, joint_tasks)
                   if joint_tasks else None)
            v_i = (_mean_overall_f1(scores_per_split, m, k, metric_key, indiv_tasks)
                   if indiv_tasks else None)
            cells_j.append(_fmt_f1(v_j).rjust(cell_w))
            cells_i.append(_fmt_f1(v_i).rjust(cell_w))
        row = (_disp(m).ljust(name_w) + " "
               + _display_k(k, sup_map.get(m)).rjust(3) + "  "
               + "  ".join(cells_j) + "  "
               + "  ".join(cells_i))
        print("  " + row)
        prev_model = m
    print("  " + line)


def _print_strategy_comparison_latex(label_suffix, scores_per_split, models,
                                       shots, sup_map, tasks, caption,
                                       model_aliases=None):
    joint_tasks = _filter_tasks(tasks, _JOINT_TASKS)
    indiv_tasks = _filter_tasks(tasks, _INDIVIDUAL_TASKS)
    if not joint_tasks and not indiv_tasks:
        return
    # LLMs only — supervised models don't train Individual variants
    # (ACH1 / E2ACH1), so the side-by-side isn't meaningful for them.
    row_keys = _comparison_row_keys(scores_per_split, models, shots, sup_map,
                                     joint_tasks + indiv_tasks, llm_only=True)
    if not row_keys:
        return
    def _disp(m): return _display_model(m, model_aliases)

    print()
    print("% Strategy comparison — Joint (JHE) vs Individual (IHE)")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{l c ccc ccc}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{\textbf{Model}}")
    print(r"& \multirow{2}{*}{\textbf{$k$}}")
    print(r"& \multicolumn{3}{c}{\textbf{Joint}}")
    print(r"& \multicolumn{3}{c}{\textbf{Individual}} \\")
    print(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    print(r"&  & \textbf{E} & \textbf{S} & \textbf{K}"
          r" & \textbf{E} & \textbf{S} & \textbf{K} \\")
    print(r"\midrule")

    prev_model = None
    for (m, k) in row_keys:
        if prev_model is not None and m != prev_model:
            print(r"\midrule")
        cells = [_esc_latex(_disp(m)), _display_k(k, sup_map.get(m))]
        for tlist_key, tlist in (("joint", joint_tasks), ("indiv", indiv_tasks)):
            for metric_key, _ in _TB_METRICS:
                v = (_mean_overall_f1(scores_per_split, m, k, metric_key, tlist)
                     if tlist else None)
                cells.append(_fmt_f1(v))
        print(" & ".join(cells) + r" \\")
        prev_model = m
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{" + caption + "}")
    print(r"\label{tab:analysis-" + label_suffix + "}")
    print(r"\end{table*}")


# ---------------------------------------------------------------------------
# TABLE 2 (grouped) — sub-causal factors only, exact F1 only, with a
# 2-level column header (parent main → its sub-factors). One value per
# sub-factor cell, so the whole table fits in a single tabular without
# splitting. Used in place of the generic _print_factor_table_* for tb2.
# ---------------------------------------------------------------------------

def _group_subs_by_parent(sub_factors, structure):
    """Return [(main_key, [sub_keys])] in structure order, dropping any
    parent that has no sub present in `sub_factors`."""
    main_to_subs = {m: list((structure or {}).get(m, []))
                    for m in (structure or {}).get("accident_report", [])}
    out = []
    for main in main_to_subs:
        subs_present = [s for s in main_to_subs[main] if s in sub_factors]
        if subs_present:
            out.append((main, subs_present))
    return out


def _print_subcausal_grouped_text(title, sub_factors, scores_per_split,
                                    models, shots, sup_map, tasks, structure,
                                    model_aliases=None, metric_key="exact"):
    """Plain-text sub-causal table — one cell per (model, kshot, sub-factor)
    under a single metric (default: exact). Sub-factors are grouped under
    their parent main in a 2-row header."""
    if not sub_factors or not models or not tasks:
        return
    groups = _group_subs_by_parent(sub_factors, structure)
    if not groups:
        return

    row_keys = []
    for m in sorted(models, key=lambda mm: _row_sort_key(mm, sup_map.get(mm))):
        is_sup = sup_map.get(m)
        for k in shots:
            if any((m, t, k) in scores_per_split for t in tasks):
                row_keys.append((m, k))
                if is_sup:
                    break
    if not row_keys:
        return

    def _disp(m): return _display_model(m, model_aliases)
    def _kshow(m, k): return _display_k(k, sup_map.get(m))
    name_w = max(len("model"), max(len(_disp(m)) for m, _ in row_keys))

    # Per-sub column width: F1 cell (~5 chars) or the sub label, whichever
    # is wider — keeps the parent header centered over its sub block.
    sub_widths = {s: max(5, len(_short(s)))
                  for _, subs in groups for s in subs}

    # Row 1: parent main names centered over their sub block. Block width =
    # sum of sub widths + 2-space gaps between subs.
    head1_blocks = []
    for main, subs in groups:
        block_w = sum(sub_widths[s] for s in subs) + 2 * (len(subs) - 1)
        head1_blocks.append(_short(main).center(max(block_w, len(_short(main)))))
        # Bump width if parent name was wider than its sub block.
        if len(_short(main)) > block_w:
            extra = len(_short(main)) - block_w
            sub_widths[subs[-1]] += extra
    header1 = (
        "model".ljust(name_w) + "  " + "k".rjust(3) + "  "
        + "  ".join(head1_blocks))

    # Row 2: sub-factor names.
    sub_cells = []
    for _, subs in groups:
        for s in subs:
            sub_cells.append(_short(s).rjust(sub_widths[s]))
    header2 = (
        " " * name_w + "  " + " " * 3 + "  "
        + "  ".join(["  ".join(_short(s).rjust(sub_widths[s]) for s in subs)
                      for _, subs in groups]))
    line = "-" * max(len(header1), len(header2))

    print(f"\n  {title}")
    print("  " + line)
    print("  " + header1)
    print("  " + header2)
    print("  " + line)

    prev_model = None
    for (m, k) in row_keys:
        if prev_model is not None and m != prev_model:
            print("  " + "-" * len(header1))
        cells = []
        for _, subs in groups:
            for s in subs:
                v = _mean_factor_f1(scores_per_split, m, k, s, metric_key, tasks)
                cells.append(_fmt_f1(v).rjust(sub_widths[s]))
        # Re-group sub cells by parent for the same "  " spacing as headers.
        parts = []
        idx = 0
        for _, subs in groups:
            parts.append("  ".join(cells[idx:idx + len(subs)]))
            idx += len(subs)
        row = (_disp(m).ljust(name_w) + "  "
               + _kshow(m, k).rjust(3) + "  "
               + "  ".join(parts))
        print("  " + row)
        prev_model = m
    print("  " + line)


def _print_subcausal_grouped_latex(title, label_suffix, sub_factors,
                                     scores_per_split, models, shots, sup_map,
                                     tasks, structure, caption,
                                     model_aliases=None, metric_key="exact"):
    if not sub_factors or not models or not tasks:
        return
    groups = _group_subs_by_parent(sub_factors, structure)
    if not groups:
        return
    row_keys = []
    for m in sorted(models, key=lambda mm: _row_sort_key(mm, sup_map.get(mm))):
        is_sup = sup_map.get(m)
        for k in shots:
            if any((m, t, k) in scores_per_split for t in tasks):
                row_keys.append((m, k))
                if is_sup:
                    break
    if not row_keys:
        return

    def _disp(m): return _display_model(m, model_aliases)
    def _kshow(m, k): return _display_k(k, sup_map.get(m))
    total_subs = sum(len(subs) for _, subs in groups)

    print()
    print(f"% {title}")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{2pt}")
    print(r"\renewcommand{\arraystretch}{0.95}")
    print(r"\resizebox{\textwidth}{!}{%")
    # Col-spec: Model + k + one col per sub-factor.
    print(r"\begin{tabular}{l c " + " ".join("c" for _ in range(total_subs)) + "}")
    print(r"\toprule")
    # Row 1: parent main names spanning their subs.
    print(r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{$k$}}")
    for main, subs in groups:
        print(r"& \multicolumn{" + str(len(subs)) + r"}{c}{\textbf{"
              + _esc_latex(_short(main)) + "}}")
    print(r" \\")
    # cmidrules under each parent block.
    col = 3  # Model=1, k=2, first sub=3
    for _, subs in groups:
        lo = col
        hi = col + len(subs) - 1
        print(r"\cmidrule(lr){" + f"{lo}-{hi}" + "}")
        col += len(subs)
    # Row 2: sub-factor names.
    sub_cells = []
    for _, subs in groups:
        for s in subs:
            sub_cells.append(_esc_latex(_short(s)))
    print(r" &  & " + " & ".join(sub_cells) + r" \\")
    print(r"\midrule")

    col_max_sub = {}
    for _, subs in groups:
        for s in subs:
            vals = [_mean_factor_f1(scores_per_split, m, k, s, metric_key, tasks)
                    for m, k in row_keys]
            valid = [v for v in vals if v is not None]
            col_max_sub[s] = max(valid) if valid else None

    prev_model = None
    for (m, k) in row_keys:
        if prev_model is not None and m != prev_model:
            print(r"\midrule")
        cells = [_esc_latex(_disp(m)), _kshow(m, k)]
        for _, subs in groups:
            for s in subs:
                v = _mean_factor_f1(scores_per_split, m, k, s, metric_key,
                                     tasks)
                fmt = _fmt_f1(v)
                mx = col_max_sub.get(s)
                if v is not None and mx is not None and v == mx:
                    fmt = _tex_bold(fmt)
                cells.append(fmt)
        print(" & ".join(cells) + r" \\")
        prev_model = m
    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\caption{" + caption + "}")
    print(r"\label{tab:analysis-" + label_suffix + "}")
    print(r"\end{table*}")


# ---------------------------------------------------------------------------
# TABLE (kept) — Task-level F1, COMBINED across all tasks.
#   Single table. Cols: model + (task × metric) groups. Rows: all models in
#   the run. Each cell is the model's best F1 across k-shots under that
#   (task, metric); operating-point k is dropped to keep the combined grid
#   readable.
# ---------------------------------------------------------------------------

def _task_level_grid(tasks, models, scores_by_metric, shots_by_task):
    """Build a {(model, task, metric_key) -> best_f1} grid for the combined
    table. `scores_by_metric` is dict[metric_label -> scores_dict]."""
    grid = {}
    for metric_label, scores in scores_by_metric.items():
        for task in tasks:
            best = _best_per_metric(scores, models, task, shots_by_task.get(task, []))
            for m in models:
                pair = best.get(m)
                grid[(m, task, metric_label)] = (None if pair is None else pair[1])
    return grid


def _model_avg_soft(models, tasks, scores_soft, shots_by_task):
    """Mean of each model's best-soft F1 across tasks (None when no task has
    a real F1). Used to sort the combined table by overall strength."""
    avg = {}
    for m in models:
        vals = []
        for t in tasks:
            best = _best_per_metric(scores_soft, [m], t, shots_by_task.get(t, []))
            pair = best.get(m)
            if pair is not None:
                vals.append(pair[1])
        avg[m] = (sum(vals) / len(vals)) if vals else None
    return avg


_METRIC_LABELS = [("exact", "Exact"), ("soft", "Soft"), ("kw", "Keyword")]


def _print_task_level_combined_text(tasks, models, scores_exact, scores_soft,
                                     scores_kw, shots_by_task):
    """Combined task-level F1: cols = task × metric, rows = models."""
    if not tasks or not models:
        return
    scores_by_metric = {"exact": scores_exact, "soft": scores_soft,
                        "kw": scores_kw}
    grid = _task_level_grid(tasks, models, scores_by_metric, shots_by_task)
    avg_soft = _model_avg_soft(models, tasks, scores_soft, shots_by_task)

    name_w = max(len("model"), max(len(m) for m in models))
    cell_w = 6  # "xx.xx"
    metric_w = cell_w

    # Header row 1: per-task group label (centered over its 3 metric cells)
    task_block_w = metric_w * 3 + 2 * 2  # 3 cells + 2 inter-cell gaps
    header1 = "model".ljust(name_w) + "  " + "  ".join(
        t.center(task_block_w) for t in tasks)
    # Header row 2: metric sub-labels under each task block
    header2 = " " * name_w + "  " + "  ".join(
        "  ".join(lbl.center(metric_w) for _, lbl in _METRIC_LABELS)
        for _ in tasks)
    line = "-" * max(len(header1), len(header2))

    print(f"\n  TASK-LEVEL F1  (best k per (model, task, metric))")
    print("  " + line)
    print("  " + header1)
    print("  " + header2)
    print("  " + line)

    def _sort_key(m):
        a = avg_soft.get(m)
        return (-(a if a is not None else -1), _natural_key(m))

    for m in sorted(models, key=_sort_key):
        row = m.ljust(name_w) + "  " + "  ".join(
            "  ".join(_fmt_f1(grid.get((m, t, key))).rjust(metric_w)
                       for key, _ in _METRIC_LABELS)
            for t in tasks)
        print("  " + row)
    print("  " + line)


def _print_task_level_combined_latex(tasks, models, scores_exact, scores_soft,
                                      scores_kw, shots_by_task):
    if not tasks or not models:
        return
    scores_by_metric = {"exact": scores_exact, "soft": scores_soft,
                        "kw": scores_kw}
    grid = _task_level_grid(tasks, models, scores_by_metric, shots_by_task)
    avg_soft = _model_avg_soft(models, tasks, scores_soft, shots_by_task)
    n_metrics = len(_METRIC_LABELS)

    print()
    print(f"% Combined task-level F1 — tasks: {', '.join(tasks)}")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{4pt}")
    # 1 model col + 3 metric cells per task.
    print(r"\begin{tabular}{l" + "c" * (n_metrics * len(tasks)) + "}")
    print(r"\toprule")
    # Row 1: task group headers, each spanning n_metrics cols.
    print(r"\multirow{2}{*}{\textbf{Model}} & "
          + " & ".join(r"\multicolumn{" + str(n_metrics) + r"}{c}{\textbf{"
                        + _esc_latex(t) + "}}" for t in tasks) + r" \\")
    print(" ".join(
        r"\cmidrule(lr){" + f"{2 + i * n_metrics}-{1 + (i + 1) * n_metrics}" + "}"
        for i in range(len(tasks))))
    # Row 2: metric sub-labels under each task.
    print(" & " + " & ".join(
        r"\textbf{" + lbl + "}" for _ in tasks for _, lbl in _METRIC_LABELS)
          + r" \\")
    print(r"\midrule")

    def _sort_key(m):
        a = avg_soft.get(m)
        return (-(a if a is not None else -1), _natural_key(m))

    col_max_task = {}
    for t in tasks:
        for key, _ in _METRIC_LABELS:
            vals = [grid.get((m, t, key)) for m in models]
            valid = [v for v in vals if v is not None]
            col_max_task[(t, key)] = max(valid) if valid else None

    for m in sorted(models, key=_sort_key):
        cells = [_esc_latex(m)]
        for t in tasks:
            for key, _ in _METRIC_LABELS:
                v = grid.get((m, t, key))
                fmt = _fmt_f1(v)
                mx = col_max_task.get((t, key))
                if v is not None and mx is not None and v == mx:
                    fmt = _tex_bold(fmt)
                cells.append(fmt)
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Task-level F1: each model's best F1 across $k$-shots, "
          r"reported per task under exact (sim=1.0), soft (sim=0.8), and "
          r"keyword matching. Rows sorted by mean Soft F1 across tasks.}")
    print(r"\label{tab:analysis-task-combined}")
    print(r"\end{table*}")


# ---------------------------------------------------------------------------
# TABLE 2 — Few-shot scaling, COMBINED across tasks.
#   One table per metric (exact / soft / keyword). Cols: Task | Model | k=…
#   Rows: every (task, LLM model) pair, grouped by task with a separator
#   between groups. Cells show F1; `--` when the model didn't run at that k.
# ---------------------------------------------------------------------------

def _all_shots(shots_by_task):
    """Union of every k-shot across tasks, sorted numerically."""
    union = set()
    for ks in shots_by_task.values():
        union.update(ks)
    return sorted(union, key=int)


def _print_scaling_combined_text(tasks, llm_models_by_task, score_dict,
                                  metric_label, shots_by_task):
    """Plain-text scaling table — rows = (task, model)."""
    shots = _all_shots(shots_by_task)
    if not shots:
        return
    # Drop tasks where no LLM has any entry under this metric.
    tasks_present = [t for t in tasks if llm_models_by_task.get(t)
                     and any(_scope_f1(score_dict, (m, t, k)) is not None
                              for m in llm_models_by_task[t]
                              for k in shots_by_task.get(t, []))]
    if not tasks_present:
        return

    task_w = max(len("task"), max(len(t) for t in tasks_present))
    name_w = max(len("model"), max(len(m)
                                    for t in tasks_present
                                    for m in llm_models_by_task[t]))
    cell_w = 6  # "xx.xx"

    header = (
        "task".ljust(task_w) + "  "
        + "model".ljust(name_w) + "  "
        + "  ".join(f"k={k}".rjust(cell_w) for k in shots))
    line = "-" * len(header)

    print(f"\n  FEW-SHOT SCALING [{metric_label}]")
    print("  " + line)
    print("  " + header)
    print("  " + line)
    for i, t in enumerate(tasks_present):
        if i > 0:
            print("  " + "-" * len(header))  # visual break between tasks
        for m in sorted(llm_models_by_task[t], key=_natural_key):
            row = (
                t.ljust(task_w) + "  "
                + m.ljust(name_w) + "  "
                + "  ".join(_fmt_f1(_scope_f1(score_dict, (m, t, k))).rjust(cell_w)
                             for k in shots))
            print("  " + row)
    print("  " + line)


def _print_scaling_combined_latex(tasks, llm_models_by_task, score_dict,
                                   metric_label, shots_by_task):
    shots = _all_shots(shots_by_task)
    if not shots:
        return
    tasks_present = [t for t in tasks if llm_models_by_task.get(t)
                     and any(_scope_f1(score_dict, (m, t, k)) is not None
                              for m in llm_models_by_task[t]
                              for k in shots_by_task.get(t, []))]
    if not tasks_present:
        return

    print()
    print(f"% Combined few-shot scaling [{metric_label}] — tasks: "
          f"{', '.join(tasks_present)}")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{5pt}")
    print(r"\begin{tabular}{ll" + "c" * len(shots) + "}")
    print(r"\toprule")
    print(r"\textbf{Task} & \textbf{Model} & "
          + " & ".join(r"\textbf{$k$=" + str(k) + "}" for k in shots) + r" \\")
    print(r"\midrule")
    col_max_scaling = {}
    for t in tasks_present:
        for k in shots:
            vals = [_scope_f1(score_dict, (m, t, k)) for m in llm_models_by_task[t]]
            valid = [v for v in vals if v is not None]
            col_max_scaling[(t, k)] = max(valid) if valid else None

    for i, t in enumerate(tasks_present):
        if i > 0:
            print(r"\midrule")
        for m in sorted(llm_models_by_task[t], key=_natural_key):
            row = [_esc_latex(t), _esc_latex(m)]
            for k in shots:
                v = _scope_f1(score_dict, (m, t, k))
                fmt = _fmt_f1(v)
                mx = col_max_scaling.get((t, k))
                if v is not None and mx is not None and v == mx:
                    fmt = _tex_bold(fmt)
                row.append(fmt)
            print(" & ".join(row) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Few-shot scaling (" + metric_label
          + "): F1 by LLM and $k$-shot size, across tasks. Cells are `--' "
            r"when the model didn't run at that $k$ for that task.}")
    print(r"\label{tab:analysis-scaling-" + metric_label.lower() + "}")
    print(r"\end{table}")


# ---------------------------------------------------------------------------
# TABLE 3 — Field-level F1, COMBINED across tasks.
#   Single table. Cols: factor + (task × metric) groups (Exact / Soft /
#   Keyword under each task). Rows: each factor (with main factors in bold
#   and sub-factors indented). Each cell = peak F1 across all (model,
#   k-shot) for that (factor, task, metric) — the "what's the best any
#   system achieves" view.
# ---------------------------------------------------------------------------

def _best_field_f1(score_dict, factor, task):
    """Max F1 across every (model, k-shot) entry for a (factor, task)
    under the supplied scores dict. Returns None when no entry has a real
    numeric F1 for that factor on that task."""
    best = None
    for key in score_dict:
        if key[1] != task:
            continue
        v = _factor_f1(score_dict, key, factor)
        if v is None:
            continue
        if best is None or v > best:
            best = v
    return best


def _collect_combined_factor_order(tasks, scores_by_metric, structure,
                                    force_factors=None):
    """Union the factor sets across all metrics & tasks, then order via
    the same main → subs → leftovers logic as the cross-model table."""
    combined = {}
    for sc_dict in scores_by_metric.values():
        combined.update(sc_dict)
    return _collect_factor_order(combined, structure, force_factors=force_factors)


def _print_field_combined_text(tasks, scores_exact, scores_soft, scores_kw,
                                structure, force_factors=None,
                                classification_keys=None):
    scores_by_metric = [("exact", scores_exact),
                        ("soft", scores_soft),
                        ("kw", scores_kw)]
    factors, is_main, is_sub = _collect_combined_factor_order(
        tasks, dict(scores_by_metric), structure,
        force_factors=force_factors)
    if not factors or not tasks:
        return

    fac_w = max(len("field"), max(len(_decorate(f, is_main, is_sub))
                                   for f in factors))
    metric_w = 6
    task_block_w = metric_w * 3 + 2 * 2

    header_top = "field".ljust(fac_w) + "  " + "  ".join(
        t.center(task_block_w) for t in tasks)
    header_bot = " " * fac_w + "  " + "  ".join(
        "  ".join(lbl.center(metric_w) for _, lbl in _METRIC_LABELS)
        for _ in tasks)
    line = "-" * max(len(header_top), len(header_bot))

    print(f"\n  FIELD-LEVEL F1  (peak across models & $k$, per task × metric)")
    print("  " + line)
    print("  " + header_top)
    print("  " + header_bot)
    print("  " + line)

    def _emit_factor_row(f):
        is_cls = f in (classification_keys or ())
        cells = []
        for t in tasks:
            for metric_key, _ in _METRIC_LABELS:
                sc_dict = dict(scores_by_metric)[metric_key]
                if is_cls and metric_key != "exact":
                    v = None
                else:
                    v = _best_field_f1(sc_dict, f, t) if sc_dict else None
                cells.append(_fmt_f1(v).rjust(metric_w))
            cells.append("|")  # visual separator (replaced below)
        # Drop trailing "|" sentinel and join with task-block separator " | "
        # — actually simpler to just join with 2-space gap inside task, and
        # 2-space gap between tasks.
        out = []
        idx = 0
        for t in tasks:
            triple = "  ".join(
                cells[idx + j].split("|")[0] for j in range(3))
            out.append(triple)
            idx += 4
        return _decorate(f, is_main, is_sub).ljust(fac_w) + "  " + "  ".join(out)

    # Group output: emit a blank-line separator before each new main /
    # leftover (matches the \midrule grouping in the LaTeX template).
    for i, f in enumerate(factors):
        # A "group break" precedes every non-sub factor after the first row.
        if i > 0 and f not in is_sub:
            print("  " + "-" * len(header_top))
        print("  " + _emit_factor_row(f))
    print("  " + line)


def _print_field_combined_latex(tasks, scores_exact, scores_soft, scores_kw,
                                 structure, force_factors=None,
                                 classification_keys=None):
    scores_by_metric = [("exact", scores_exact),
                        ("soft", scores_soft),
                        ("kw", scores_kw)]
    factors, is_main, is_sub = _collect_combined_factor_order(
        tasks, dict(scores_by_metric), structure,
        force_factors=force_factors)
    if not factors or not tasks:
        return
    n_metrics = len(_METRIC_LABELS)

    print()
    print(f"% Combined field-level F1 — tasks: {', '.join(tasks)}")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\renewcommand{\arraystretch}{1.08}")
    print(r"\begin{tabular}{l " + " ".join("ccc" for _ in tasks) + "}")
    print(r"\toprule")
    # Row 1: task headers
    print(r"\multirow{2}{*}{\textbf{Field}}")
    for t in tasks:
        print(r"& \multicolumn{" + str(n_metrics)
              + r"}{c}{\textbf{" + _esc_latex(t) + "}}")
    print(r" \\")
    # cmidrules under each task block
    for i in range(len(tasks)):
        lo = 2 + i * n_metrics
        hi = 1 + (i + 1) * n_metrics
        print(r"\cmidrule(lr){" + f"{lo}-{hi}" + "}")
    # Row 2: metric sub-labels under each task
    print(" ".join([" "] + ["& " + " & ".join(lbl for _, lbl in _METRIC_LABELS)
                             for _ in tasks]) + r" \\")
    print(r"\midrule")

    def _row_label(f):
        if f in is_main:
            return r"\textbf{" + _esc_latex(_short(f)) + "}"
        if f in is_sub:
            return r"\quad " + _esc_latex(_short(f))
        return r"\textbf{" + _esc_latex(_short(f)) + "}"

    sc_map = dict(scores_by_metric)
    col_max_field = {}
    for t in tasks:
        for metric_key, _ in _METRIC_LABELS:
            vals = [None if (f in (classification_keys or ()) and metric_key != "exact")
                    else _best_field_f1(sc_map[metric_key], f, t) for f in factors]
            valid = [v for v in vals if v is not None]
            col_max_field[(t, metric_key)] = max(valid) if valid else None

    for i, f in enumerate(factors):
        # Match the template: \midrule before each new main / leftover.
        if i > 0 and f not in is_sub:
            print(r"\midrule")
        cells = [_row_label(f)]
        is_cls = f in (classification_keys or ())
        for t in tasks:
            for metric_key, _ in _METRIC_LABELS:
                sc_dict = sc_map[metric_key]
                if is_cls and metric_key != "exact":
                    v = None
                else:
                    v = _best_field_f1(sc_dict, f, t) if sc_dict else None
                fmt = _fmt_f1(v)
                mx = col_max_field.get((t, metric_key))
                if v is not None and mx is not None and v == mx:
                    fmt = _tex_bold(fmt)
                cells.append(fmt)
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Field-level F1 across tasks under exact, soft, and "
          r"keyword matching. Each cell is the peak F1 across all models and "
          r"$k$-shots for that (field, task, metric). Bold rows indicate "
          r"main causal-factor fields; indented rows indicate their "
          r"sub-factors.}")
    print(r"\label{tab:field_task_metric_f1}")
    print(r"\end{table*}")


# ---------------------------------------------------------------------------
# Charts — few-shot scaling plots (one figure per metric, 2x2 subplots).
# ---------------------------------------------------------------------------

# Color/marker pairs cycle through a colorblind-safe-ish palette. Matches the
# example chart's blue/orange/green/purple ordering on the most common
# 4-LLM run; extra models pick up additional combos.
_PLOT_STYLES = [
    ('^', '#1f77b4'),   # blue triangle
    ('D', '#ff7f0e'),   # orange diamond
    ('s', '#2ca02c'),   # green square
    ('o', '#9467bd'),   # purple circle
    ('v', '#d62728'),   # red triangle-down
    ('P', '#8c564b'),   # brown plus
    ('*', '#e377c2'),   # pink star
    ('X', '#7f7f7f'),   # gray X
]

_METRIC_PRETTY = {
    "exact": "Exact", "soft": "Soft", "kw": "Keyword", "keyword": "Keyword",
}


def _render_scaling_chart(tasks, llm_models_by_task, scores_per_split,
                           shots_by_task, output_path, model_aliases=None):
    """Save ONE figure (1×3 grid of subplots, one per metric) showing
    few-shot scaling for every LLM model. Y values are F1 sum-averaged
    across (5 splits × N tasks in scope), matching table-1's semantics.

    Extraction factors only — accident_type and any other classification
    factor are excluded (`_mean_overall_f1` reads `factors.overall`, which
    already has accident_type stripped for E2 tasks; classification
    factors are skip_columns'd out of `overall` entirely for every task).

    Skipped without error when matplotlib isn't installed — analysis still
    produces text + LaTeX. Output goes to `output_path` (a .png path)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend; safe on headless boxes
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("⚠️  matplotlib not installed — skipping scaling chart")
        return

    # Tasks that actually contributed per-split data for some LLM.
    tasks_present = [
        t for t in tasks
        if llm_models_by_task.get(t)
        and any((m, t, k) in scores_per_split
                for m in llm_models_by_task[t]
                for k in shots_by_task.get(t, []))]
    if not tasks_present:
        return

    # Consistent legend order across subplots: union of LLMs in any
    # contributing task, sorted by natural key.
    all_llms = sorted({m for t in tasks_present for m in llm_models_by_task[t]},
                      key=_natural_key)
    style_map = {m: _PLOT_STYLES[i % len(_PLOT_STYLES)]
                 for i, m in enumerate(all_llms)}
    aliases = model_aliases or {}
    def _disp(m): return aliases.get(m, _VARIANT_SUFFIX_RE.sub("", m))

    # Union of k-shots across the tasks that contributed, sorted numerically.
    shots = sorted({k for t in tasks_present for k in shots_by_task.get(t, [])},
                   key=int)
    if not shots:
        return
    k_vals = [int(k) for k in shots]

    # Subplot titles use full descriptive names. Suptitle removed — the
    # caller's caption documents what the figure shows.
    metrics = [
        ("exact", "Exact String Match"),
        ("soft",  "Soft String Match"),
        ("kw",    "Keyword Match"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for i, (metric_key, pretty) in enumerate(metrics):
        ax = axes[i]
        for m in all_llms:
            xs, ys = [], []
            for k_str, k_int in zip(shots, k_vals):
                v = _mean_overall_f1(scores_per_split, m, k_str,
                                      metric_key, tasks_present)
                if v is None:
                    continue
                xs.append(k_int)
                ys.append(v)
            if not xs:
                continue
            marker, color = style_map[m]
            ax.plot(xs, ys, marker=marker, color=color, label=_disp(m),
                    linewidth=1.6, markersize=7)
        ax.set_title(pretty, fontsize=22, fontweight="bold")
        ax.set_xlabel(r"$k$", fontsize=20)
        if i == 0:
            ax.set_ylabel("F1 score", fontsize=20)
        ax.set_ylim(0, 90)
        ax.set_yticks(range(0, 91, 10))
        ax.set_xticks(k_vals)
        # Bump tick labels too so axis numbers stay readable when the
        # figure is shrunk to column width in the paper.
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Shared legend below the figure — pull handles from the first subplot
    # that has them (some subplots may be empty when a metric has no data).
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        # ncol=4 wraps 5+ models onto two legend rows; anchor it a bit below
        # the axes (not right at y=0) so the upper row clears the "$k$"
        # xlabel instead of overlapping it, and reserve a matching bottom
        # margin in tight_layout below.
        fig.legend(handles, labels,
                    loc="lower center", ncol=min(len(handles), 4),
                    bbox_to_anchor=(0.5, -0.06),
                    fontsize=18, frameon=False)

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    # PDF is vector — DPI is irrelevant for the lines/text, but kept for
    # any raster sub-elements (none today). `bbox_inches="tight"` trims
    # the page to the figure extent including the bottom legend.
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_analysis_tables(preds_golds, skip_unions=None, *, structure=None,
                           classes_map=None, norm=False, is_supervised_map=None,
                           model_aliases=None, log_path=None):
    """Build and print all three analysis tables (text + LaTeX), once per
    dataset.

    `preds_golds` is keyed by (model, task, dataset, shot) — split apart by
    dataset first (same helper `render_cross_model_comparison` uses) so two
    datasets that happen to share (model, task, shot) never get pooled into
    one F1. Each dataset gets its own full report (and its own log/tex/chart
    files when `log_path` is set).
    """
    if not preds_golds:
        return

    from main.progress_analyzer import (split_preds_golds_by_dataset,
                                        per_dataset_log_path)
    by_dataset = split_preds_golds_by_dataset(preds_golds)
    for dataset_key in sorted(by_dataset):
        logger.info("\n%s", f" DATASET: {dataset_key} ".center(78, "#"))
        _render_analysis_tables_one(
            by_dataset[dataset_key],
            {k: v for k, v in skip_unions.items() if k[1] == dataset_key}
            if skip_unions else skip_unions,
            structure=structure, classes_map=classes_map, norm=norm,
            is_supervised_map=is_supervised_map, model_aliases=model_aliases,
            log_path=per_dataset_log_path(log_path, dataset_key),
            dataset_key=dataset_key,
        )


def _render_analysis_tables_one(preds_golds, skip_unions=None, *, structure=None,
                                classes_map=None, norm=False, is_supervised_map=None,
                                model_aliases=None, log_path=None,
                                dataset_key=None):
    """Build and print all three analysis tables (text + LaTeX) for ONE
    dataset's slice.

    Mirrors `render_cross_model_comparison`'s signature so it can be called
    from the same place in `run.py` with the same inputs. `log_path` controls
    file-output: when a directory or extensionless path, writes
    `analysis_<ts>.log` (+ sibling `.tex`); when a path with a suffix, uses
    it as the .log and derives the .tex from `os.path.splitext`.
    """
    if not preds_golds:
        return

    # Compute the three score views. Reuses the same skip union logic as the
    # cross-model rescore so cell values agree with the cross-model log.
    (scores_exact, scores_soft, scores_kw, scores_per_split,
     sup_map) = _build_all_scores(
        preds_golds, skip_unions, structure, classes_map, norm=norm)
    if is_supervised_map:
        sup_map.update(is_supervised_map)

    # Classification factors per the key map (accident_type, and any other
    # closed-vocabulary label) — only their exact-match F1 is meaningful.
    # tb1/tb4 (per-factor column blocks) drop soft/keyword to a single
    # Exact column for these via `_metrics_for_factor`; the field-level
    # heatmap (factors as rows sharing a fixed task×metric column axis,
    # so a row can't have fewer columns than its neighbors) instead blanks
    # those cells to "--".
    classification_keys = set(classes_map.keys()) if classes_map else set()

    # Collect models / tasks / shots from the pooled keys.
    tasks = sorted({t for (_, t, _) in preds_golds})
    all_models = sorted({m for (m, _, _) in preds_golds}, key=_natural_key)
    llm_models = [m for m in all_models if not sup_map.get(m)]
    shots_by_task = {}
    for (_, t, k) in preds_golds:
        shots_by_task.setdefault(t, set()).add(k)
    shots_by_task = {t: sorted(ks, key=int)
                     for t, ks in shots_by_task.items()}

    # Decide whether accident_type stays as a row across every per-task
    # field-level heatmap. Only force it when AT LEAST ONE task in scope
    # actually scores it (E2*). If the run is ACH/ACH1-only, the row would
    # be uniformly "--" and gets dropped instead.
    def _any_task_scores(scores_map, factor):
        for sc in scores_map.values():
            node = ((sc.get("factors", {}).get("per_factor") or {})
                    .get(factor) or {}).get("overall", {})
            v = node.get("f1")
            if v is not None and not isinstance(v, str):
                return True
        return False
    show_acc_type = (_any_task_scores(scores_exact, "accident_type")
                     or _any_task_scores(scores_soft, "accident_type")
                     or _any_task_scores(scores_kw, "accident_type"))
    force_factors = {"accident_type"} if show_acc_type else set()

    # Pre-compute LLM models per task / shared maps.
    llm_models_by_task = {}
    task_models_by_task = {}
    for task in tasks:
        tm = sorted({m for (m, t, _) in preds_golds if t == task},
                     key=_natural_key)
        task_models_by_task[task] = tm
        llm_models_by_task[task] = [m for m in tm if not sup_map.get(m)]

    # k-shot rows for tb1/tb2 — keep the default {0, 5, 30} unless an entry
    # only has a subset of those available.
    tb_shots = [k for k in _ANALYSIS_KSHOTS
                if any(k in shots_by_task.get(t, []) for t in tasks)]

    # Causal vs sub-causal factor split — drives tb1 and tb2 respectively.
    # Only include factors with at least one real numeric F1 in some score
    # (filters out skip_columns like accident_report / id, whose per_factor
    # entries land as f1='NaN' strings).
    all_seen_factors = set()
    for sc_dict in (scores_exact, scores_soft, scores_kw):
        for sc in sc_dict.values():
            for fk, node in (sc.get("factors", {}).get("per_factor") or {}).items():
                v = (node.get("overall") or {}).get("f1")
                if v is not None and not isinstance(v, str):
                    all_seen_factors.add(fk)
    # accident_type is shown as a row whenever ANY task in scope scored it
    # (matches the heatmap's behavior).
    if force_factors and "accident_type" in force_factors:
        all_seen_factors.add("accident_type")
    causal_factors, sub_factors = _split_main_and_sub_factors(
        structure, all_seen_factors)
    # accident_type is a classification label, not an extraction target —
    # it sits in `strict_keys` and is stripped from `factors.overall` for
    # E2 tasks via `_score_one_combo` (see per-record scoring), but
    # `per_factor.accident_type` is left intact, and any task in scope that
    # actually predicts it (JHE/IHE — both are E2* end-to-end variants)
    # should surface that number in the causal-factor table too. It lands
    # in `causal_factors` as a leftover (no parent in `structure`); hoist it
    # to the front, matching `_collect_factor_order`'s convention for the
    # per-task field-level heatmap so it's easy to find in both tables.
    if "accident_type" in causal_factors:
        causal_factors.remove("accident_type")
        causal_factors.insert(0, "accident_type")

    # --- TEXT ---
    text_buf = io.StringIO()
    with contextlib.redirect_stdout(text_buf):
        print()
        print("=" * 78)
        report_title = (f" ANALYSIS REPORT — dataset={dataset_key} "
                        if dataset_key else " ANALYSIS REPORT ")
        print(report_title.center(78, "="))
        print("=" * 78)

        # TABLE 1: causal factors. Rows = (model, kshot). Cells = F1 mean
        # across (5 splits × tasks in scope). The task list is interpolated
        # so the title makes clear which tasks contributed (e.g. ACH only
        # vs ACH + E2ACH).
        task_phrase = "task " + tasks[0] if len(tasks) == 1 else \
            "tasks " + ", ".join(tasks)
        _print_factor_table_text(
            f"CAUSAL FACTORS  [F1 mean over splits × {task_phrase}]",
            causal_factors, scores_per_split, all_models, tb_shots,
            sup_map, tasks, structure, model_aliases=model_aliases,
            classification_keys=classification_keys)
        # TABLE 2: sub-causal factors — exact F1 only, grouped under
        # their parent main in a 2-level header. Only one cell per
        # sub-factor → fits a single tabular.
        _print_subcausal_grouped_text(
            f"SUB-CAUSAL FACTORS  [exact F1, mean over splits × {task_phrase}]",
            sub_factors, scores_per_split, all_models, tb_shots,
            sup_map, tasks, structure, model_aliases=model_aliases)

        # TABLE 3: setting comparison (Accident-Conditioned vs End-to-End)
        _print_setting_comparison_text(
            "SETTING COMPARISON  [Accident-Conditioned vs End-to-End]",
            scores_per_split, all_models, tb_shots,
            sup_map, tasks, model_aliases=model_aliases)
        # TABLE 4: strategy comparison (Joint vs Individual). Joint pools
        # ACH + E2ACH, Individual pools ACH1 + E2ACH1. E2 overalls already
        # have accident_type stripped (see _score_one_combo), so both
        # settings combine on equal footing. LLMs only.
        _print_strategy_comparison_text(
            "STRATEGY COMPARISON  [Joint (JHE) vs Individual (IHE)]",
            scores_per_split, all_models, tb_shots,
            sup_map, tasks, model_aliases=model_aliases)

        # Field-level peak-F1 heatmap (unchanged) — single table, cols =
        # task × metric, cell = peak F1 across all models & k-shots.
        _print_field_combined_text(
            tasks, scores_exact, scores_soft, scores_kw, structure,
            force_factors=force_factors,
            classification_keys=classification_keys)
    text_output = text_buf.getvalue()
    logger.info("%s", text_output.rstrip("\n"))

    # --- LaTeX ---
    tex_buf = io.StringIO()
    with contextlib.redirect_stdout(tex_buf):
        # TABLE 1 + TABLE 2 — causal / sub-causal factor F1, per (model, k).
        # Task phrase: explicit ("JHE" vs "JHE, IHE") so
        # readers don't have to guess what tasks are in scope.
        tex_task_phrase = (
            f"task {_esc_latex(tasks[0])}" if len(tasks) == 1
            else "tasks " + ", ".join(_esc_latex(t) for t in tasks))
        _print_factor_table_latex(
            f"Causal factors — F1 mean over splits × {', '.join(tasks)}",
            "causal-factors",
            causal_factors, scores_per_split, all_models, tb_shots,
            sup_map, tasks, structure,
            caption=(r"Per-(model, $k$) F1 on causal factors, "
                     r"sum-averaged across the 5 cross-validation splits "
                     r"and across " + tex_task_phrase + r". "
                     r"Columns E / S / K denote exact (sim=1.0), soft "
                     r"(sim=0.8), and keyword matching. Supervised models "
                     r"grouped first; LLMs below. Classification factors "
                     r"(e.g. accident\_type) only report Exact — soft/"
                     r"keyword match isn't meaningful for a label."),
            model_aliases=model_aliases,
            classification_keys=classification_keys)
        _print_subcausal_grouped_latex(
            f"Sub-causal factors — exact F1 over splits × {', '.join(tasks)}",
            "subcausal-factors",
            sub_factors, scores_per_split, all_models, tb_shots,
            sup_map, tasks, structure,
            caption=(r"Per-(model, $k$) exact-match F1 on sub-causal "
                     r"factors, sum-averaged across the 5 cross-validation "
                     r"splits and across " + tex_task_phrase + r". "
                     r"Sub-factors are grouped under their parent main "
                     r"factor in the column header. Supervised models "
                     r"grouped first; LLMs below."),
            model_aliases=model_aliases)

        # TABLE 3 (LaTeX): setting comparison.
        _print_setting_comparison_latex(
            "setting-comparison",
            scores_per_split, all_models, tb_shots,
            sup_map, tasks,
            caption=(r"Model-level comparison between accident-conditioned "
                     r"and end-to-end settings. For each model, LLM results "
                     r"are reported at representative shot settings "
                     r"$k=0,5,30$, while supervised models are reported "
                     r"once. Scores are averaged across the corresponding "
                     r"extraction strategies: JHE/IHE for accident-"
                     r"conditioned extraction and E2JHE/E2IHE for end-to-"
                     r"end extraction. Columns E / S / K denote exact "
                     r"(sim=1.0), soft (sim=0.8), and keyword matching."),
            model_aliases=model_aliases)
        # TABLE 4 (LaTeX): strategy comparison.
        _print_strategy_comparison_latex(
            "strategy-comparison",
            scores_per_split, all_models, tb_shots,
            sup_map, tasks,
            caption=(r"Comparison between joint (JHE) and individual (IHE) "
                     r"hierarchical extraction (LLMs only — supervised "
                     r"models don't train the individual variants). "
                     r"accident\_type is excluded from overall F1 so both "
                     r"settings contribute on equal footing. Columns "
                     r"E / S / K denote exact (sim=1.0), soft (sim=0.8), "
                     r"and keyword matching. Reported at representative "
                     r"shot settings $k=0,5,30$."),
            model_aliases=model_aliases)

        # Combined field-level F1 — kept; uses pooled scores for the peak.
        _print_field_combined_latex(
            tasks, scores_exact, scores_soft, scores_kw, structure,
            force_factors=force_factors,
            classification_keys=classification_keys)
    tex_output = tex_buf.getvalue()

    if log_path:
        lp = log_path
        if os.path.isdir(lp) or not os.path.splitext(lp)[1]:
            os.makedirs(lp, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            lp = os.path.join(lp, f"analysis_{ts}.log")
        else:
            os.makedirs(os.path.dirname(lp) or ".", exist_ok=True)
        with open(lp, "w", encoding="utf-8") as f:
            f.write(text_output)
        logger.info("\U0001f4dd Analysis tables written to %s", lp)
        if tex_output.strip():
            tex_lp = os.path.splitext(lp)[0] + ".tex"
            with open(tex_lp, "w", encoding="utf-8") as f:
                f.write(tex_output)
            logger.info("\U0001f4dd LaTeX analysis tables written to %s", tex_lp)

        # Few-shot scaling chart — ONE vector PDF with 3 metric subplots
        # (Exact / Soft / Keyword), sibling to the .log. Uses the same
        # `analysis_<ts>_fewshot_scaling.pdf` stem so chart history stays
        # in sync with the per-run log/tex artifacts.
        stem = os.path.splitext(lp)[0]
        chart_path = f"{stem}_fewshot_scaling.pdf"
        _render_scaling_chart(
            tasks, llm_models_by_task, scores_per_split,
            shots_by_task, chart_path,
            model_aliases=model_aliases)
        if os.path.exists(chart_path):
            logger.info("\U0001f4ca Few-shot scaling chart → %s", chart_path)
