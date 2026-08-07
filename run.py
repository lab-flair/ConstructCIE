import json
import logging
import sys
import os
import glob as _glob
import itertools
import argparse
import datetime

from main.progress_analyzer import (
    check_combo,
    render_cross_model_comparison,
    supervised_status,
)
from main.job_balancer import bucket_shots_balanced, parse_slurm_time_to_hours
from main.job_manager import (
    create_sbatch_script,
    create_grouped_sbatch_parts,
    create_bucket_wrapper,
    create_master_submit,
    make_log_filename,
)
from main.job_runner import run_command, build_llm_cmd_list

from main.data_analyzer import render_analysis_tables

# --- CONFIGURATION & TEMPLATES ---

logger = logging.getLogger(__name__)

main_dir = os.path.dirname(os.path.realpath(__file__))

CONFIG = {}
SCRATCH_PATH = ""  # resolved after config load + -u/--user arg
GPU_OVERRIDE = None  # set by -g/--gpus; short-circuits gpu_mappings lookup
CONFIG_VARIANT = None  # set by -C/--config; short-circuits config_variant lookup
SIM_THRESHOLD_OVERRIDE = None  # set by --ac_sim_threshold; short-circuits ac_similarity_threshold lookup

_CONFIG_MAP = {
    0: "global_mapping.json",
    1: "global_mapping_1.json",
    2: "global_mapping_2.json",
    3: "global_mapping_3.json",
}

# "global" pointer fields that downstream code dereferences as file PATHS
# (get_config_details/get_supervised_config_details/collect_skip_unions all
# do `smart_path(CONFIG.get("key_mapping", ""))` then open() it themselves).
# Every other "global" pointer (currently just "model_mapping") is loaded
# here and flattened straight into CONFIG, mirroring how those fields sat
# inline in the old global_mapping_1..4.json files.
_GLOBAL_PATH_FIELDS = {"key_mapping", "column_definition"}


def _expand_pointer_config(raw, config_path):
    """Materialize a global_mapping_5-style pointer config into the same
    flat CONFIG shape the rest of this module already understands.

    raw["global"]  -> {field: filename}   filenames live under main/global/.
    raw["args"]    -> flat scalar overrides (scratch_path, log_dir, ...),
                       plus a "gpu_mapping" pointer naming which
                       main/args/gpu_mapping_<N>.json to load.
    raw["configs"] -> per-model default -C variant (e.g. {"TagPrime-C": "c4"}),
                       stashed for `_default_variant_for_model`.

    The GPU-tunable subset (gpu_mappings, gpu_requirements, eval_batch_size,
    few_shot_buckets/weights, gpu_partition) lives separately under
    main/args/. Which file to load is picked by editing the "gpu_mapping"
    key inside this config's "args" section — not a CLI flag — and is
    merged last so it wins over any duplicate key already pulled in from
    model_mapping.json.
    """
    main_root = os.path.dirname(config_path)  # .../main
    global_dir = os.path.join(main_root, "global")
    merged = {}
    for field, fname in (raw.get("global") or {}).items():
        if field in _GLOBAL_PATH_FIELDS:
            merged[field] = f"main/global/{fname}"
            continue
        fpath = os.path.join(global_dir, fname)
        try:
            with open(fpath) as f:
                content = json.load(f)
        except FileNotFoundError:
            logger.warning("⚠️  global mapping file not found: %s", fpath)
            continue
        if isinstance(content, dict):
            merged.update(content)
        else:
            merged[field] = content

    merged.update(raw.get("args") or {})

    gpu_filename = merged.pop("gpu_mapping", "gpu_mapping_1.json")
    gpu_path = os.path.join(main_root, "args", gpu_filename)
    try:
        with open(gpu_path) as f:
            merged.update(json.load(f))
    except FileNotFoundError:
        logger.warning("⚠️  args mapping file not found: %s — GPU/batch fields "
                       "will be missing (set \"gpu_mapping\" inside this "
                       "config's \"args\" section to an existing "
                       "main/args/gpu_mapping_<N>.json)", gpu_path)

    merged.setdefault("supervised_config_dir", "main/configs/")
    merged["_default_config_variants"] = raw.get("configs") or {}
    merged["_gpu_mapping_file"] = gpu_filename
    return merged


def load_global_config(index=1):
    global CONFIG
    filename = _CONFIG_MAP.get(index, "global_mapping_1.json")
    config_path = os.path.join(main_dir, "main", filename)
    try:
        with open(config_path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        logger.error("❌ Error: Could not find config at %s", config_path)
        sys.exit(1)

    # Pointer format (global_mapping_5.json+) has a "global" dict pointing at
    # main/global/*.json; legacy global_mapping_1..4.json carry every field
    # inline at the top level and have no such key.
    if isinstance(raw, dict) and isinstance(raw.get("global"), dict):
        CONFIG = _expand_pointer_config(raw, config_path)
        logger.info("📄 Using config: %s (pointer format → main/global/*.json + "
                    "main/args/%s)", filename, CONFIG.get("_gpu_mapping_file"))
    else:
        CONFIG = raw
        logger.info("📄 Using config: %s", filename)
def _resolve_k(entry, k):
    """If entry is a keyed dict, resolve by k then 'default'; otherwise return as-is.

    Supports both numeric k (with floor fallback to largest stored key <= k —
    used by the few-shot k-shot resolver) and non-numeric/string k (exact-match
    only, then "default" — used by e.g. gpu_partition keyed by gpu_type like
    "a40"/"h100").
    """
    if not isinstance(entry, dict):
        return entry
    if k is not None:
        # exact match first
        if str(k) in entry:
            return entry[str(k)]
        # numeric floor: largest stored key <= k_int. Only runs when k itself
        # is numeric; falls through silently for string keys.
        try:
            k_int = int(k)
        except (TypeError, ValueError):
            k_int = None
        if k_int is not None:
            numeric_keys = sorted(
                (int(kk) for kk in entry if kk != "default" and str(kk).isdigit()),
                reverse=True,
            )
            for kk in numeric_keys:
                if kk <= k_int:
                    return entry[str(kk)]
    if "default" in entry:
        return entry["default"]
    # Value-dict short-circuit: when caller didn't request a k-shot lookup
    # AND the dict isn't k-keyed (no numeric keys), treat the dict as the
    # leaf value rather than a lookup. Used by e.g. few_shots' {sizes, min}
    # payload — the caller wants the structured value back, not None.
    if k is None and not any(str(kk).isdigit() for kk in entry):
        return entry
    return None  # no match — caller falls through to next lookup level

def get_global_config(field, model=None, task=None, k=None):
    """Resolve a config field with lookup order: task+model+k → task+model → task → model+k → model → default.

    Each level's value may be a plain scalar/list, a k-keyed dict
    {"default":…, "40":…}, or a value dict (e.g. few_shots' {sizes, min}).
    When a k-keyed dict has no matching k and no "default", the lookup falls
    through to the next level. Raises KeyError if the field has no "default"
    key and no specific match is found.
    """
    if field == "gpu_mappings" and GPU_OVERRIDE is not None:
        return list(GPU_OVERRIDE)
    if field == "ac_similarity_threshold" and SIM_THRESHOLD_OVERRIDE is not None:
        return SIM_THRESHOLD_OVERRIDE
    # Supervised-config override: when the (model, task) supervised model.json
    # defines `field`, that value wins over the global mapping. Lets per-model
    # configs (e.g. TagPrime-C's AC_model_c1.json setting eval_batch_size=4)
    # override the per-task default without editing the global mapping. The
    # loader is lazy + cached, so non-supervised models pay one miss only.
    if model and task:
        sup_cfg = _load_supervised_config(model, task)
        if field in sup_cfg:
            return sup_cfg[field]
    cfg = CONFIG.get(field, {})
    if not isinstance(cfg, dict):
        return cfg
    task_subtree_is_model_keyed = False
    if task and model:
        task_map = cfg.get(task, {})
        if isinstance(task_map, dict):
            # If the caller supplied a model and the task subtree is a dict
            # that didn't match the model or a "default" key, treat the subtree
            # as model-keyed and skip the task-only fall-through below — otherwise
            # `_resolve_k` would return the whole model-keyed dict as a "leaf".
            task_subtree_is_model_keyed = True
        if isinstance(task_map, dict) and model in task_map:
            val = _resolve_k(task_map[model], k)
            if val is not None:
                return val
        if isinstance(task_map, dict) and "default" in task_map:
            val = _resolve_k(task_map["default"], k)
            if val is not None:
                return val
    # Task-only lookup: top-level task key, used by e.g. few_shots where the
    # structure is `{"default": {...}, "AC": {...}, "AC1": {...}}` without any
    # model-keyed nesting. Skipped when the (task, model) branch already
    # probed this subtree — it's a model lookup table, not a leaf value.
    if task and task in cfg and not task_subtree_is_model_keyed:
        val = _resolve_k(cfg[task], k)
        if val is not None:
            return val
    if model and model in cfg:
        val = _resolve_k(cfg[model], k)
        if val is not None:
            return val
    if "default" in cfg:
        val = _resolve_k(cfg["default"], k)
        if val is not None:
            return val
    raise KeyError(f"No value resolved for field='{field}', task={task!r}, model={model!r}, k={k!r} — add a 'default' key to the mapping")

# Target scripts
TARGET_SCRIPT            = os.path.join(main_dir, "TextEE", "evaluate_llm.py")
TARGET_SCRIPT_SUPERVISED = os.path.join(main_dir, "TextEE", "evaluate_supervised.py")

# SLURM script rendering/writing lives in main.job_manager.create_sbatch_script;
# subprocess execution lives in main.job_runner.run_command. This module only
# resolves the arguments (paths, gpu/batch grouping, durations, cmd lists)
# that those two hand off to.

# ----------------------------------------------- #


_CLUSTER_NAME = None


def get_gpu_partition(gpu_type):
    """Resolve SLURM `--partition` for a given gpu_type.

    Cluster name is read once from `/etc/slurm{,/slurm}.conf` and cached —
    no subprocess, no `$SLURM_CLUSTER_NAME` (env var only exists inside a
    job allocation; generation runs on the login node). Defaults to
    "default" when no slurm.conf is present — that key is also the natural
    fallback in `gpu_partition` lookups.

    Lookup reuses `get_global_config`'s standard fallback over
    CONFIG["gpu_partition"]: cluster name in the `model` slot, `gpu_type` in
    the `k` slot. Example structure:

        "gpu_partition": {
            "default": "gpu",
            "grace": {"a40": "gpu-a40", "default": "gpu"}
        }
    """
    global _CLUSTER_NAME
    if _CLUSTER_NAME is None:
        _CLUSTER_NAME = "default"
        for path in ("/etc/slurm/slurm.conf", "/etc/slurm.conf"):
            try:
                with open(path) as f:
                    for line in f:
                        if line.strip().startswith("ClusterName="):
                            _CLUSTER_NAME = line.split("=", 1)[1].strip().lower()
                            break
            except (FileNotFoundError, OSError):
                continue
            if _CLUSTER_NAME != "default":
                break
    return get_global_config("gpu_partition", model=_CLUSTER_NAME, k=gpu_type)


def get_shot_resources(model_key, task, shots_list):
    """Return [(k, gpu_list, batch_size, gpu_type), ...] for every shot in shots_list.

    Done shots are NOT filtered — evaluate_llm.py handles per-shot resume
    internally, and we want re-submission to recompute scores from the stored
    predictions (cheap; no inference re-run unless --repredict).
    """
    return [
        (
            s,
            get_global_config("gpu_mappings",     model=model_key, task=task, k=s),
            get_global_config("eval_batch_size",  model=model_key, task=task, k=s),
            get_global_config("gpu_requirements", model=model_key, task=task, k=s),
        )
        for s in shots_list
    ]

def _variant_suffix(model_key=None, task=None):
    """Variant tag (e.g. 'c1', 'c2') that routes alt configs and isolates artifacts.

    Set via `-C/--config`. Empty string (no flag) means legacy behavior: load
    <task>_model.json or model.json and write to the un-suffixed folder.
    """
    return CONFIG_VARIANT or ""


def _default_variant_for_model(model_key):
    """Per-model default -C variant when the user didn't pass -C at all.

    Sourced from the pointer-format mapping's "configs" table (e.g.
    global_mapping_5.json's {"default": "c1", "TagPrime-C": "c4"}), set
    aside by `_expand_pointer_config` under "_default_config_variants".
    Legacy global_mapping_1..4.json carry no such table, so every model
    falls through to "c1" — identical to the previous hardcoded
    `default=["1"]` CLI behavior.
    """
    table = CONFIG.get("_default_config_variants") or {}
    raw = table.get(model_key, table.get("default", "1"))
    s = str(raw).strip()
    return s if s.lower().startswith("c") else f"c{s}"


_SUPERVISED_CONFIG_CACHE = {}


def _load_supervised_config(model_key, task):
    """Lazy-load and cache the supervised model.json for (model_key, task, variant).

    Resolution mirrors `_resolve_supervised_config`'s variant fallback:
    `<task>_model_<variant>.json` → `<task>_model.json` → `model.json`.
    Cache key includes `_variant_suffix()` so multi-variant runs don't reuse
    the first variant's file for every subsequent pass.

    Returns {} when the model isn't supervised or no config file exists,
    so callers (e.g. `get_global_config`) can do a single `in`-check without
    a separate "is this supervised" guard.
    """
    variant = _variant_suffix(model_key=model_key, task=task)
    key = (model_key, task, variant)
    if key in _SUPERVISED_CONFIG_CACHE:
        return _SUPERVISED_CONFIG_CACHE[key]
    cfg = {}
    if model_key:
        alt_name = CONFIG.get("model_alts", {}).get(model_key, model_key)
        sup_dir  = smart_path(CONFIG.get("supervised_config_dir", "main/configs/"))
        path = _resolve_supervised_config(sup_dir, alt_name, task, model_key=model_key)
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        cfg = loaded
            except Exception:
                cfg = {}
    _SUPERVISED_CONFIG_CACHE[key] = cfg
    return cfg


def _resolve_supervised_config(sup_dir, alt_name, task, model_key=None):
    """Pick the config file for (alt_name, task), honoring the variant suffix.

    Fallback order:
      1. <task>_model_<variant>.json      (per-task + per-variant)
      2. <task>_model.json                (per-task, variant-agnostic)
      3. model.json                       (the original single-config layout)
    """
    variant = _variant_suffix(model_key=model_key, task=task)
    candidates = []
    if task and variant:
        candidates.append(f"{task}_model_{variant}.json")
    if task:
        candidates.append(f"{task}_model.json")
    candidates.append("model.json")
    for name in candidates:
        path = os.path.join(sup_dir, alt_name, name)
        if os.path.exists(path):
            return path
    return os.path.join(sup_dir, alt_name, candidates[-1])


def is_supervised_model(model_key, task=None):
    """True if model_key is registered under CONFIG['model_type']['supervised'].

    Falls back to a config-file existence check for global mappings that haven't
    declared model_type yet (legacy behavior).
    """
    mt = CONFIG.get("model_type", {})
    if isinstance(mt, dict):
        if model_key in mt.get("supervised", []):
            return True
        if model_key in mt.get("LLM", []):
            return False
    alt_name = CONFIG.get("model_alts", {}).get(model_key, model_key)
    sup_dir  = smart_path(CONFIG.get("supervised_config_dir", "main/configs/"))
    return os.path.exists(_resolve_supervised_config(sup_dir, alt_name, task, model_key=model_key))

def smart_path(path):
    if not path:
        return ""
    if SCRATCH_PATH and path.startswith("/scratch_path"):
        return SCRATCH_PATH.rstrip("/") + path[len("/scratch_path"):]
    if path.startswith("/"):
        return path
    return os.path.join(main_dir, path)

def get_supervised_config_details(model_key, dataset_key, task):
    """Paths and GPU settings for a supervised model job."""
    alt_name    = CONFIG.get("model_alts", {}).get(model_key, model_key)
    sup_dir     = smart_path(CONFIG.get("supervised_config_dir", "main/configs/"))
    variant     = _variant_suffix(model_key=model_key, task=task)
    config_path = _resolve_supervised_config(sup_dir, alt_name, task, model_key=model_key)

    gpu_list      = get_global_config("gpu_mappings", model=model_key, task=task)
    num_gpus      = len(gpu_list)
    gpu_list_str  = [str(i) for i in gpu_list]
    gpu_comma_str = ",".join(gpu_list_str)

    base_data_dir        = smart_path(CONFIG.get("input_dir", ""))
    dataset_internal_name = CONFIG.get("dataset_alts", {}).get(dataset_key, dataset_key)
    dataset_path         = os.path.join(base_data_dir, dataset_key)

    # When a variant is active, suffix the output model folder so each variant's
    # pred/score/progress artifacts live in their own directory.
    model_out_key   = f"{model_key}_{variant}" if variant else model_key
    base_output_dir = smart_path(CONFIG.get("output_dir", "results/"))
    output_path     = os.path.join(base_output_dir, task, dataset_key, model_out_key)

    keymap_path = smart_path(CONFIG.get("key_mapping", ""))
    eval_bs     = get_global_config("eval_batch_size", model=model_key, task=task)

    # Resolve the backbone path from the chosen config's pretrained_model_name.
    backbone_path = ""
    model_dir = smart_path(CONFIG.get("model_dir", ""))
    if model_dir and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                _mcfg = json.load(f)
            hf_name = _mcfg.get("pretrained_model_name", "")
            alt = CONFIG.get("model_alts", {}).get(hf_name, hf_name.split("/")[-1])
            candidate = os.path.join(model_dir, alt)
            if os.path.isdir(candidate):
                backbone_path = candidate
        except Exception:
            pass

    # evaluate_supervised builds checkpoint paths as
    # <trained_model_dir>/<task>/<dataset>/<model>/<split>. Prepend the variant
    # so each variant gets its own checkpoint tree without colliding.
    trained_model_dir = smart_path(CONFIG.get("trained_model_dir", ""))
    if trained_model_dir and variant:
        trained_model_dir = os.path.join(trained_model_dir, variant)

    return (config_path, gpu_list_str, gpu_comma_str, num_gpus,
            dataset_path, dataset_internal_name, output_path, eval_bs, keymap_path,
            backbone_path, trained_model_dir)

def get_config_details(model_key, dataset_key, task):
    """Retrieves paths and gpu settings."""
    
    # 1. Model Details
    base_model_dir = smart_path(CONFIG.get("model_dir", ""))
    model_subpath = CONFIG.get("model_alts", {}).get(model_key, model_key)
    model_path = os.path.join(base_model_dir, model_subpath)
    
    gpu_list      = get_global_config("gpu_mappings", model=model_key, task=task)
    num_gpus      = len(gpu_list)
    gpu_list_str  = [str(i) for i in gpu_list]
    gpu_comma_str = ",".join(gpu_list_str)

    # 2. Dataset Details
    base_data_dir = smart_path(CONFIG.get("input_dir", ""))
    dataset_internal_name = CONFIG.get("dataset_alts", {}).get(dataset_key, dataset_key)
    dataset_path = os.path.join(base_data_dir, dataset_key)

    # 3. Output Path
    base_output_dir = smart_path(CONFIG.get("output_dir", "results/"))
    output_path = os.path.join(base_output_dir, task, dataset_key, model_key)    

    # 4. JSON configs for evaluate_llm
    keymap_path = smart_path(CONFIG.get("key_mapping", ""))
    col_def_path = smart_path(CONFIG.get("column_definition", ""))

    return (model_path, gpu_list_str, gpu_comma_str, num_gpus, dataset_path, dataset_internal_name,
            output_path, get_global_config("eval_batch_size", model=model_key, task=task),
            CONFIG.get("max_output_length", 256), 
            CONFIG.get("model_max_length", 4096), 
            keymap_path, col_def_path)

def get_duration(task, model_key):
    """Return SLURM time string (HH:MM:00). Values are fractional hours (8.5 → 08:30:00)."""
    hours_f = float(get_global_config("duration", model=model_key, task=task))
    h = int(hours_f)
    m = round((hours_f - h) * 60)
    return f"{h:02d}:{m:02d}:00"


def collect_skip_unions(combinations, user_few_shot):
    """Pre-pass for `-a error`: for each (task, dataset, shot) that any LLM
    in the run targets, RECOMPUTE the skip vector fresh against each split's
    training data using the task's own algorithm (LLMACTrainer._skip_algorithm
    or LLMAC1Trainer._skip_algorithm), then aggregate per-split decisions by
    the configured `skip_threshold`. Returns dict[(task, dataset, shot)] -> list[col].

    Recomputing (vs unioning stored prediction_metadata) guarantees the skip
    vector reflects the current trainer logic. Supervised models contribute
    nothing — their cells never get NaN'd by skip columns.
    """
    # Load the factor structure once.
    keymap_path = smart_path(CONFIG.get("key_mapping", ""))
    structure = {}
    if keymap_path and os.path.exists(keymap_path):
        try:
            with open(keymap_path) as f:
                structure = (json.load(f) or {}).get("structure", {}) or {}
        except Exception:
            structure = {}
    input_dir = smart_path(CONFIG.get("input_dir", "data/processed_data/"))

    # Cache per-split training data per dataset (avoid rereading).
    # The pool is train.json + val.json per split — matches the few-shot
    # pool that evaluate_llm.py's process_split uses for prediction, so
    # skip-vector recompute here sees the same example set the model saw.
    train_cache = {}
    def _load_train(dataset_key):
        if dataset_key in train_cache:
            return train_cache[dataset_key]
        d_path = os.path.join(input_dir, dataset_key)
        splits = []  # list of (split_name, examples)
        for split_dir in sorted(_glob.glob(os.path.join(d_path, "split*"))):
            examples = []
            for fname in ("train.json", "val.json"):
                fpath = os.path.join(split_dir, fname)
                if not os.path.exists(fpath):
                    continue
                try:
                    with open(fpath) as f:
                        examples.extend(json.loads(line)
                                        for line in f if line.strip())
                except Exception:
                    continue
            if examples:
                splits.append((os.path.basename(split_dir), examples))
        train_cache[dataset_key] = splits
        return splits

    unions = {}
    # Lazy-imported once and reused across combos — TextEE.utils pulls in
    # torch/stanza/_jsonnet (several seconds on Windows).
    convert_predictions = None
    for task, dataset_key, model_key in combinations:
        if not validate_compatibility(task, dataset_key, model_key):
            continue
        if is_supervised_model(model_key, task):
            continue  # skip vectors only meaningful for LLM few-shot runs
        try:
            if task == "IHE":
                from TextEE.models.LLM.AC1Trainer import LLMAC1Trainer
                algo = LLMAC1Trainer._skip_algorithm
            else:
                from TextEE.models.LLM.ACTrainer import LLMACTrainer
                algo = LLMACTrainer._skip_algorithm
        except ImportError:
            algo = None
        if algo is None:
            continue
        fs = get_global_config("few_shots", task=task)
        shots = user_few_shot or [str(x) for x in fs["sizes"]]
        threshold = float(get_global_config("skip_threshold", model=model_key, task=task))
        if threshold <= 0:
            continue
        splits_data = _load_train(dataset_key)
        if not splits_data:
            continue
        # The skip algorithm reads flat factor keys (e.g. ex.get("working_circumstances")).
        # JHE/IHE train.json is the nested constee shape — every factor returns
        # None and the algorithm marks everything as skip. Flatten via the same
        # converter the runtime uses at predict time so the algorithm sees the
        # same factor names it would in practice.
        src_fmt = "ACH" if task in ("JHE", "IHE") else "AC"
        if src_fmt != "AC":
            if convert_predictions is None:
                from TextEE.utils import convert_predictions
            splits_data = [(name, convert_predictions(exs, src_fmt, "AC", config=None))
                           for name, exs in splits_data]
        # min[i] pairs with sizes[i]: factors with fewer than min[i] populated
        # examples get added to the skip vector by the trainer's algorithm.
        size_to_min = dict(zip(fs["sizes"], fs["min"]))
        for shot in shots:
            key = (task, dataset_key, str(shot))
            if key in unions:
                continue  # same (task, dataset, shot) — algorithm is deterministic
            min_count = int(size_to_min.get(int(shot), int(shot)))
            per_split = {}
            counts = {}
            for split_name, examples in splits_data:
                cols = sorted(set(algo(examples, min_count, structure)))
                per_split[split_name] = cols
                for c in cols:
                    counts[c] = counts.get(c, 0) + 1
            n = len(splits_data)
            agg = sorted(c for c, hits in counts.items() if hits / n >= threshold)
            unions[key] = {"aggregate": agg, "per_split": per_split}
    return unions


def get_combo_paths(model_key, dataset_key, task):
    """Resolve (d_path, d_name, out_path) for a (task, dataset, model) combination,
    routing supervised vs LLM config getters. Centralizes the unpacking so every
    action handler accesses these the same way."""
    if is_supervised_model(model_key, task):
        _, _, _, _, d_path, d_name, out_path, _, _, _, _ = get_supervised_config_details(model_key, dataset_key, task)
    else:
        _, _, _, _, d_path, d_name, out_path, _, _, _, _, _ = get_config_details(model_key, dataset_key, task)
    return d_path, d_name, out_path


def get_supervised_train_dir(model_key, dataset_key, task):
    """Resolve the directory where `training_progress.json` lives for a
    supervised combo — matches the layout used by evaluate_supervised:
        <trained_model_dir>/<task>/<dataset>/<model_key>/
    Returns None when `trained_model_dir` isn't configured (in which case
    progress falls back to out_path)."""
    _, _, _, _, _, _, _, _, _, _, trained_model_dir = \
        get_supervised_config_details(model_key, dataset_key, task)
    if not trained_model_dir:
        return None
    return os.path.join(trained_model_dir, task, dataset_key, model_key)


def validate_compatibility(task, dataset_key, model_key):
    """Validates if the dataset and model are approved for the task."""
    # Check dataset
    allowed_datasets = CONFIG.get("task_mapping", {}).get("dataset", {}).get(task)
    if allowed_datasets is not None and dataset_key not in allowed_datasets:
        return False
        
    # Check model
    allowed_models = CONFIG.get("task_mapping", {}).get("model", {}).get(task)
    if allowed_models is not None and model_key not in allowed_models:
        return False
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Manage TextEE experiments: Generate SLURM scripts or Run directly.")
    
    parser.add_argument("-M", "--map", dest="map", type=int, choices=[0, 1, 2, 3], default=0,
                        help="Global mapping config index: 0=global_mapping.json, "
                             "1=global_mapping_1.json, 2=global_mapping_2.json, "
                             "3=global_mapping_3.json "
                             "-> main/global/*.json + main/args/<gpu_mapping>.json")
    parser.add_argument("-u", "--user", default=os.environ.get("USER", ""),
                        help="Username (or numeric index into config 'userid') for scratch path "
                             "resolution. Defaults to $USER from the environment.")
    parser.add_argument("-a", "--action", choices=["generate", "run", "check", "error", "analysis"], default="generate",
                        help="generate / run / check / error: see README. "
                             "analysis: pool preds+golds (like `check --aggregate`) then emit "
                             "task-level, few-shot scaling, and field-level heatmap tables "
                             "(text .log + LaTeX .tex) via TextEE.data_analyzer.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-t", "--task", nargs='+')
    parser.add_argument("-d", "--dataset", nargs='+')
    parser.add_argument("-m", "--model", nargs='+')
    parser.add_argument("-F", "--few-shot", nargs='+')
    parser.add_argument("-g", "--gpus", nargs='+', type=int, default=None,
                        help="Override gpu_mappings for every combo and shot (e.g. -g 0 1). "
                             "Bypasses the per-(task, model, k) lookup in the global config.")
    parser.add_argument("-C", "--config", dest="config", nargs="+", default=None,
                        help="One or more config variants to process (e.g. -C 1 2 c3). "
                             "Each value becomes a variant tag with auto-prepended 'c' "
                             "(so `-C 2` → c2). Suffixes the model output folder and "
                             "routes <task>_model_<variant>.json. With multiple values, "
                             "the combo loop runs once per variant — generate emits a "
                             "separate set of sbatches per variant, check/error report "
                             "each independently. Omit to use each model's default "
                             "variant from the active global mapping's \"configs\" table "
                             "(falls back to c1 for models with no entry).")
    parser.add_argument("--retrain",   action="store_true",
                        help="Force retrain even if a matching checkpoint exists (also triggers re-predict)")
    parser.add_argument("--repredict", action="store_true",
                        help="Force re-prediction even if pred file already exists")
    parser.add_argument("--clear_progress", action="store_true",
                        help="Clear progress files: combine with --repredict to clear prediction_progress.json, "
                             "or --retrain to also clear training_progress.json (supervised only)")
    parser.add_argument("--debug", action="store_true", help="Set CUDA_LAUNCH_BLOCKING=1 for synchronous CUDA error reporting")
    parser.add_argument("--ac_sim_threshold", type=float, default=None,
                        help="Override ac_similarity_threshold from the global "
                             "mapping for every (model, task) lookup. Propagates "
                             "to evaluate_llm/evaluate_supervised via --ac_sim_threshold "
                             "and to -a check/-a error in-process. Bypasses the "
                             "per-(model, task) entries in the global config.")
    parser.add_argument("--aggregate", action="store_true",
                        help="With -a check: print a cross-model F1 comparison table "
                             "(rows=models, cols=kshots) by pooling preds+golds across datasets. "
                             "Without this flag, -a check just prints per-dataset stored scores. "
                             "With -a error: also emit cross-run comparison logs "
                             "(per-combo compare_<model>_<dataset>_<task>.log and a global "
                             "compare_all.log under the same <log_dir>/<run_id>/ folder). "
                             "Without it, -a error only writes per-k errors_*.json/log.")
    parser.add_argument("--log_path", default=None,
                        help="With -a check --aggregate: directory (or full file path) to "
                             "write the comparison table to. With -a error: directory to "
                             "write errors_*.json into (defaults to each model's output dir). "
                             "Default: not logged.")
    parser.add_argument("--slurm-dir", default=os.path.join(main_dir, "jobs"))
    parser.add_argument("--log_level", default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO")
    args = parser.parse_args()

    # Reconfigure stdout/stderr to UTF-8 so emoji-bearing log messages don't
    # crash on Windows consoles that default to cp1252.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(message)s",
    )

    load_global_config(args.map)

    global SCRATCH_PATH
    userid_map = CONFIG.get("userid", {})
    if str(args.user).isdigit():
        username = userid_map.get(str(args.user), "")
    else:
        username = args.user or ""
    if not username:
        # Fallback: first entry in the userid map (preserves prior default behavior)
        username = next(iter(userid_map.values()), "")
    scratch_template = CONFIG.get("scratch_path", "/scratch/user/userid")
    SCRATCH_PATH = scratch_template.replace("userid", username)
    logger.info("👤 User: %s → scratch: %s", username, SCRATCH_PATH)

    global GPU_OVERRIDE
    if args.gpus is not None:
        GPU_OVERRIDE = list(args.gpus)
        logger.info("🎮 GPU override: %s (applies to every combo and shot)", GPU_OVERRIDE)

    global SIM_THRESHOLD_OVERRIDE
    if args.ac_sim_threshold is not None:
        SIM_THRESHOLD_OVERRIDE = float(args.ac_sim_threshold)
        logger.info("📏 ac_similarity_threshold override: %s (applies to every combo)",
                    SIM_THRESHOLD_OVERRIDE)

    global CONFIG_VARIANT
    # `-C` carries one or more config numbers; each becomes a variant tag of
    # form `c<value>` (so -C 2 → c2). The "c" prefix stands for "config" and
    # matches the AC_model_c<N>.json file-name convention. A `None` entry
    # represents the "no-variant" default (un-suffixed paths). CONFIG_VARIANT
    # is set inside the combo loop, once per variant pass.
    if args.config is not None:
        config_variants = []
        for raw in args.config:
            s = str(raw).strip()
            config_variants.append(s if s.lower().startswith("c") else f"c{s}")
        logger.info("🏷️  Config variants: %s (combo loop will iterate over each)",
                    config_variants)
    else:
        config_variants = [None]

    if args.dry_run:
        logger.info("\n🚧 --- DRY RUN MODE ENABLED --- 🚧\n")

    logger.info("🤖 Loading configuration...")

    all_tasks = list(CONFIG.get("task_mapping", {}).get("dataset", {}).keys())
    all_datasets = list(CONFIG.get("dataset_alts", {}).keys())
    all_models = list(CONFIG.get("model_alts", {}).keys())

    tasks = args.task if args.task else all_tasks
    datasets = args.dataset if args.dataset else all_datasets
    models = args.model if args.model else all_models

    user_few_shot = [str(x) for x in args.few_shot] if args.few_shot else None
    if user_few_shot:
        logger.info("🎯 Few-shot sizes: %s (User Override)", user_few_shot)

    if not tasks or not datasets or not models:
        logger.error("❌ Error: Valid targets not found. Check your filters or config.")
        sys.exit(1)

    slurm_dir = args.slurm_dir
    log_dir = smart_path(CONFIG.get("log_dir", "logs/"))
    # Shared across every job/group in this invocation of `-a run` (unlike
    # `generate`, which gets uniqueness for free from SLURM's %j job id).
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.action == "generate" and not args.dry_run:
        os.makedirs(slurm_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        logger.info("📂 Scripts will be saved to: %s", slurm_dir)

    combinations = list(itertools.product(tasks, datasets, models))
    logger.info("📋 Processing %d potential combinations in '%s' mode...", len(combinations), args.action)

    # Collects every errors_*.json written during -a error so we can emit a
    # single cross-(model, dataset, task, k) comparison log at the end.
    all_error_jsons = []

    # Cross-(model, k) bundle of preds+golds for AC re-scoring at end of check
    # OR for analysis-mode table building. Allocated for both branches since
    # `analysis` reuses the same data collection path.
    preds_golds = (
        {} if (args.action == "check" and args.aggregate) or args.action == "analysis"
        else None
    )

    # key_mapping.json provides (1) the factor `structure` used to indent
    # sub-factors under their MAIN parent in the cross-model table, and
    # (2) the `classes` map needed to convert supervised trainers' IE-format
    # predictions (entity_mentions / event_mentions) into the flat AC dicts
    # compute_AC_flat_score expects. Hoisted before the combo loop so
    # check_combo can do the IE→AC conversion + the per-shot rescore.
    keymap_structure = None
    keymap_classes_map = None
    check_sim_threshold = None
    if args.action in ("check", "analysis"):
        keymap_path = smart_path(CONFIG.get("key_mapping", ""))
        if keymap_path and os.path.exists(keymap_path):
            try:
                with open(keymap_path) as f:
                    _keymap = json.load(f) or {}
                keymap_structure = _keymap.get("structure")
                keymap_classes_map = (
                    (_keymap.get("task", {}) or {})
                    .get("classification", {})
                    .get("classes", {})
                ) or None
            except Exception as exc:
                logger.warning("⚠️  failed to load key_mapping: %s", exc)
        # Read sim_threshold straight from the active global mapping — edit
        # the JSON to change. Symmetric with -a error.
        check_sim_threshold = float(get_global_config("ac_similarity_threshold"))
        # `eval_norm` is the same global flag that gets passed through to
        # evaluate_supervised/evaluate_llm as `--norm`. Apply it here too so
        # `-a check` re-scoring uses the same canonical form.
        try:
            check_norm = bool(get_global_config("eval_norm"))
        except KeyError:
            check_norm = False

    # Pre-pass for `-a error` and `-a check`: recompute the union of skip
    # vectors across all models per (task, dataset, shot) from the live
    # train+val data, so both actions exclude the same underfilled factors
    # (and stay in sync with whatever skip_threshold the global mapping
    # currently sets). collect_skip_unions reads via _load_train which
    # already pulls train.json + val.json — matching the few-shot pool
    # process_split uses at prediction time.
    skip_unions = None
    if args.action in ("error", "check", "analysis"):
        skip_unions = collect_skip_unions(combinations, user_few_shot)

    count = 0
    # Iterate the Cartesian product of (variant × combo). When the user passes
    # multiple -C values, every (task, dataset, model) combo runs once per
    # variant, each producing its own per-variant filenames / output dirs.
    # config_variants == [None] when no -C was given → exactly one pass with
    # the "no-variant" un-suffixed paths.
    #
    # Variants only meaningfully differ for SUPERVISED models (they read a
    # variant-specific AC_model_c<N>.json with overrides). LLM models ignore
    # the variant — running them N times under different `-C` values would
    # produce N identical sbatches / N identical check rows. Track LLM combos
    # we've already processed and skip the duplicates.
    _llm_done = set()
    for _variant, (task, dataset_key, model_key) in itertools.product(
            config_variants, combinations):
        if not validate_compatibility(task, dataset_key, model_key):
            continue

        if not is_supervised_model(model_key, task):
            sig = (task, dataset_key, model_key)
            if sig in _llm_done:
                continue
            _llm_done.add(sig)
            CONFIG_VARIANT = None  # no variant suffix on LLM paths / pool key
        else:
            # `-C` omitted (_variant is None) → fall back to this model's
            # default variant from the active mapping's "configs" table.
            CONFIG_VARIANT = _variant if _variant is not None else _default_variant_for_model(model_key)

        safe_model = model_key.replace("/", "_").replace(" ", "")
        # Suffix the variant tag (e.g. `c1`, `c2`) into the job name when one
        # is active, so generated .sbatch/.sh filenames and SLURM job names
        # don't collide across runs of the same (task, dataset, model) under
        # different configs. Mirrors the per-variant output folder layout.
        variant_tag = _variant_suffix(model_key=model_key, task=task)
        job_name = (f"{task}_{dataset_key}_{safe_model}_{variant_tag}"
                    if variant_tag else f"{task}_{dataset_key}_{safe_model}")

        if user_few_shot:
            shots_list = user_few_shot
        else:
            shots_list = [str(x) for x in get_global_config("few_shots", task=task)["sizes"]]
        # Always emit shots in ascending size — script generation, sbatch part
        # ordering, and within-part shot order all derive from this list.
        shots_list = sorted(shots_list, key=lambda s: int(s))

        # `--action check` / `--action analysis` skip ALL script-building /
        # cmd-list construction — we only need the progress manager + stored
        # scores. CONFIG-dependent resolution stays here; progress_analyzer.
        # check_combo is pure data.
        if args.action in ("check", "analysis"):
            d_path, d_name, out_path = get_combo_paths(model_key, dataset_key, task)
            is_sup = is_supervised_model(model_key, task)
            skip_thr = (
                0.0 if is_sup
                else float(get_global_config("skip_threshold",
                                             model=model_key, task=task))
            )
            train_dir = (get_supervised_train_dir(model_key, dataset_key, task)
                         if is_sup else None)
            # Base skip vector from the global mapping (e.g. ACH:
            # ["accident_report", "id", "accident_type"]). evaluate_llm.py
            # unions this with the few-shot-derived skip vector at predict
            # time — mirror that union here so check scores match.
            base_skip = list(get_global_config(
                "skip_columns", model=model_key, task=task) or [])
            # Resolve the display label for the cross-model table: prefer
            # the model_alts value (full/HF name) over the short alias.
            model_display = CONFIG.get("model_alts", {}).get(model_key, model_key)
            check_combo(
                job_name, task, dataset_key, model_key, shots_list,
                d_path=d_path, d_name=d_name, out_path=out_path,
                is_supervised=is_sup, skip_threshold=skip_thr,
                sim_threshold=check_sim_threshold,
                preds_golds=preds_golds, skip_unions=skip_unions,
                base_skip=base_skip,
                structure=keymap_structure, classes_map=keymap_classes_map,
                norm=check_norm,
                variant=_variant_suffix(model_key=model_key, task=task),
                train_dir=train_dir,
                model_display=model_display,
            )
            count += 1
            continue

        # `--action error` runs LLM AC error analysis against existing predictions.
        # Writes errors_<model>_<dataset>_<task>_k<K>.json next to the pred files
        # (or under --log_path if provided). No SLURM / cmd construction needed.
        if args.action == "error":
            from main.error_analysis import (
                analyze_combination, print_summary_tables,
                write_comparison_log, make_run_id,
            )
            d_path, d_name, out_path = get_combo_paths(model_key, dataset_key, task)

            if not os.path.isdir(out_path):
                logger.info("  ⬜ [%s] no output dir at %s — nothing to analyze", job_name, out_path)
                count += 1
                continue

            keymap_path = smart_path(CONFIG.get("key_mapping", ""))
            input_dir   = smart_path(CONFIG.get("input_dir", "data/processed_data/"))
            if not os.path.exists(keymap_path):
                logger.error("❌ key_mapping not found at %s", keymap_path)
                sys.exit(1)
            with open(keymap_path) as f:
                keymap = json.load(f)
            structure = (keymap or {}).get("structure", {})
            classes_map = (
                ((keymap or {}).get("task", {}) or {})
                .get("classification", {})
                .get("classes", {})
            ) or None

            # Per-combo lookup: lets a supervised model.json override the
            # global mapping's ac_similarity_threshold (the (model, task)
            # branch in get_global_config). Symmetric with -a check.
            sim = float(get_global_config("ac_similarity_threshold",
                                          model=model_key, task=task))
            # Same global flag passed through to evaluate_supervised/llm as
            # --norm. Apply here so -a error parsing matches the scorer.
            try:
                err_norm = bool(get_global_config("eval_norm",
                                                  model=model_key, task=task))
            except KeyError:
                err_norm = False

            # Pull the per-shot skip vector for this (task, dataset) from the
            # precomputed unions. Supervised models have no entry — skip_map
            # stays empty and analyze_combination acts as before.
            skip_map = {}
            for shot_str in shots_list:
                entry = (skip_unions or {}).get((task, dataset_key, shot_str))
                if entry:
                    skip_map[int(shot_str)] = entry.get("aggregate") or []

            # Base skip vector from the global mapping (same source -a check
            # uses via base_skip): structural columns like accident_report /
            # id that must never be analyzed as factors.
            err_base_skip = list(get_global_config(
                "skip_columns", model=model_key, task=task) or [])

            # error_analysis owns the subfolder naming. We just hand it the
            # bare log_dir (or --log_path override) and let it wrap with
            # <run_id>/<task>/<dataset>/<model>/ when auto_subdir is on.
            logger.info("\n🔍 [error] %s  →  %s  (sim=%s, norm=%s)", job_name,
                        args.log_path or log_dir, sim,
                        "on" if err_norm else "off")
            written = analyze_combination(
                out_path=out_path,
                model=model_key,
                dataset=dataset_key,
                task=task,
                gold_root=input_dir,
                structure=structure,
                sim_threshold=sim,
                shots=[int(s) for s in shots_list],
                log_dir=args.log_path or log_dir,
                classes_map=classes_map,
                auto_subdir=(args.log_path is None),
                write_compare=args.aggregate,
                skip_factors_by_shot=skip_map,
                base_skip=err_base_skip,
                norm=err_norm,
                variant=_variant_suffix(model_key=model_key, task=task),
            )
            for path in written:
                print_summary_tables(path)
            all_error_jsons.extend(written)
            count += 1
            continue

        force_train   = args.retrain
        force_predict = force_train or args.repredict

        if is_supervised_model(model_key, task):
            # ---- supervised model ----
            cfg_path, gpu_list_str, gpu_comma_str, num_gpus, d_path, d_name, out_path, eval_bs, key_map, backbone_path, trained_model_dir = \
                get_supervised_config_details(model_key, dataset_key, task)

            # Skip the NFS-heavy supervised_status read when force_train/force_predict
            # is set — we're processing the job either way, and evaluate_supervised.py
            # has its own per-split resume logic that picks up cached checkpoints.
            if force_train or force_predict:
                status = "train"
            else:
                train_dir = (os.path.join(trained_model_dir, task, dataset_key, model_key)
                             if trained_model_dir else None)
                status = supervised_status(task, dataset_key, model_key, d_path, d_name,
                                           out_path, train_dir=train_dir)
                if status == "done":
                    logger.info("  ⏭️  [SKIP] %s: scores already exist", job_name)
                    continue

            cmd_list = [
                "python", TARGET_SCRIPT_SUPERVISED,
                "-t", task,
                "-d", d_path,
                "-D", d_name,
                "-m", model_key,
                "-o", out_path,
                "-g"
            ] + gpu_list_str + [
                "-b", str(eval_bs),
                "--config", cfg_path,
                "--ac_sim_threshold", str(get_global_config(
                    "ac_similarity_threshold", model=model_key, task=task)),
            ]
            skip_cols = get_global_config("skip_columns", model=model_key, task=task) or []
            if skip_cols:
                cmd_list += ["--skip_columns", *[str(c) for c in skip_cols]]
            try:
                if get_global_config("eval_norm", model=model_key, task=task):
                    cmd_list.append("--norm")
            except KeyError:
                pass  # eval_norm absent in older global maps → no --norm flag
            if backbone_path:
                cmd_list += ["--backbone_path", backbone_path]
            if trained_model_dir:
                cmd_list += ["--trained_model_dir", trained_model_dir]
            stanza_path = smart_path(CONFIG.get("stanza_path", ""))
            if stanza_path:
                cmd_list += ["--stanza_path", stanza_path]
            # Supervised configs no longer carry "log_path"; pass the global
            # mapping's log_dir so evaluate_supervised can populate it on config.
            cmd_list += ["--log_path", log_dir]
            if task in ("JHE", "IHE"):
                cmd_list += ["--key_map", str(key_map)]
            if status == "eval_only":
                cmd_list.append("--eval_only")
                logger.info("  🔁 [EVAL_ONLY] %s: checkpoints found, skipping training", job_name)
            if args.retrain:
                cmd_list.append("--retrain")
            if force_predict and not args.retrain:
                cmd_list.append("--repredict")
            if args.clear_progress:
                cmd_list.append("--clear_progress")

        else:
            # ---- LLM model ----
            model_path, gpu_list_str, gpu_comma_str, num_gpus, d_path, d_name, out_path, eval_bs, max_out, model_max, key_map, col_defs = \
                get_config_details(model_key, dataset_key, task)

            # (check action short-circuits at the top of the combo loop; nothing to do here.)

            # Per-shot GPU/batch resolution. We split into sub-jobs grouped by
            # GPU config so each SLURM job requests exactly the GPUs it needs.
            resources = get_shot_resources(model_key, task, shots_list)
            if not resources:
                # Defensive: shots_list was empty (or get_shot_resources
                # returned nothing). Skip the combo rather than emit an
                # empty SLURM script.
                logger.info("  ⏭️  [SKIP] %s: no remaining shots", job_name)
                continue

            # Per-shot (gpu, batch) grouping is computed per bucket inside
            # the generate branch below; the legacy fallback path further down
            # only needs `resources` directly. Keep base_cmd_tail here because
            # it's the same across every bucket and group.
            base_cmd_tail = []
            stanza_path = smart_path(CONFIG.get("stanza_path", ""))
            if stanza_path:
                base_cmd_tail += ["--stanza_path", stanza_path]
            try:
                if get_global_config("eval_norm", model=model_key, task=task):
                    base_cmd_tail.append("--norm")
            except KeyError:
                pass  # eval_norm absent in older global maps → no --norm flag
            if force_predict:
                base_cmd_tail.append("--repredict")
            if args.clear_progress:
                base_cmd_tail.append("--clear_progress")
            if args.debug:
                # Forward to evaluate_llm.py's own --debug, which dumps the
                # model's per-prompt I/O to terminal + prompt_logs file.
                # Independent of `debug_env=CUDA_LAUNCH_BLOCKING` used in the
                # SLURM script template.
                base_cmd_tail.append("--debug")

        if not is_supervised_model(model_key, task) and args.action == "generate":
            # ---- LLM generate path ----
            # 1. Bucket the shots by weight under the SLURM wall-clock budget
            #    (see `few_shot_buckets` / `few_shot_weights` in global config).
            #    A single bucket means bucketing is off — filenames stay
            #    identical to the legacy single-wrapper layout.
            # 2. For each bucket, fall back to the existing (gpu, batch)
            #    grouping → emit one .sbatch per group + a chained .sh wrapper.
            # 3. When >1 bucket, emit a master submit.sh that submits every
            #    bucket wrapper as an INDEPENDENT job chain (no afterok between
            #    buckets — each bucket is its own SLURM submission).
            log_filename = os.path.join(log_dir, make_log_filename(job_name, "%j"))
            duration = get_duration(task, model_key)
            capacity_hours = parse_slurm_time_to_hours(duration)

            shot_order = [r[0] for r in resources]
            shot_buckets = bucket_shots_balanced(
                shot_order, task, model_key, get_global_config,
                capacity_hours, logger=logger,
            )
            multi_bucket = len(shot_buckets) > 1
            res_lookup = {s: (gl, bs, gt) for s, gl, bs, gt in resources}

            if multi_bucket:
                logger.info("  📦 [%s] %d bucket(s) at capacity %.1fh",
                            job_name, len(shot_buckets), capacity_hours)
            bucket_wrappers = []

            for b_idx, bucket_shot_list in enumerate(shot_buckets, 1):
                # Re-run the (gpu, batch, gpu_type) grouping on this bucket's
                # shots. gpu_type is part of the key so shots whose per-k
                # `gpu_requirements` differ (e.g. a100 for k=20, h100 for k=40)
                # split into separate sbatch parts.
                b_resources = [(s, *res_lookup[s]) for s in bucket_shot_list]
                shot_groups = {}
                for s, gl, bs, gt in b_resources:
                    shot_groups.setdefault((tuple(gl), bs, gt), []).append(
                        (s, list(gl), bs, gt))
                for k in shot_groups:
                    shot_groups[k].sort(key=lambda x: int(x[0]))
                sorted_groups = sorted(
                    shot_groups.items(),
                    key=lambda kv: (min(int(s) for s, _, _, _ in kv[1]),
                                    len(kv[0][0]), kv[0][1]))

                bucket_tag = f"_b{b_idx}" if multi_bucket else ""
                bucket_job_name = f"{job_name}{bucket_tag}"

                # Resolve each (gpu, batch, gpu_type) group into a plain dict
                # of ARGUMENTS — no cmd_list/cmd_str here. Turning resolved
                # args into an actual argv is build_llm_cmd_list's job
                # (main/job_runner.py); deciding how many sbatch files those
                # sub-jobs become is create_grouped_sbatch_parts's job
                # (main/job_manager.py). run.py only resolves values.
                sub_jobs = []
                for (gpu_tuple, eval_bs, gpu_type), group_shots in sorted_groups:
                    shots_str = [str(s) for s, _, _, _ in group_shots]
                    # Parallel min[i] per shot — drives evaluate_llm.py's
                    # per-split skip-vector threshold via --shot_mins.
                    fs_task   = get_global_config("few_shots", task=task)
                    size2min  = dict(zip(fs_task["sizes"], fs_task["min"]))
                    mins_str  = [str(size2min.get(int(s), int(s))) for s in shots_str]

                    sub_jobs.append({
                        "gpu_list": list(gpu_tuple), "gpu_type": gpu_type, "eval_bs": eval_bs,
                        "shots": shots_str, "mins": mins_str,
                        "task": task, "d_path": d_path, "d_name": d_name,
                        "model_path": model_path, "out_path": out_path,
                        "max_out": max_out, "model_max": model_max,
                        "sim_threshold": get_global_config(
                            "ac_similarity_threshold", model=model_key, task=task),
                        "skip_threshold": get_global_config(
                            "skip_threshold", model=model_key, task=task),
                        "key_map": key_map, "col_defs": col_defs,
                        "base_cmd_tail": base_cmd_tail,
                        "skip_cols": get_global_config(
                            "skip_columns", model=model_key, task=task) or [],
                    })

                part_files = create_grouped_sbatch_parts(
                    slurm_dir, bucket_job_name, log_filename, main_dir,
                    TARGET_SCRIPT, sub_jobs, get_gpu_partition, duration,
                    debug_env="export CUDA_LAUNCH_BLOCKING=1" if args.debug else "",
                    dry_run=args.dry_run,
                )

                # Per-job batch/k-shot breakdown is logged by
                # create_grouped_sbatch_parts itself (job_manager.py) as it
                # writes each part file — it's the only place that knows
                # which sub-jobs actually landed in which part.
                if multi_bucket:
                    logger.info("    bucket %d → shots %s, %d job(s)",
                                b_idx, bucket_shot_list, len(part_files))
                else:
                    logger.info("  📊 [%s] %d job(s)", job_name, len(part_files))

                # Per-bucket wrapper: parts inside one bucket still chain via
                # afterok so they share log file and serial GPU usage. Bucket
                # wrappers themselves are independent (the master script below
                # submits each one without dependencies).
                wrapper_path = os.path.join(slurm_dir, f"{bucket_job_name}.sh")
                create_bucket_wrapper(wrapper_path, part_files, log_dir,
                                       bucket_job_name, dry_run=args.dry_run)
                bucket_wrappers.append(os.path.basename(wrapper_path))

            # Master submit.sh: only emitted when more than one bucket exists.
            # It invokes each bucket wrapper sequentially in the same shell
            # (so SLURM job submissions happen back-to-back but the resulting
            # cluster jobs are independent — buckets do NOT chain via afterok).
            if multi_bucket:
                master_path = os.path.join(slurm_dir, f"{job_name}.sh")
                create_master_submit(master_path, bucket_wrappers, dry_run=args.dry_run)

            # Skip the legacy single-script generation block below for LLM jobs.
            count += 1
            continue

        # ---- LLM "run" path ----
        # Mirror the generate split: one subprocess per (gpu_list, batch_size)
        # group, so each invocation gets exactly the GPUs and batch size
        # resolved for its shot subset (instead of collapsing to a single
        # bundle with min-gpu/max-batch, which loses per-shot tuning).
        if not is_supervised_model(model_key, task):
            shot_groups = {}
            # gpu_type is irrelevant on the run path (no SLURM allocation),
            # so collapse over it — only (gpu_list, batch) decides subprocess grouping.
            for s, gl, bs, _ in resources:
                shot_groups.setdefault((tuple(gl), bs), []).append(s)
            sorted_groups = sorted(
                shot_groups.items(),
                key=lambda kv: (min(int(s) for s in kv[1]),
                                len(kv[0][0]), kv[0][1]))

            for i, ((gpu_tuple, batch), group_shots) in enumerate(sorted_groups, 1):
                gl_str    = [str(x) for x in gpu_tuple]
                gl_comma  = ",".join(gl_str)
                shots_str = sorted((str(s) for s in group_shots), key=int)
                # Parallel min[i] per shot — drives evaluate_llm.py's
                # per-split skip-vector threshold via --shot_mins.
                fs_task   = get_global_config("few_shots", task=task)
                size2min  = dict(zip(fs_task["sizes"], fs_task["min"]))
                mins_str  = [str(size2min.get(int(s), int(s))) for s in shots_str]

                sub_cmd = build_llm_cmd_list(
                    TARGET_SCRIPT, task=task, d_path=d_path, d_name=d_name,
                    model_path=model_path, out_path=out_path,
                    gpu_list=gpu_tuple, eval_bs=batch,
                    max_out=max_out, model_max=model_max,
                    sim_threshold=get_global_config(
                        "ac_similarity_threshold", model=model_key, task=task),
                    skip_threshold=get_global_config(
                        "skip_threshold", model=model_key, task=task),
                    key_map=key_map, col_defs=col_defs,
                    shots=shots_str, mins=mins_str,
                    base_cmd_tail=base_cmd_tail,
                    skip_cols=get_global_config(
                        "skip_columns", model=model_key, task=task) or [],
                )
                group_label = (f"{job_name} [group {i}/{len(sorted_groups)} "
                               f"k={shots_str} gpu={gl_comma} batch={batch}]")

                env_overrides = {"CUDA_VISIBLE_DEVICES": gl_comma}
                if args.debug:
                    env_overrides["CUDA_LAUNCH_BLOCKING"] = "1"
                run_log_path = os.path.join(log_dir, make_log_filename(job_name, run_timestamp))
                run_command(sub_cmd, env_overrides=env_overrides,
                            label=group_label, dry_run=args.dry_run,
                            log_path=run_log_path)

            count += 1
            continue

        # ---- Supervised path (generate | run) ----
        # cmd_list was built earlier in the supervised branch above.
        cmd_str = " ".join([str(c) for c in cmd_list])

        if args.action == "generate":
            slurm_filename = os.path.join(slurm_dir, f"{job_name}.sbatch")
            log_filename = os.path.join(log_dir, make_log_filename(job_name, "%j"))
            gpu_type = get_global_config("gpu_requirements", model=model_key, task=task)

            create_sbatch_script({
                "partition": get_gpu_partition(gpu_type),
                "num_gpus": num_gpus,
                "gpu_type": gpu_type,
                "job_name": job_name,
                "log_path": log_filename,
                "work_dir": main_dir,
                "cmd_str": cmd_str,
                "debug_env": "export CUDA_LAUNCH_BLOCKING=1" if args.debug else "",
                "duration": get_duration(task, model_key),
                "path": slurm_filename,
            }, dry_run=args.dry_run)

        elif args.action == "run":
            env_overrides = {"CUDA_VISIBLE_DEVICES": gpu_comma_str}
            if args.debug:
                env_overrides["CUDA_LAUNCH_BLOCKING"] = "1"
            run_log_path = os.path.join(log_dir, make_log_filename(job_name, run_timestamp))
            run_command(cmd_list, env_overrides=env_overrides,
                        label=job_name, dry_run=args.dry_run,
                        log_path=run_log_path)

        count += 1

    if args.action == "check":
        if preds_golds:
            render_cross_model_comparison(
                preds_golds, skip_unions,
                sim_threshold=check_sim_threshold,
                structure=keymap_structure,
                classes_map=keymap_classes_map,
                log_path=args.log_path,
                norm=check_norm,
            )
        logger.info("\n🔎 Progress check complete. %d valid combinations checked.", count)
    elif args.action == "analysis":
        if preds_golds:
            # Build display-name -> alias map so tb1/tb2 use the short
            # alias key (e.g. "Qwen3.5-9B") instead of the full HF name
            # (e.g. "Qwen3.5-9B-instruct" — whatever model_alts stored).
            # check_combo uses the value (full name) as the row key, so
            # we reverse model_alts here.
            model_aliases = {
                full: short
                for short, full in (CONFIG.get("model_alts") or {}).items()
            }
            render_analysis_tables(
                preds_golds, skip_unions,
                structure=keymap_structure,
                classes_map=keymap_classes_map,
                norm=check_norm,
                model_aliases=model_aliases,
                log_path=args.log_path or log_dir,
            )
        else:
            logger.warning("⚠️  No (model, task, k) combinations had preds+golds to analyze.")
        logger.info("\n📊 Analysis complete. %d valid combinations processed.", count)
    elif args.action == "error":
        from main.error_analysis import (
            analyze_combination, print_summary_tables,
            write_aggregate_comparison_log, make_run_id,
        )
        # Cross-(model, dataset, task, k) comparison log — opt-in via
        # --aggregate. Nested one subfolder per dataset under the same
        # per-run folder analyze_combination wrote into (or directly under
        # --log_path when caller pinned a flat dir) — same layout -a
        # analysis uses (progress_analyzer.per_dataset_log_path).
        if args.aggregate and len(all_error_jsons) >= 2:
            global_compare_dir = args.log_path or os.path.join(
                log_dir, make_run_id("error_analysis"))
            os.makedirs(global_compare_dir, exist_ok=True)
            # Same alias map as analysis mode — keeps the LaTeX by-model
            # summary using short names (e.g. "Qwen3.5-9B" rather than the
            # full HF/checkpoint key).
            err_model_aliases = {
                full: short
                for short, full in (CONFIG.get("model_alts") or {}).items()
            }
            # Supervised list from the global mapping — orders by-model table
            # rows as supervised first, then LLMs, ascending name within each.
            err_supervised = set(
                (CONFIG.get("model_type") or {}).get("supervised") or [])
            # Row order for the grouped LaTeX table — model_alts' own key
            # order (main/global/model_mapping.json), so ordering and short
            # names both come from the same config instead of a hardcoded list.
            err_model_order = list((CONFIG.get("model_alts") or {}).keys())
            write_aggregate_comparison_log(
                all_error_jsons, global_compare_dir,
                model_aliases=err_model_aliases,
                supervised_models=err_supervised,
                model_order=err_model_order)
        logger.info("\n🔎 Error analysis complete. %d valid combinations processed.", count)
    elif args.dry_run:
        logger.info("\n🚧 Dry Run Complete. Checked %d valid combinations.", count)
    elif args.action == "generate":
        logger.info("\n🎉 Done! Generated %d job(s).", count)
        logger.info("👉 LLM jobs (*.sh wrappers chain their *.sbatch parts): "
                    "for f in %s/*.sh; do bash \"$f\"; done", slurm_dir)
        logger.info("   (Supervised jobs emit a single *.sbatch — submit with sbatch directly.)")
    else:
        logger.info("\n🎉 Done! Executed %d jobs.", count)

if __name__ == "__main__":
    main()