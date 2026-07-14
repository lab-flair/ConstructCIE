import os, logging, json, glob, gc, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from utils import (load_config, load_data, set_seed, set_gpus, update_namespace,
                   print_vram, get_trainer_class, convert_predictions)
from scorer import (compute_scores, print_scores, strict_keys_from_keymap,
                     exclude_factor_from_overall)

# JHE/IHE predict accident_type; ACH/ACH1 receive it as gold input via
# skip_columns. Strip accident_type out of JHE/IHE scores' OVERALL aggregate
# so the two task families are directly comparable in cross-model tables.
# accident_type stays in per_factor, so its standalone F1 is still available.
E2_TASKS = {"JHE", "IHE"}


def _strip_acc_type_for_e2(scores, task):
    if scores and task in E2_TASKS:
        exclude_factor_from_overall(scores, "accident_type")
    return scores

logger = logging.getLogger(__name__)

AC_FAMILY_TASKS = frozenset({
    "AC", "AC1", "ACH", "ACH1", "JHE", "IHE",
})

# Lazy model-load: set True after `trainer.load_model(...)` runs once.
# Re-score-only invocations (every shot's pred file already on disk, no
# --repredict) never enter the predict branch in process_split, so we skip
# the multi-minute weight load entirely.
_MODEL_LOADED = False


def _ensure_model_loaded(trainer, args):
    """Load the trainer's checkpoint on first call. Idempotent — subsequent
    calls are O(1). Invoked from process_split right before `trainer.predict`."""
    global _MODEL_LOADED
    if _MODEL_LOADED:
        return
    logger.info(f"Loading model {args.model}...")
    trainer.load_model(checkpoint=args.model)
    logger.info("Model loaded successfully.")
    _MODEL_LOADED = True


# Split Processing Logic
def process_split(split_dir, shot_size, trainer, args, config, trainer_class, model_name, progress_mgr):
    split_name = os.path.basename(split_dir)
    test_file = os.path.join(split_dir, "test.json")
    train_file = os.path.join(split_dir, "train.json")
    val_file   = os.path.join(split_dir, "val.json")

    # 1. Base filename format
    pred_filename = args.pred_pattern.format(
        model=model_name, dataset=config.dataset, task=config.task.lower(), split=split_name, shot=shot_size
    )
    prompts_filename = args.prompt_pattern.format(
        model=model_name, dataset=config.dataset, task=config.task.lower(), split=split_name, shot=shot_size
    )

    pred_output_path = os.path.join(args.output_dir, pred_filename)
    prompts_output_path = os.path.join(args.output_dir, prompts_filename)

    print_vram("Before Proccessing")

    if not os.path.exists(test_file):
        logger.warning(f"Test file not found for {split_name}, skipping.")
        return None

    # Few-shot pool: this split's train.json + val.json. Test never enters.
    # Cross-split supplementation is still avoided (it leaks per-split train
    # composition: a factor under-represented in split A's own train could
    # be pushed past the skip threshold by examples that are A's test-set
    # members appearing in other splits' train).
    few_shot_data = None
    if shot_size > 0:
        pool = []
        if os.path.exists(train_file):
            split_train, _ = load_data(config.task, train_file, trainer_class.add_extra_info_fn, config)
            pool.extend(split_train)
        if os.path.exists(val_file):
            split_val, _ = load_data(config.task, val_file, trainer_class.add_extra_info_fn, config)
            pool.extend(split_val)
        few_shot_data = pool or None

    # Recompute the per-split skip vector every run (cheap) and persist — keeps
    # prediction_metadata.json fresh even when predictions are resumed from disk.
    # `min_count` comes from few_shots[task].min via --shot_mins (parallel to -F);
    # falls back to shot_size when --shot_mins wasn't supplied (legacy behavior).
    split_skip = []
    if shot_size > 0 and few_shot_data and hasattr(trainer, "compute_skip_columns"):
        min_count = (args.size_to_min or {}).get(shot_size, shot_size)
        split_skip = trainer.compute_skip_columns(few_shot_data, shot_size,
                                                  min_count=min_count)
        progress_mgr.set_skip_columns(shot_size, split_name, split_skip)

    # Check Progress (We use pred_filename for tracking completion)
    existing_fname, _ = progress_mgr.get_result(shot_size, split_name)

    base_skip = list(args.skip_columns or [])
    if not args.repredict and existing_fname and os.path.exists(os.path.join(args.output_dir, existing_fname)):
        logger.info(f"--> [Resume] {split_name} already completed: {existing_fname}")
        try:
            split_predictions = progress_mgr.load_split(existing_fname)
            # Reloaded preds for ACH-family tasks are nested constee shape
            # (trainer.save_predictions wrote them that way); flatten before
            # scoring so they line up with load_data's always-flat gold.
            _src_fmt = "ACH" if args.task in ("ACH", "ACH1", "JHE", "IHE") else "AC"
            split_predictions = convert_predictions(
                split_predictions, _src_fmt, "AC", config)
            eval_data, _ = load_data(args.task, test_file, trainer_class.add_extra_info_fn, config)
            shot_skip = (sorted(set(base_skip) | set(split_skip))
                         if args.skip_threshold > 0 else base_skip)
            split_scores = compute_scores(split_predictions, eval_data, args.task,
                             sim_threshold=getattr(config, "ac_similarity_threshold", 0.8),
                             skip_columns=shot_skip,
                             norm=args.norm,
                             strict_keys=strict_keys_from_keymap(getattr(config, "key_map", None)))
            _strip_acc_type_for_e2(split_scores, args.task)
            print_scores(split_scores, split=split_name, shot=shot_size)
            return split_scores
        except Exception as e:
            logger.warning(f"Failed to score existing file, re-running inference: {e}")

    # Run Inference. Model weights are lazy-loaded — when every (shot, split)
    # already has predictions on disk and the user just wants to re-score,
    # we never enter this block and skip the multi-minute load entirely.
    _ensure_model_loaded(trainer, args)
    logger.info(f"Processing {split_name} (K={shot_size})...")
    eval_data, _ = load_data(args.task, test_file, trainer_class.add_extra_info_fn, config)

    pred_output = trainer.predict(
        eval_data,
        few_shot_size=shot_size,
        few_shot_data=few_shot_data
    )
    
    # Check if the model returned the dual-list tuple (predictions, prompt_logs)
    prompt_logs = []
    if isinstance(pred_output, tuple) and len(pred_output) == 2:
        split_predictions, prompt_logs = pred_output
    else:
        split_predictions = pred_output
    
    # --- SAVE PREDICTIONS ---
    # Delegate to the trainer's save_predictions hook. The base AC trainer's
    # impl writes flat-AC JSONL; ACH-family tasks (ACH/ACH1/E2ACH/E2ACH1)
    # are detected from `self.config.task` and converted back to nested
    # constee shape before write.
    trainer.save_predictions(split_predictions, pred_output_path)
    logger.info(f"Saved predictions to {pred_output_path}")
    
    # Save Prompts Output
    if prompt_logs:
        with open(prompts_output_path, 'w', encoding='utf-8') as f:
            json.dump(prompt_logs, f, indent=4)
        logger.info(f"Saved prompt logs to {prompts_output_path}")

    # Register the split as complete using the PREDICTIONS filename
    progress_mgr.complete_split(shot_size, split_name, pred_filename)

    # Local Score — always honor --skip_columns; union with the per-split
    # vector only when --skip_threshold > 0 enables few-shot-based skipping.
    shot_skip = (sorted(set(base_skip) | set(split_skip))
                 if args.skip_threshold > 0 else base_skip)
    split_scores = compute_scores(split_predictions, eval_data, args.task,
                             sim_threshold=getattr(config, "ac_similarity_threshold", 0.8),
                             skip_columns=shot_skip,
                             norm=args.norm,
                             strict_keys=strict_keys_from_keymap(getattr(config, "key_map", None)))
    _strip_acc_type_for_e2(split_scores, args.task)
    print_scores(split_scores, split=split_name, shot=shot_size)

    print_vram("After Proccessing")
    del split_predictions, eval_data
    gc.collect()

    return split_scores

def main():
    parser = ArgumentParser()
    parser.add_argument('-t', '--task', required=True,
                        choices=["E2E", "ED", "EAE", "EARL",
                                 "AC", "AC1", "ACH", "ACH1", "JHE", "IHE"])
    parser.add_argument('-d', '--dataset_dir', required=True, help="Base folder containing split subfolders")
    parser.add_argument('-D', '--dataset', type=str, required=True)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-g', '--gpu_device', type=int, nargs='+', default=[0])
    parser.add_argument('-F', '--few_shot_size', type=int, nargs='+', default=[0])
    parser.add_argument('--shot_mins', type=int, nargs='+', default=None,
                        help="Per-shot scorer-skip threshold, parallel to -F. "
                             "Comes from few_shots[task].min in the global mapping. "
                             "When absent, the trainer falls back to using few_shot_size.")
    parser.add_argument('-b', '--eval_batch_size', type=int, default=8)
    parser.add_argument('-M', '--max_output_length', type=int, default=256)
    parser.add_argument('-l', '--model_max_length', type=int, default=4096)
    parser.add_argument('-o', '--output_dir', type=str, default="./results")
    
    parser.add_argument('--repredict', action='store_true', help="Force re-prediction even if pred file already exists")
    parser.add_argument('--clear_progress', action='store_true',
                        help="When combined with --repredict, also clears prediction_progress.json")
    parser.add_argument('--debug', action='store_true', help="If set, output model prompts and answer to terminal")

    parser.add_argument('--split_pattern', type=str, 
                        default="{model}_{dataset}_{task}_{split}_k{shot}.json",
                        help="Pattern for individual split predictions. Must contain {model}, {dataset}, {task}, {split}, {shot}")
    
    parser.add_argument('--shot_pattern', type=str,
                        default="{model}_{dataset}_{task}_k{shot}.json",
                        help="Pattern for final score summary. Must contain {model}, {dataset}, {task}, {shot}")
                        
    # AC Specific Arguments
    parser.add_argument('--key_map', type=str, default=None, help="Path to JSON schema file (Required for AC task)")
    parser.add_argument('--col_defs', type=str, default=None, help="Path to JSON factor descriptions file (Required for AC task)")
    parser.add_argument('--ac_sim_threshold', type=float, default=0.8, help="Similarity threshold for AC scoring")
    parser.add_argument('--norm', action='store_true',
                        help="Enable canonical normalization (lowercase + "
                             "trailing-punct strip + whitespace collapse) "
                             "inside parse_ac_texts when scoring. Off by "
                             "default — scoring is then case- and punct-strict.")
    parser.add_argument('--skip_threshold', type=float, default=0.0,
                        help="At AC aggregate scoring, drop a column if it was under-represented in "
                             "fraction >= skip_threshold of splits. 0 disables.")
    parser.add_argument('--skip_columns', nargs="+", default=[],
                        help="Columns to always exclude from AC scoring "
                             "(unioned with the per-split skip vector when "
                             "skip_threshold>0). Sourced from `skip_columns` "
                             "in the global mapping — typically ids, source "
                             "fields, and event_type fields that aren't "
                             "supposed to be scored as factors.")
    parser.add_argument('--stanza_path', type=str, default=None,
                        help="Path to stanza models directory (passed to stanza.Pipeline(dir=...))")

    args, unknown_args = parser.parse_known_args()

    # Build size→min map from parallel `-F` and `--shot_mins` lists. Used by
    # process_split to pass `min_count` into trainer.compute_skip_columns so
    # the recorded per-split skip vector reflects few_shots[task].min.
    args.size_to_min = None
    if args.shot_mins:
        if len(args.shot_mins) != len(args.few_shot_size):
            logger.warning(
                "--shot_mins (%d values) and -F (%d values) length mismatch — ignoring --shot_mins",
                len(args.shot_mins), len(args.few_shot_size))
        else:
            args.size_to_min = dict(zip(args.few_shot_size, args.shot_mins))

    set_seed(args.seed)
    set_gpus(args.gpu_device)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')

    os.makedirs(args.output_dir, exist_ok=True)

    # Inject AC configuration files into args before creating config namespace
    if args.task in AC_FAMILY_TASKS:
        if not args.key_map or not os.path.exists(args.key_map):
            raise ValueError("Valid --key_map is required for AC-family tasks.")
        if not args.col_defs or not os.path.exists(args.col_defs):
            raise ValueError("Valid --col_defs is required for the AC/AC1 task.")

        with open(args.key_map, 'r', encoding='utf-8') as f:
            args.key_map = json.load(f)
        with open(args.col_defs, 'r', encoding='utf-8') as f:
            args.col_defs = json.load(f)

    config = load_config(os.path.join(args.model, "config.json"), intask=args.task)
    update_namespace(config, args)

    config.ac_similarity_threshold = args.ac_sim_threshold
    config.stanza_path             = args.stanza_path

    model_name = os.path.basename(os.path.normpath(args.model))
    
    args.score_pattern = "scores_" + args.shot_pattern
    args.pred_pattern = "pred_" + args.split_pattern
    args.prompt_pattern = "prompt_" + args.split_pattern
    
    # Import ProgressManager locally to avoid circular dependencies if any exist
    from main.progress_manager import ProgressManager
    progress_mgr = ProgressManager(
        args.output_dir, model_name, config.dataset, config.task, args.pred_pattern, args.score_pattern
    )
    if args.repredict and args.clear_progress:
        progress_mgr.clear_pred_state()

    trainer_class = get_trainer_class("LLM", config.task)

    # Trainer instance is cheap (just stores config); model weights are
    # loaded lazily via `_ensure_model_loaded` from inside process_split
    # right before the first predict() call. Keeps re-score runs fast when
    # every shot's predictions are already on disk.
    trainer = trainer_class(config)
    
    # Get ordered list of splits
    split_dirs = sorted(glob.glob(os.path.join(args.dataset_dir, "split*")))
    if not split_dirs:
        logger.error(f"No 'split' directories found in {args.dataset_dir}")
        return
    
    # Used for ordering the load_shot logic
    split_names = [os.path.basename(s) for s in split_dirs]

    logger.info(f"Found {len(split_dirs)} splits.")

    # Pre-load train + val across splits for the underfilled-factor heads-up
    # warning later. Matches the per-split pool used by process_split (which
    # also pulls from train.json + val.json) so the warning's "examples
    # available" count is consistent with what each split's few-shot can
    # actually draw on.
    all_train_data = []
    for split_dir in split_dirs:
        train_file = os.path.join(split_dir, "train.json")
        val_file   = os.path.join(split_dir, "val.json")
        if os.path.exists(train_file):
            td, _ = load_data(args.task, train_file, trainer_class.add_extra_info_fn, config)
            all_train_data.extend(td)
        if os.path.exists(val_file):
            vd, _ = load_data(args.task, val_file, trainer_class.add_extra_info_fn, config)
            all_train_data.extend(vd)

    def _min_factor_count(pool):
        """Return the minimum number of non-empty examples across all AC factors."""
        if (args.task not in AC_FAMILY_TASKS
                or not hasattr(config, 'key_map') or not config.key_map):
            return len(pool)
        structure = config.key_map.get("structure", {})
        all_keys = list(structure.get("accident_report", []))
        for mf in all_keys:
            all_keys.extend(structure.get(mf, []))
        if not all_keys:
            return len(pool)
        def has_val(ex, k):
            v = ex.get(k)
            return bool(v) and str(v).strip().lower() not in ['none', 'null', 'n/a', 'unknown', '']
        return min(sum(1 for ex in pool if has_val(ex, k)) for k in all_keys)

    for shot_size in config.few_shot_size:
        logger.info(f"=== Starting K={shot_size} ===")

        # Underfilled-factor heads-up: every factor is still predicted at
        # every shot, but log a warning when some factor has fewer than K
        # populated training examples across all splits — the user can
        # raise --skip_threshold to drop those factors at SCORING time.
        if shot_size > 0:
            available = _min_factor_count(all_train_data)
            if available < shot_size:
                logger.warning(
                    f"K={shot_size}: only {available} examples available for some factors "
                    f"across all splits. Predictions will still run; raise --skip_threshold > 0 "
                    f"to drop the underfilled factors at scoring time."
                )

        if not args.repredict:
            _, existing_total = progress_mgr.get_result(shot_size, None)
            if existing_total:
                logger.info(f"K={shot_size} global score file exists. Loading existing predictions to re-score...")

        scores_log = {
            "model": model_name, "k_shot": shot_size, "dataset": config.dataset,
            "task": config.task, "splits": {}, "aggregate": {}
        }

        # 1. Generate Preds (or load them if they exist)
        for split_dir in split_dirs:
            split_scores = process_split(
                split_dir, shot_size, trainer, args, config, trainer_class, model_name, progress_mgr
            )
            if split_scores:
                scores_log["splits"][os.path.basename(split_dir)] = split_scores

        # 2. Gather Data
        all_preds = progress_mgr.load_shot(shot_size, split_names=split_names)
        # ACH-family pred files are saved nested by trainer.save_predictions
        # — flatten so they align with load_data's always-flat gold.
        _src_fmt = "ACH" if args.task in ("ACH", "ACH1", "JHE", "IHE") else "AC"
        all_preds = convert_predictions(all_preds, _src_fmt, "AC", config)

        all_eval_data = []
        for split_dir in split_dirs:
            test_file = os.path.join(split_dir, "test.json")
            eval_data, _ = load_data(args.task, test_file, trainer_class.add_extra_info_fn, config)
            all_eval_data.extend(eval_data)

        # 3. Compute Global Score
        if all_preds:
            if len(all_preds) != len(all_eval_data):
                logger.warning(f"Size Mismatch! Preds: {len(all_preds)}, Truth: {len(all_eval_data)}")

            # Aggregate skip vector: always include --skip_columns; union with
            # the few-shot-based vector only when skip_threshold > 0.
            agg_skip = list(args.skip_columns or [])
            if (args.task in AC_FAMILY_TASKS
                    and args.skip_threshold > 0 and split_names):
                counts = {}
                for sn in split_names:
                    for col in (progress_mgr.get_skip_columns(shot_size, sn) or []):
                        counts[col] = counts.get(col, 0) + 1
                n = len(split_names)
                few_shot_skip = [c for c, k in counts.items() if (k / n) >= args.skip_threshold]
                agg_skip = sorted(set(agg_skip) | set(few_shot_skip))
                if few_shot_skip:
                    logger.info(f"[skip_threshold={args.skip_threshold}] Dropping {len(few_shot_skip)} columns "
                                f"from aggregate score: {few_shot_skip}")

            logger.info(f"Calculating GLOBAL Score for K={shot_size}...")
            global_scores = compute_scores(all_preds, all_eval_data, args.task,
                                           sim_threshold=getattr(config, "ac_similarity_threshold", 0.8),
                                           skip_columns=agg_skip,
                                           norm=args.norm,
                                           strict_keys=strict_keys_from_keymap(getattr(config, "key_map", None)))
            _strip_acc_type_for_e2(global_scores, args.task)
            scores_log["aggregate"] = global_scores
            scores_log["skipped_columns"] = agg_skip
            print_scores(global_scores, split=None, shot=shot_size)

            score_filename = args.score_pattern.format(
                model=model_name, dataset=config.dataset, task=config.task.lower(), shot=shot_size
            )
            with open(os.path.join(args.output_dir, score_filename), 'w') as f:
                json.dump(scores_log, f, indent=4)
            
            progress_mgr.complete_shot(shot_size, score_filename)
            del all_preds, all_eval_data
            gc.collect()

if __name__ == "__main__":
    main()