"""Train and evaluate any supervised TextEE model across all data splits.

Mirrors the multi-split / progress-tracking structure of evaluate_llm.py but
runs a supervised train → eval loop instead of few-shot LLM inference.

Compatible with run.py: the same -t/-d/-D/-m/-g/-b/-o/--retrain/--repredict/--split_pattern/
--shot_pattern flags are accepted so run.py can build SLURM commands for both
LLM and supervised models without special-casing.

Supported (model, task) pairs (must exist in TRAINER_REGISTRY in utils.py):
    E2E  : DyGIEpp, OneIE, Degree, AMRIE
    EAE  : EEQA, RCEE, QueryAndExtract, TagPrime, Degree, PAIE,
           XGear, BartGen, Ampere, CRFTagging
    ED   : EEQA, RCEE, QueryAndExtract, TagPrime, Degree, CRFTagging, UniST, CEDAR
    EARL : QueryAndExtract
    AC   : DyGIEpp, OneIE, AMRIE, Degree              (E2E-based)
           EEQA, RCEE, TagPrime, BartGen, XGear,
           PAIE, QueryAndExtract, Ampere               (EAE-based)

    E2E-based AC models train jointly on full documents; inference uses
    predict_ac_hierarchical (two-stage E2E).
    EAE-based AC models train on per-event EAE instances (AC_LOAD_FORMAT="EAE");
    inference uses predict_ac_eae_hierarchical (two-stage EAE).

Each split saves its checkpoint to:
    <output_dir>/<split>_checkpoint/

New args (not in evaluate_llm.py):
    --config        path to model config.json  [REQUIRED]
                    Contains all model/training hyperparams
                    (pretrained_model_name, max_epoch, batch_size, etc.)
                    CLI flags override individual keys in this file.
    --eval_only     skip training; load checkpoint from <output_dir>/<split>_checkpoint/
    --key_map       path to key_mapping.json   [REQUIRED for AC task]

Example:
    python evaluate_supervised.py \\
        -t AC -d data/processed_data/accd \\
        -D accd -m DyGIEpp \\
        --config global_data/configs/DyGIEpp/model.json \\
        --key_map global_data/key_mapping.json \\
        -g 0 -b 8 -o results/AC/accd/DyGIEpp
"""

import os, logging, json, glob, gc, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser, Namespace

import torch

from utils import (load_config, load_data, load_AC_supervised_data,
                   load_AC_eae_data, load_gold_data, convert_predictions,
                   set_seed, set_gpus, print_vram, get_trainer_class)
from scorer import (compute_scores, print_scores, strict_keys_from_keymap,
                     exclude_factor_from_overall)

# JHE/IHE predict accident_type (vs ACH/ACH1 where it's given as input).
# Strip accident_type from OVERALL after scoring so families are comparable.
E2_TASKS = {"JHE", "IHE"}


def _strip_acc_type_for_e2(scores, task):
    """Apply `exclude_factor_from_overall(... , 'accident_type')` only for
    E2-family tasks. No-op otherwise. Returns the scores dict for chaining."""
    if scores and task in E2_TASKS:
        exclude_factor_from_overall(scores, "accident_type")
    return scores
from main.progress_manager import ProgressManager

logger = logging.getLogger(__name__)

SHOT = 0  # supervised models have no few-shot dimension; use 0 as placeholder

AC_FAMILY_TASKS = frozenset({
    "AC", "AC1", "ACH", "ACH1", "JHE", "IHE",
})

PREDICT_METHOD = {
    "E2E":    "predict",
    "EAE":    "predictEAE",
    "ED":     "predict",
    "EARL":   "predictEARL",
    "AC":     "predictAC",
    "AC1":    "predictAC",
    "ACH":    "predictAC",
    "ACH1":   "predictAC",
    "JHE":    "predictAC",
    "IHE":    "predictAC",
}


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def _load_split_data(task, file_path, trainer_class, config, is_train=False):
    """Load data for a split, routing to the appropriate AC loader when task
    is in the AC family (`AC`/`ACH` and their AC1/ACH1 variants).

    For ACH (constee, nested input) records are first flattened into the
    flat AC shape via `convert_predictions(..., "ACH", "AC", ...)` so the
    same downstream IE conversion used for accd applies. From the
    supervised trainer's point of view ACH and AC are indistinguishable
    post-load.

    EAE-based AC trainers set AC_LOAD_FORMAT = 'EAE' on their class; those
    receive one-instance-per-event EAE data during training/dev so their
    train() method gets the format it expects. Eval (test) data is always
    loaded as E2E document-level instances regardless of model family.
    """
    # AC family includes E2 variants. From the trainer's perspective E2AC*
    # is identical to AC* (same data shape, same model); the E2 prefix only
    # changes whether accident_type is in skip_columns. ACH / E2ACH* records
    # arrive nested and are flattened to the flat AC shape before downstream
    # IE conversion. Routing all eight variants here ensures they go through
    # load_AC_supervised_data / load_AC_eae_data, which build proper
    # type_sets — load_data's ACH branch returns type_set={} and would leave
    # the trainer with no vocab.
    if task in AC_FAMILY_TASKS:
        with open(file_path, encoding="utf-8") as f:
            raw_records = [json.loads(line) for line in f if line.strip()]
        if task in ("ACH", "ACH1", "JHE", "IHE"):
            # nested constee → flat AC; everything downstream operates on
            # the flat shape (load_AC_supervised_data → convert_AC_to_EAE).
            raw_records = convert_predictions(raw_records, "ACH", "AC", config)
        if is_train and getattr(trainer_class, "AC_LOAD_FORMAT", "E2E") == "EAE":
            return load_AC_eae_data(raw_records, trainer_class.add_extra_info_fn, config)
        return load_AC_supervised_data(raw_records, trainer_class.add_extra_info_fn, config)
    return load_data(task, file_path, trainer_class.add_extra_info_fn, config)


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def process_split(split_dir, args, config, trainer_class, progress_mgr):
    split_name = os.path.basename(split_dir)
    train_file = os.path.join(split_dir, "train.json")
    val_file   = os.path.join(split_dir, "val.json")
    test_file  = os.path.join(split_dir, "test.json")
    _tmd = getattr(config, "trained_model_dir", None)
    if _tmd:
        ckpt_dir = os.path.join(_tmd, config.task, config.dataset,
                                args.model_type, split_name)
    else:
        ckpt_dir = os.path.join(args.output_dir, f"{split_name}_checkpoint")

    pred_filename = args.pred_pattern.format(
        model=args.model_type, dataset=config.dataset,
        task=config.task.lower(), split=split_name, shot=SHOT,
    )
    pred_output_path = os.path.join(args.output_dir, pred_filename)

    if not os.path.exists(test_file):
        logger.warning(f"Test file not found for {split_name}, skipping.")
        return None

    # ---- resume check ----
    existing_fname, _ = progress_mgr.get_result(SHOT, split_name)
    if not (args.retrain or args.repredict) and existing_fname and os.path.exists(
            os.path.join(args.output_dir, existing_fname)):
        logger.info(f"--> [Resume] {split_name} already completed: {existing_fname}")
        try:
            split_predictions = progress_mgr.load_split(existing_fname)
            gold_data, _ = load_gold_data(config.task, test_file, trainer_class.add_extra_info_fn, config)
            split_scores = compute_scores(convert_predictions(split_predictions,
                                "IE" if config.task in AC_FAMILY_TASKS else config.task,
                                "AC"  if config.task in AC_FAMILY_TASKS else config.task,
                                config), gold_data, config.task,
                         sim_threshold=getattr(config, "ac_similarity_threshold", 0.8),
                         skip_columns=args.skip_columns,
                         norm=args.norm,
                         strict_keys=strict_keys_from_keymap(getattr(config, "key_map", None)))
            _strip_acc_type_for_e2(split_scores, config.task)
            print_scores(split_scores, split=split_name, shot=SHOT, stage="test")
            return split_scores
        except Exception as exc:
            logger.warning(f"Failed to score existing file, re-running: {exc}")

    print_vram(f"Before {split_name}")

    # ---- load data ----
    eval_data,  eval_type_set  = _load_split_data(config.task, test_file,  trainer_class, config, is_train=False)
    train_data, train_type_set = _load_split_data(config.task, train_file, trainer_class, config, is_train=True)
    dev_data,   dev_type_set   = _load_split_data(config.task, val_file,   trainer_class, config, is_train=True)

    # Union type_set across splits so the trainer's vocab covers everything seen.
    type_set = None
    for ts in (train_type_set, dev_type_set, eval_type_set):
        if not ts:
            continue
        if type_set is None:
            type_set = {k: set(v) for k, v in ts.items()}
        else:
            for k, v in ts.items():
                type_set.setdefault(k, set()).update(v)

    # ---- per-split output dir ----
    split_config = Namespace(**vars(config))
    split_config.output_dir = ckpt_dir
    # Plumb split_name explicitly so trainers can label per-epoch dev
    # prints (`[split1] k=0`) without resorting to output_dir basename
    # hacks. Ignored by the param-match check below.
    split_config.split_name = split_name
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- check if existing checkpoint was trained with same params ----
    _IGNORE_KEYS = {"output_dir", "cache_dir", "log_path", "trained_model_dir",
                    "gpu_device", "eval_batch_size", "split_name"}
    ckpt_state   = os.path.join(ckpt_dir, "best_model.state")
    ckpt_cfg     = os.path.join(ckpt_dir, "config.json")
    ckpt_exists  = os.path.exists(ckpt_state)
    params_match = False
    if ckpt_exists and os.path.exists(ckpt_cfg):
        try:
            with open(ckpt_cfg) as _f:
                saved = json.load(_f)
            current = vars(split_config)
            params_match = all(
                current.get(k) == saved.get(k)
                for k in set(current) | set(saved)
                if k not in _IGNORE_KEYS
            )
        except Exception:
            pass

    with open(ckpt_cfg, "w") as f:
        json.dump(vars(split_config), f, indent=4)

    # ---- train or load existing checkpoint ----
    trainer = trainer_class(split_config, type_set=type_set)
    if ckpt_exists and params_match and not args.retrain:
        logger.info("=" * 60)
        logger.info(f"[SKIP TRAIN] same-config checkpoint detected for {split_name}")
        logger.info(f"            ckpt: {ckpt_state}")
        logger.info(f"            cfg : {ckpt_cfg}")
        logger.info(f"            (use --retrain to override)")
        logger.info("=" * 60)
        trainer.load_model(checkpoint=ckpt_dir)
        progress_mgr.complete_training(split_name, ckpt_dir)
    else:
        if args.eval_only:
            raise FileNotFoundError(
                f"--eval_only set but no matching checkpoint found at {ckpt_dir}")
        if ckpt_exists and not params_match:
            logger.info(f"[RETRAIN] config differs from saved checkpoint — retraining {split_name}")
        elif args.retrain and ckpt_exists:
            logger.info(f"[RETRAIN] --retrain set — overwriting checkpoint at {ckpt_dir}")
        else:
            logger.info(f"[TRAIN] no checkpoint at {ckpt_dir} — fresh training for {split_name}")
        logger.info(f"Training {args.model_type} on {split_name}...")
        trainer.train(train_data, dev_data)
        trainer.load_model(checkpoint=ckpt_dir)
        progress_mgr.complete_training(split_name, ckpt_dir)

    # ---- predict ----
    logger.info(f"Predicting on {split_name} test set...")
    predict_fn = getattr(trainer, PREDICT_METHOD.get(config.task, "predict"))
    pred_output = predict_fn(eval_data)
    split_predictions = pred_output[0] if isinstance(pred_output, tuple) else pred_output

    # ---- save predictions ----
    with open(pred_output_path, "w", encoding="utf-8") as f:
        for pred in split_predictions:
            f.write(json.dumps(pred) + "\n")
    logger.info(f"Saved predictions → {pred_output_path}")

    progress_mgr.complete_split(SHOT, split_name, pred_filename)

    gold_data, _ = load_gold_data(config.task, test_file, trainer_class.add_extra_info_fn, config)
    split_scores = compute_scores(convert_predictions(split_predictions,
                                "IE" if config.task in AC_FAMILY_TASKS else config.task,
                                "AC"  if config.task in AC_FAMILY_TASKS else config.task,
                                config), gold_data, config.task,
                         sim_threshold=getattr(config, "ac_similarity_threshold", 0.8),
                         skip_columns=args.skip_columns,
                         norm=args.norm,
                         strict_keys=strict_keys_from_keymap(getattr(config, "key_map", None)))
    _strip_acc_type_for_e2(split_scores, config.task)
    print_scores(split_scores, split=split_name, shot=SHOT, stage="test")

    print_vram(f"After {split_name}")
    del trainer, train_data, dev_data, eval_data, split_predictions
    gc.collect()
    torch.cuda.empty_cache()

    return split_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description=__doc__)

    # ---- shared with evaluate_llm.py (same short flags) ----
    parser.add_argument("-t", "--task", required=True,
                        choices=["E2E", "EAE", "ED", "EARL",
                                 "AC", "AC1", "ACH", "ACH1", "JHE", "IHE"],
                        help="IE task type")
    parser.add_argument("-d", "--dataset_dir", required=True,
                        help="Root folder containing split1/, split2/, ...")
    parser.add_argument("-D", "--dataset",    required=True,
                        help="Dataset name tag  (e.g. accd)")
    parser.add_argument("-m", "--model_type", required=True,
                        help="Model architecture name  (e.g. DyGIEpp, OneIE, EEQA)")
    parser.add_argument("-s", "--seed",       type=int, default=42)
    parser.add_argument("-g", "--gpu_device", type=int, nargs="+", default=[0])
    parser.add_argument("-b", "--eval_batch_size", type=int, default=8)
    parser.add_argument("-o", "--output_dir", default="./results")
    parser.add_argument("--retrain",   action="store_true",
                        help="Force retrain even if a matching checkpoint exists (also triggers re-predict)")
    parser.add_argument("--repredict", action="store_true",
                        help="Force re-prediction even if pred file already exists")
    parser.add_argument("--clear_progress", action="store_true",
                        help="When combined with --repredict, clears prediction_progress.json; "
                             "when combined with --retrain, clears both progress files")
    parser.add_argument("--split_pattern",
                        default="{model}_{dataset}_{task}_{split}_k{shot}.json")
    parser.add_argument("--shot_pattern",
                        default="{model}_{dataset}_{task}_k{shot}.json")

    # ---- new args (not in evaluate_llm.py) ----
    parser.add_argument("--config", required=True,
                        help="Path to model config.json  "
                             "(pretrained_model_name, max_epoch, batch_size, …)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; load existing checkpoint from output_dir")
    parser.add_argument("--backbone_path", default="",
                        help="Full path to the pretrained backbone model folder (e.g. /models/roberta-large)")
    parser.add_argument("--trained_model_dir", default=None,
                        help="Exact folder where trained model checkpoints are saved")
    parser.add_argument("--ac_sim_threshold", type=float, default=0.8,
                        help="Similarity threshold for AC scoring")
    parser.add_argument("--norm", action="store_true",
                        help="Enable canonical normalization (lowercase + "
                             "trailing-punct strip + whitespace collapse) "
                             "inside parse_ac_texts when scoring. Off by "
                             "default — scoring is then case- and punct-strict.")
    parser.add_argument("--skip_columns", nargs="+", default=[],
                        help="Role columns to strip from AC records before "
                             "train/eval. Sourced from `skip_columns` in the "
                             "global mapping (replaces the legacy `drop_cols` "
                             "field that used to live in model.json).")
    parser.add_argument("--key_map",
                        help="Path to key_mapping.json  [required for AC task]")
    parser.add_argument("--stanza_path", default=None,
                        help="Path to stanza models directory (passed to stanza.Pipeline(dir=...))")
    parser.add_argument("--log_path", default="logs/",
                        help="Log directory (from the active global_mapping's log_dir). "
                             "Used to populate config.log_path now that supervised configs "
                             "no longer carry their own log_path.")

    args = parser.parse_args()

    if args.task in AC_FAMILY_TASKS and not args.key_map:
        parser.error(f"--key_map is required when -t {args.task}")

    set_seed(args.seed)
    set_gpus(args.gpu_device)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- load key_map if provided ----
    key_map = {}
    if args.key_map:
        with open(args.key_map, encoding="utf-8") as f:
            key_map = json.load(f)

    # ---- build config from file, then apply CLI overrides ----
    config = load_config(args.config, intask=args.task)
    config.dataset         = args.dataset
    config.task            = args.task
    config.model_type      = args.model_type
    config.gpu_device      = (args.gpu_device[0]
                              if isinstance(args.gpu_device, list)
                              else args.gpu_device)
    config.eval_batch_size = args.eval_batch_size
    config.output_dir      = args.output_dir
    if key_map:
        config.key_map = key_map

    # ---- resolve backbone and trained_model_dir from args ----
    if args.backbone_path and os.path.isdir(args.backbone_path):
        logger.info(f"Backbone resolved: {getattr(config, 'pretrained_model_name', '')} -> {args.backbone_path}")
        config.pretrained_model_name = args.backbone_path
    config.trained_model_dir       = args.trained_model_dir
    config.ac_similarity_threshold = args.ac_sim_threshold
    config.stanza_path             = args.stanza_path
    config.log_path                = args.log_path
    config.skip_columns            = list(args.skip_columns)

    logger.info(f"Config: model={config.model_type}, task={config.task}, dataset={config.dataset}, "
                f"backbone={getattr(config, 'pretrained_model_name', 'N/A')}, "
                f"epochs={getattr(config, 'max_epoch', 'N/A')}, batch={getattr(config, 'batch_size', 'N/A')}")

    trainer_class = get_trainer_class(args.model_type, args.task)

    args.pred_pattern  = "pred_"   + args.split_pattern
    args.score_pattern = "scores_" + args.shot_pattern

    # Training progress sits next to the per-split checkpoints, matching the
    # `<trained_model_dir>/<task>/<dataset>/<model>/<split>` layout used in
    # `process_split`. Without `trained_model_dir`, checkpoints land under
    # `output_dir/<split>_checkpoint`, so the progress file stays there too.
    if args.trained_model_dir:
        train_progress_dir = os.path.join(args.trained_model_dir, config.task,
                                          config.dataset, args.model_type)
        os.makedirs(train_progress_dir, exist_ok=True)
    else:
        train_progress_dir = args.output_dir

    progress_mgr = ProgressManager(
        args.output_dir, args.model_type, config.dataset,
        config.task, args.pred_pattern, args.score_pattern,
        train_dir=train_progress_dir,
    )
    if args.clear_progress and args.retrain:
        progress_mgr.clear_train_state()
    if args.clear_progress and (args.repredict or args.retrain):
        progress_mgr.clear_pred_state()

    split_dirs  = sorted(glob.glob(os.path.join(args.dataset_dir, "split*")))
    if not split_dirs:
        logger.error(f"No split* directories found in {args.dataset_dir}")
        return

    split_names = [os.path.basename(s) for s in split_dirs]
    logger.info(f"Found {len(split_dirs)} splits: {split_names}")

    scores_log = {
        "model":     args.model_type,
        "k_shot":    SHOT,
        "dataset":   config.dataset,
        "task":      config.task,
        "splits":    {},
        "aggregate": {},
    }

    # ---- per-split train + eval ----
    for split_dir in split_dirs:
        split_scores = process_split(
            split_dir, args, config, trainer_class, progress_mgr,
        )
        if split_scores:
            scores_log["splits"][os.path.basename(split_dir)] = split_scores

    # ---- aggregate across all splits ----
    all_preds = progress_mgr.load_shot(SHOT, split_names=split_names)
    all_eval_data = []
    for split_dir in split_dirs:
        test_file = os.path.join(split_dir, "test.json")
        if not os.path.exists(test_file):
            continue
        data, _ = load_gold_data(config.task, test_file, trainer_class.add_extra_info_fn, config)
        all_eval_data.extend(data)

    if all_preds:
        if len(all_preds) != len(all_eval_data):
            logger.warning(
                f"Size mismatch — preds: {len(all_preds)}, gold: {len(all_eval_data)}"
            )
        logger.info("Calculating aggregate scores...")
        global_scores = compute_scores(convert_predictions(all_preds,
                                       "IE" if config.task in AC_FAMILY_TASKS else config.task,
                                       "AC"  if config.task in AC_FAMILY_TASKS else config.task,
                                       config), all_eval_data, config.task,
                                       sim_threshold=getattr(config, "ac_similarity_threshold", 0.8),
                                       skip_columns=args.skip_columns,
                                       norm=args.norm,
                                       strict_keys=strict_keys_from_keymap(getattr(config, "key_map", None)))
        _strip_acc_type_for_e2(global_scores, config.task)
        scores_log["aggregate"] = global_scores
        print_scores(global_scores, split=None, shot=SHOT)

        score_filename = args.score_pattern.format(
            model=args.model_type, dataset=config.dataset,
            task=config.task.lower(), shot=SHOT,
        )
        with open(os.path.join(args.output_dir, score_filename), "w") as f:
            json.dump(scores_log, f, indent=4)

        progress_mgr.complete_shot(SHOT, score_filename)


if __name__ == "__main__":
    main()
