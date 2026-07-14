"""Progress + cross-model F1 comparison logic for `run.py -a check`.

Split out of run.py so the main script keeps its sweep-orchestration shape
clean; this module owns:

  * `make_progress_mgr` — read-only ProgressManager factory used wherever we
    need to inspect what's already been written for a (task, model, dataset)
    combination.
  * `supervised_status` — 'done' | 'eval_only' | 'train' verdict for a
    supervised combo (also used by run.py's generate path to decide whether
    to re-emit a training job).
  * `check_combo` — per-(task, dataset, model) progress line; optionally
    pools preds+golds into a caller-owned dict when `preds_golds` is passed.
  * `render_cross_model_comparison` — post-sweep renderer that takes that
    pooled dict and produces the cross-model F1 table (plus the SKIP COLUMNS
    diagnostic when `skip_unions` is supplied).

Path resolution stays in run.py — callers hand us already-resolved
(d_path, d_name, out_path) tuples. That way this module doesn't need to know
about CONFIG / smart_path / scratch-path substitution.
"""

import datetime
import glob as _glob
import io
import json
import logging
import os
import contextlib
from types import SimpleNamespace

from main.progress_manager import ProgressManager
# `convert_predictions` is imported lazily at each call site —
# TextEE.utils pulls in torch/stanza/numpy/_jsonnet at module load (several
# seconds on Windows), and `-a generate` never needs it.

logger = logging.getLogger(__name__)

_PRED_PATTERN  = "pred_{model}_{dataset}_{task}_{split}_k{shot}.json"
_SCORE_PATTERN = "scores_{model}_{dataset}_{task}_k{shot}.json"


def make_progress_mgr(out_path, model_key, d_name, task, train_dir=None):
    """Instantiate a read-only ProgressManager; returns None if output_dir
    doesn't exist yet (caller renders that as a fresh-combo placeholder).

    `train_dir` should be the trainer-states parent
    (`<trained_model_dir>/<task>/<dataset>/<model>/`) so the readonly check
    finds `training_progress.json` at the same location evaluate_supervised
    wrote it to. Falls back to `out_path` for back-compat / LLM combos."""
    if not os.path.isdir(out_path):
        return None
    return ProgressManager(out_path, model_key, d_name, task,
                           _PRED_PATTERN, _SCORE_PATTERN, readonly=True,
                           train_dir=train_dir)


def supervised_status(task, dataset_key, model_key, d_path, d_name, out_path,
                      train_dir=None):
    """'done' | 'eval_only' | 'train' verdict for a supervised combo.

    - 'done'      — k=0 score JSON exists; nothing left to do.
    - 'eval_only' — every split has a trained checkpoint but no score JSON;
                    inference is all that's left.
    - 'train'     — at least one split is untrained.

    `train_dir` mirrors the location used by evaluate_supervised to write
    training_progress.json (next to the trainer states).
    """
    mgr = make_progress_mgr(out_path, model_key, d_name, task, train_dir=train_dir)
    if mgr is None:
        return "train"

    if mgr.is_done(0):
        return "done"

    split_names = [os.path.basename(s)
                   for s in sorted(_glob.glob(os.path.join(d_path, "split*")))]
    if split_names and mgr.all_splits_trained(split_names):
        return "eval_only"

    return "train"


def _load_split_golds(d_path):
    """Concatenate raw test.json records across every split* dir."""
    golds = []
    for split_dir in sorted(_glob.glob(os.path.join(d_path, "split*"))):
        test_file = os.path.join(split_dir, "test.json")
        if not os.path.exists(test_file):
            continue
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    golds.append(json.loads(line))
    return golds


def _load_split_golds_per_split(d_path):
    """Like `_load_split_golds`, but keeps each split separate.

    Returns dict[split_name -> list[gold_record]]. Used by analysis mode to
    compute F1 averaged across splits (vs the pooled micro-F1 used by the
    cross-model comparison)."""
    out = {}
    for split_dir in sorted(_glob.glob(os.path.join(d_path, "split*"))):
        test_file = os.path.join(split_dir, "test.json")
        if not os.path.exists(test_file):
            continue
        split_name = os.path.basename(split_dir)
        recs = []
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        if recs:
            out[split_name] = recs
    return out


def _load_shot_per_split(mgr, shot):
    """Same as `mgr.load_shot` but returns dict[split_name -> preds] instead
    of a flat concatenation. Used by analysis mode for split-averaged F1."""
    k_key = f"k_shot_{shot}"
    if k_key not in mgr.pred_state:
        return {}
    out = {}
    for split_name, split_path in mgr.pred_state[k_key].items():
        if not split_name.startswith("split"):
            continue
        try:
            out[split_name] = mgr.load_split(split_path)
        except Exception as exc:
            logger.warning("⚠️  failed loading %s for shot %s: %s",
                            split_name, shot, exc)
    return out


def _supervised_classification_skip(classes_map, preds=None):
    """Classification column names that the supervised trainer couldn't emit.

    NER-only trainers (e.g. DyGIEpp/OneIE) have no cls head, so every gold
    task/severity span hits the recall denominator with no chance of a
    match. Those columns belong in `skip_columns` so recall isn't
    artificially deflated.

    Trainers WITH a cls head (TagPrime-C/CR) do predict classifications —
    skipping them would hide real signal. When `preds` is supplied, we
    detect head presence per-column: a classification key is skipped only
    when zero pred records carry a value for it. Without `preds` (legacy
    call sites), we fall back to skipping every classification key —
    matches the original NER-only behavior.

    `accident_type` is always dropped from the skip list because
    compute_AC_flat_score's IGNORE_KEYS already excludes it.
    """
    if not classes_map:
        return []
    candidates = [k for k in classes_map.keys() if k != "accident_type"]
    if preds is None:
        return candidates
    return [
        k for k in candidates
        if not any(rec.get(k) for rec in preds if isinstance(rec, dict))
    ]


def _maybe_convert_ie_to_ac(preds, structure, classes_map):
    """Convert supervised-trainer IE-format predictions to the flat AC dict
    format that compute_AC_flat_score consumes. Detection mirrors
    error_analysis.analyze_combination: if the first record has
    `entity_mentions` or `event_mentions`, the whole batch is IE.

    No-op for already-flat AC predictions (LLM trainers) or empty lists.
    Returns the (possibly converted) preds list.
    """
    if not preds:
        return preds
    first = preds[0]
    if "entity_mentions" not in first and "event_mentions" not in first:
        return preds
    cfg = SimpleNamespace(key_map={
        "structure": structure or {},
        "task": {"classification": {"classes": classes_map or {}}},
    })
    from TextEE.utils import convert_predictions
    return convert_predictions(preds, "IE", "AC", cfg)


def _recompute_and_print(mgr, d_path, shot, *,
                         sim_threshold, is_supervised, skip_threshold,
                         structure=None, classes_map=None,
                         override_skip=None, per_split_skip=None,
                         base_skip=None, norm=False):
    """Fresh per-split + aggregate score render for one (model, dataset, shot).

    Replaces `mgr.print_aggregate_scores` (which reads stored scores baked at
    predict time) so callers can change `sim_threshold`, change the skip
    vector, or pick up newer scorer logic without re-running prediction.

    `override_skip` / `per_split_skip` follow the same semantics as
    `print_aggregate_scores`: a list overrides the AGGREGATE skip vector; a
    dict overrides per-split, falling back to `override_skip` for any split
    not in the dict, falling back to the stored per-split skip when neither
    override is supplied.

    Output shape matches `print_aggregate_scores` so the per-shot section in
    the check-action log looks the same — just with up-to-date numbers. A
    KEYWORD MATCH block is appended after each split (and after the
    aggregate) when any gold record carries `<factor>_keywords` annotations.
    """
    try:
        from TextEE.scorer import (compute_AC_scores, print_scores,
                                   compute_AC_keyword_scores,
                                   print_keyword_scores)
    except ImportError:
        from scorer import (compute_AC_scores, print_scores,
                            compute_AC_keyword_scores, print_keyword_scores)
    from TextEE.utils import convert_predictions

    src_fmt = "ACH"
    # Classification factors must match exactly — partial overlap like
    # "fatality" vs "fatalities" (ratio 0.89) would otherwise count as a
    # hit under sim=0.8. Extraction factors keep `sim_threshold`.
    strict_keys = list(classes_map.keys()) if classes_map else []

    split_dirs = sorted(_glob.glob(os.path.join(d_path, "split*")))
    split_names = [os.path.basename(sd) for sd in split_dirs]
    pooled_preds, pooled_golds = [], []

    for sd, sn in zip(split_dirs, split_names):
        split_preds = mgr.load_shot(shot, split_names=[sn])
        if is_supervised:
            # IE→flat-AC. Predictions are already flat after this, so we
            # skip the subsequent ACH→AC pass (which would treat them as
            # nested and drop fields).
            split_preds = _maybe_convert_ie_to_ac(split_preds, structure, classes_map)
        else:
            # LLM trainers save ACH-family preds nested; AC-family already flat.
            split_preds = convert_predictions(split_preds, src_fmt, "AC", config=None)
        test_file = os.path.join(sd, "test.json")
        if not os.path.exists(test_file):
            continue
        with open(test_file, "r", encoding="utf-8") as f:
            split_golds = [json.loads(line) for line in f if line.strip()]
        split_golds = convert_predictions(split_golds, src_fmt, "AC", config=None)
        if not split_preds or not split_golds:
            continue
        if len(split_preds) != len(split_golds):
            logger.warning(
                "⚠️  size mismatch [%s] k=%d: preds=%d golds=%d — skipping split",
                sn, shot, len(split_preds), len(split_golds))
            continue
        # Per-split cls-skip: only drops classification columns this split's
        # preds didn't populate (so TagPrime-C's construction_trade survives;
        # NER-only DyGIEpp's gets skipped).
        cls_skip = (_supervised_classification_skip(classes_map, split_preds)
                    if is_supervised else [])
        if per_split_skip is not None and sn in per_split_skip:
            split_skip = list(per_split_skip[sn])
        elif override_skip is not None:
            split_skip = list(override_skip)
        else:
            split_skip = list(mgr.get_skip_columns(shot, sn) or [])
        # Union with cls_skip (NER-only supervised) and base_skip (global
        # config skip_columns). Mirrors evaluate_llm.py's
        # `shot_skip = base_skip | split_skip` at predict time.
        split_skip = sorted(set(split_skip) | set(cls_skip) | set(base_skip or []))
        scores = compute_AC_scores(split_preds, split_golds,
                                   sim_threshold=sim_threshold,
                                   skip_columns=split_skip,
                                   norm=norm,
                                   strict_keys=strict_keys)
        print(f"    skip[{sn}]: {split_skip}")
        print_scores(scores, split=sn, shot=shot)
        if _has_keywords(split_golds):
            kw_scores = compute_AC_keyword_scores(split_preds, split_golds,
                                                  skip_columns=split_skip)
            print_keyword_scores(kw_scores, split=sn, shot=shot)
        pooled_preds.extend(split_preds)
        pooled_golds.extend(split_golds)

    if pooled_preds and pooled_golds:
        if override_skip is not None:
            agg_skip = list(override_skip)
        else:
            agg_skip = (
                list(mgr.aggregate_skip_columns(shot, split_names, skip_threshold))
                if skip_threshold and skip_threshold > 0 else []
            )
        agg_cls_skip = (_supervised_classification_skip(classes_map, pooled_preds)
                        if is_supervised else [])
        # Union with base_skip too — mirrors evaluate_llm.py's
        # `agg_skip = list(args.skip_columns) | few_shot_skip` at predict time.
        agg_skip = sorted(set(agg_skip) | set(agg_cls_skip) | set(base_skip or []))
        agg_scores = compute_AC_scores(pooled_preds, pooled_golds,
                                       sim_threshold=sim_threshold,
                                       skip_columns=agg_skip,
                                       norm=norm,
                                       strict_keys=strict_keys)
        print(f"    skip: {agg_skip}")
        print_scores(agg_scores, split=None, shot=shot)
        if _has_keywords(pooled_golds):
            kw_agg = compute_AC_keyword_scores(pooled_preds, pooled_golds,
                                               skip_columns=agg_skip)
            print_keyword_scores(kw_agg, split=None, shot=shot)



def _has_keywords(records):
    """True when at least one record carries a `*_keywords` parallel list.
    Cheap guard so the keyword-scoring block only fires for datasets that
    actually have annotations (ACCD, constee post-flatten)."""
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for k in rec:
            if isinstance(k, str) and k.endswith("_keywords"):
                return True
    return False


def _recompute_skipped_columns(mgr, d_path, shot, skip_threshold):
    """Re-aggregate per-split skip vectors from prediction_metadata.json under
    a caller-chosen threshold. threshold=1.0 → only skip if every split flagged
    the column; 0.5 → majority; ≤0 → never skip.
    """
    if skip_threshold is None or skip_threshold <= 0:
        return []
    split_names = [
        os.path.basename(s)
        for s in sorted(_glob.glob(os.path.join(d_path, "split*")))
    ]
    if not split_names:
        return []
    return mgr.aggregate_skip_columns(int(shot), split_names, skip_threshold)


def check_combo(job_name, task, dataset_key, model_key, shots_list,
                *, d_path, d_name, out_path, is_supervised, skip_threshold,
                sim_threshold,
                preds_golds=None, skip_unions=None, base_skip=None,
                structure=None, classes_map=None, norm=False,
                variant="", train_dir=None, model_display=None):
    """Per-(task, dataset, model) progress line + optional cross-model pool.

    Caller responsibilities (everything CONFIG-dependent stays in run.py):
      - resolve `d_path` / `d_name` / `out_path` via get_combo_paths
      - decide `is_supervised` via is_supervised_model
      - look up `skip_threshold` from the global config for this (model, task)
      - read `sim_threshold` from the global config — the per-shot recompute
        uses this so JSON edits to ac_similarity_threshold take effect
      - allocate `preds_golds` (when --aggregate is on) and optionally pass
        a cross-run `skip_unions` precompute when callers want every model
        to NaN the same factors
      - load `structure` + `classes_map` from key_mapping.json so supervised
        IE-format predictions get converted to flat AC dicts before pooling
        (no-op for LLM trainers, which already emit flat AC)

    When `preds_golds is None` (default check) this RECOMPUTES per-shot
    scores against the current sim_threshold + skip settings. When provided,
    it ALSO pools the per-split preds + golds into preds_golds keyed by
    (model_key, task, dataset_key, shot) — dataset is part of the key so
    multiple datasets never get merged into one F1 — for the later
    cross-model rescore in `render_cross_model_comparison`.
    """
    mgr = make_progress_mgr(out_path, model_key, d_name, task, train_dir=train_dir)
    if mgr is None:
        logger.info("  ⬜ [%s] no output dir yet", job_name)
        return

    if is_supervised:
        split_names = [os.path.basename(s)
                       for s in sorted(_glob.glob(os.path.join(d_path, "split*")))]
        trained = [s for s in split_names if mgr.is_trained(s)]
        train_str = f"{len(trained)}/{len(split_names)} splits trained"
    else:
        train_str = "n/a (LLM)"

    done_shots = [s for s in shots_list if mgr.is_done(int(s))]
    pend_shots = [s for s in shots_list if s not in done_shots]
    infer_str  = f"k={done_shots} done" if done_shots else "no k done"
    if pend_shots:
        infer_str += f"  |  pending: k={pend_shots}"

    icon = "✅" if not pend_shots else ("🔄" if done_shots else "⏳")
    norm_tag = "norm=on" if norm else "norm=off"
    logger.info("  %s [%s]  train: %s  |  infer: %s  |  sim=%s  |  %s",
                icon, job_name, train_str, infer_str, sim_threshold, norm_tag)

    if preds_golds is None:
        # Default check: RECOMPUTE per-shot scores so a caller-supplied
        # sim_threshold / skip override actually takes effect (the stored
        # scores_*.json were baked at predict time with whatever was active
        # then). When skip_unions is supplied, override the per-model
        # stored skip with the FRESH per-split + aggregate vectors recomputed
        # by collect_skip_unions. Supervised models train on the full set
        # and never have skip columns.
        for shot in done_shots:
            entry = ((skip_unions or {}).get((task, dataset_key, str(shot)))
                     if not is_supervised else None)
            override_skip = (entry.get("aggregate") or []) if entry else None
            per_split_skip = (entry.get("per_split") or {}) if entry else None
            _recompute_and_print(
                mgr, d_path, int(shot),
                sim_threshold=sim_threshold,
                is_supervised=is_supervised,
                skip_threshold=skip_threshold,
                structure=structure, classes_map=classes_map,
                override_skip=override_skip,
                per_split_skip=per_split_skip,
                base_skip=base_skip,
                norm=norm,
            )
        return

    # --aggregate: skip per-dataset prints; collect preds+golds so the caller
    # can rescore across datasets via compute_AC_scores. Skip-vectors are
    # re-aggregated using this model's skip_threshold so columns flagged by
    # only one split (under a high threshold) don't get hidden.
    if not done_shots:
        return
    from TextEE.utils import convert_predictions
    src_fmt = "ACH"
    golds = _load_split_golds(d_path)
    golds = convert_predictions(golds, src_fmt, "AC", config=None)
    # Per-split golds for analysis mode (split-averaged F1). Loaded once per
    # combo since the same gold set is reused across shots.
    golds_by_split = _load_split_golds_per_split(d_path)
    for sn in list(golds_by_split.keys()):
        golds_by_split[sn] = convert_predictions(
            golds_by_split[sn], src_fmt, "AC", config=None)
    for shot in done_shots:
        preds = mgr.load_shot(int(shot))
        if not preds or not golds:
            continue
        # Supervised trainers emit IE-format predictions (entity_mentions/
        # event_mentions); compute_AC_flat_score consumes flat AC dicts.
        # Convert before pooling so per-record key iteration in the scorer
        # sees the same factor keys as the gold side. IE→flat-AC produces
        # already-flat output; the ACH→AC pass below would drop fields,
        # so route supervised and LLM through separate conversions.
        if is_supervised:
            preds = _maybe_convert_ie_to_ac(preds, structure, classes_map)
        else:
            preds = convert_predictions(preds, src_fmt, "AC", config=None)
        # Variant-suffix the pool key's model id so multi-variant aggregate
        # runs keep each config's preds + golds separate (otherwise variants
        # sharing the same model_key would `extend` into one entry and
        # produce an apples-to-oranges F1).
        # Use the display name (typically the model_alts value — e.g.
        # "Llama-3.2-3B-Instruct" instead of the short alias "Llama3.2-3B")
        # for the cross-model label so the rendered table shows the full
        # HF/checkpoint name. Falls back to the alias when caller didn't
        # supply one. Internal lookups (skip_unions, supervised flag, etc.)
        # still key off model_key.
        display = model_display or model_key
        pool_mk = f"{display}_{variant}" if variant else display
        # dataset_key is part of the pool key (not just a footnote in
        # `datasets` below) so two datasets sharing (model, task, shot) never
        # get their preds+golds merged into one F1 — render_cross_model_
        # comparison / render_analysis_tables split back apart by dataset
        # before scoring (see `split_preds_golds_by_dataset`).
        entry = preds_golds.setdefault(
            (pool_mk, task, dataset_key, str(shot)),
            {
                "task": task,
                "preds": [],
                "golds": [],
                "datasets": [],
                "skip_columns": set(),
                "is_supervised": is_supervised,
                # Per-split breakdown: list of {name, preds, golds}. Used
                # by analysis mode to compute F1 sum-averaged across splits
                # (instead of the pooled micro-F1 the cross-model rescore
                # produces). Pre-conversion lines up with the AC-flat shape.
                "per_split": [],
            },
        )
        entry["preds"].extend(preds)
        entry["golds"].extend(golds)
        entry["datasets"].append(f"{task}/{dataset_key}")
        # Pull this shot's per-split predictions for analysis. Use the same
        # IE→AC / ACH→AC conversion as the pooled path so factor keys match.
        preds_by_split = _load_shot_per_split(mgr, int(shot))
        for sn, sp in preds_by_split.items():
            if sn not in golds_by_split:
                continue
            if is_supervised:
                sp_conv = _maybe_convert_ie_to_ac(sp, structure, classes_map)
            else:
                sp_conv = convert_predictions(sp, src_fmt, "AC", config=None)
            entry["per_split"].append({
                "name": sn,
                "preds": sp_conv,
                "golds": golds_by_split[sn],
            })
        # Skip columns are LLM-only — supervised models score every factor.
        if not is_supervised:
            entry["skip_columns"].update(
                _recompute_skipped_columns(mgr, d_path, shot, skip_threshold)
            )
        # Base skip (e.g. accident_report/id/accident_type for ACH) lives in
        # the global config and applies to BOTH supervised and LLM —
        # mirrors evaluate_llm.py's `agg_skip = list(args.skip_columns)` at
        # predict time. Stored as its own field so it's clearly distinct
        # from the few-shot-derived skip in `skip_columns`.
        if base_skip:
            entry.setdefault("base_skip", set()).update(base_skip)


def split_preds_golds_by_dataset(preds_golds):
    """Un-nest the dataset dimension from a preds_golds pool keyed by
    (model, task, dataset, shot) -> entry.

    Returns dict[dataset] -> {(model, task, shot): entry}, so each dataset's
    slice can be fed through the existing (model, task, shot)-keyed scoring
    pipeline (TextEE.scorer's print_score_comparison and friends, and
    data_analyzer's table builders) unchanged — datasets are never pooled
    together into one F1, each gets its own independent render.
    """
    by_dataset = {}
    for (mk, task, dataset, shot), entry in preds_golds.items():
        by_dataset.setdefault(dataset, {})[(mk, task, shot)] = entry
    return by_dataset


def per_dataset_log_path(log_path, dataset_key):
    """Derive a dataset-specific log path so parallel per-dataset renders
    never collide on the same output file.

    - directory / extensionless path → nest a `<dataset>/` subdir under it
      (the caller's own auto-naming, e.g. a timestamp, then lives inside
      that subdir, so same-second calls for different datasets don't clash).
    - concrete file path → insert the dataset name before the extension.
    """
    if log_path is None:
        return None
    if os.path.isdir(log_path) or not os.path.splitext(log_path)[1]:
        return os.path.join(log_path, dataset_key)
    root, ext = os.path.splitext(log_path)
    return f"{root}_{dataset_key}{ext}"


def render_cross_model_comparison(preds_golds, skip_unions, *,
                                  sim_threshold, structure,
                                  classes_map=None, log_path=None,
                                  norm=False):
    """Render the post-sweep cross-(model, kshot) F1 table, once per dataset.

    `preds_golds` is keyed by (model, task, dataset, shot) — split apart by
    dataset first (see `split_preds_golds_by_dataset`) so two datasets that
    happen to share (model, task, shot) never get their preds+golds merged
    into one F1. Each dataset gets its own table (and its own log file when
    `log_path` is set — see `per_dataset_log_path`).
    """
    if not preds_golds:
        return
    by_dataset = split_preds_golds_by_dataset(preds_golds)
    for dataset_key in sorted(by_dataset):
        logger.info("\n%s", f" DATASET: {dataset_key} ".center(78, "#"))
        _render_cross_model_comparison_one(
            by_dataset[dataset_key],
            {k: v for k, v in skip_unions.items() if k[1] == dataset_key}
            if skip_unions else skip_unions,
            sim_threshold=sim_threshold, structure=structure,
            classes_map=classes_map,
            log_path=per_dataset_log_path(log_path, dataset_key),
            norm=norm,
        )


def _render_cross_model_comparison_one(preds_golds, skip_unions, *,
                                       sim_threshold, structure,
                                       classes_map=None, log_path=None,
                                       norm=False):
    """Render the cross-(model, kshot) F1 table for ONE dataset's slice.

    Inputs:
      - `preds_golds`: dict[(model, task, shot)] → {preds, golds, datasets,
                       skip_columns, is_supervised} — one dataset's slice,
                       produced by `render_cross_model_comparison` above.
      - `skip_unions`: precomputed per-(task, dataset, shot) skip vectors
                       for this dataset only, or None. When non-empty,
                       surfaces the per-(task, dataset, shot) SKIP COLUMNS
                       diagnostic and uses the recomputed cross-model
                       aggregate so every row's NaN cells match.
      - `sim_threshold`: ac_similarity_threshold from the global config.
      - `structure`: factor structure dict (used so MAIN factors are CAPS and
                     subs render indented). None disables structure styling.
      - `log_path`: optional dir or file path; tees the rendered table to a
                    `cross_model_<ts>.log` (when a dir) or the path itself.

    The scorer import keeps the historical fallback so this module can also
    run from a non-package install layout.
    """
    if not preds_golds:
        return

    try:
        from TextEE.scorer import (compute_AC_scores, print_score_comparison,
                                   compute_AC_keyword_scores,
                                   print_keyword_score_comparison,
                                   print_score_comparison_latex,
                                   print_keyword_score_comparison_latex)
    except ImportError:
        from scorer import (compute_AC_scores, print_score_comparison,
                            compute_AC_keyword_scores,
                            print_keyword_score_comparison,
                            print_score_comparison_latex,
                            print_keyword_score_comparison_latex)

    # When skip_unions is supplied, the unified skip vector for each
    # (task, dataset, shot) is the FRESH recompute produced by
    # collect_skip_unions (using the trainer's algorithm against the actual
    # training data) — NOT a union of stored prediction_metadata entries
    # (which can be stale across older runs). `skip_unions` here was already
    # pre-filtered to this one dataset by the caller, so this just collapses
    # the dataset key back out to (task, shot) for lookup convenience below.
    cross_model_skip = {}
    if skip_unions:
        # Print the per-split + aggregate skip vectors for each
        # (task, dataset, shot) so the source of NaN cells in the cross-model
        # table is visible.
        logger.info("\n%s", "=" * 78)
        logger.info("%s", " SKIP COLUMNS (per (task, dataset, shot)) ".center(78, "="))
        logger.info("%s", "=" * 78)
        for (task_k, ds_k, shot_k) in sorted(
                skip_unions.keys(),
                key=lambda k: (k[0], k[1], int(k[2]))):
            entry = skip_unions[(task_k, ds_k, shot_k)]
            if not isinstance(entry, dict):
                entry = {"aggregate": entry or [], "per_split": {}}
            logger.info("\n  [%s/%s  k=%s]", task_k, ds_k, shot_k)
            for sn in sorted(entry.get("per_split", {}).keys()):
                logger.info("    skip[%s]: %s", sn, entry['per_split'][sn])
            logger.info("    skip[AGG]: %s", entry.get('aggregate') or [])
            cross_model_skip.setdefault((task_k, shot_k), set()).update(
                entry.get("aggregate") or [])
        logger.info("%s", "=" * 78)

    logger.info("\n%s", "=" * 78)
    logger.info("%s", f" CROSS-MODEL F1 COMPARISON  sim={sim_threshold}  norm={'on' if norm else 'off'} ".center(78, "="))
    logger.info("%s", "=" * 78)

    scores_by_mk = {}
    keyword_scores_by_mk = {}
    coverage = {}
    # model_key -> bool. Built from the pooled entries so the supervised flag
    # set at check_combo time flows through to print_score_comparison's
    # CLASS DIFF section.
    is_supervised_map = {}
    # Classification factors get exact-match (overlap=1.0) instead of
    # sim_threshold-fuzzy. Same rule applies in the cross-model rescore.
    strict_keys = list(classes_map.keys()) if classes_map else []
    for (mk, task_, shot), entry in preds_golds.items():
        if len(entry["preds"]) != len(entry["golds"]):
            logger.warning(
                "⚠️  size mismatch for %s task=%s k=%s: preds=%d golds=%d — skipping",
                mk, task_, shot, len(entry["preds"]), len(entry["golds"]))
            continue
        if entry.get("is_supervised"):
            # Per-model cls-skip: only drops classification columns this
            # specific supervised model didn't populate (TagPrime-C's
            # construction_trade survives; NER-only DyGIEpp's doesn't).
            skip_for_this = list(_supervised_classification_skip(
                classes_map, entry["preds"]))
        elif (task_, shot) in cross_model_skip:
            skip_for_this = list(cross_model_skip[(task_, shot)])
        else:
            skip_for_this = list(entry.get("skip_columns") or [])
        # Union with base_skip (global config skip_columns:
        # accident_report/id/accident_type). evaluate_llm.py applies the
        # same union at predict time; check-time scores diverge from
        # predict-time without it.
        skip_for_this = sorted(set(skip_for_this) | set(entry.get("base_skip") or []))
        scores_by_mk[(mk, task_, shot)] = compute_AC_scores(
            entry["preds"], entry["golds"],
            sim_threshold=sim_threshold,
            skip_columns=skip_for_this,
            norm=norm,
            strict_keys=strict_keys,
        )
        # Parallel keyword-match scoring — only emitted when this entry's
        # gold pool carries `*_keywords` annotations. Same skip vector as the
        # overlap scorer so factor-by-factor cells line up.
        if _has_keywords(entry["golds"]):
            keyword_scores_by_mk[(mk, task_, shot)] = compute_AC_keyword_scores(
                entry["preds"], entry["golds"],
                skip_columns=skip_for_this,
                strict_keys=strict_keys,
            )
        coverage[(mk, task_, shot)] = entry["datasets"]
        is_supervised_map[mk] = bool(entry.get("is_supervised"))

    # Capture the comparison table; tee to stdout, plus to a file if log_path.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_score_comparison(
            scores_by_mk, coverage=coverage, structure=structure,
            is_supervised_map=is_supervised_map,
        )
        if keyword_scores_by_mk:
            print_keyword_score_comparison(
                keyword_scores_by_mk, coverage=coverage, structure=structure,
                is_supervised_map=is_supervised_map,
            )
    output = buf.getvalue()
    logger.info("%s", output.rstrip("\n"))

    if log_path:
        lp = log_path
        if os.path.isdir(lp) or not os.path.splitext(lp)[1]:
            os.makedirs(lp, exist_ok=True)
            fname = f"cross_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            lp = os.path.join(lp, fname)
        else:
            os.makedirs(os.path.dirname(lp) or ".", exist_ok=True)
        with open(lp, "w", encoding="utf-8") as f:
            f.write(output)
        logger.info("📝 Comparison table written to %s", lp)

        # Sister .tex file with LaTeX-formatted tables. Kept separate from
        # the readable .log so the latter stays grep-friendly.
        tex_buf = io.StringIO()
        with contextlib.redirect_stdout(tex_buf):
            print_score_comparison_latex(
                scores_by_mk, structure=structure,
                caption="Cross-model F1 by factor group with span-overlap matching.",
                label="tab:cross-model-f1",
            )
            if keyword_scores_by_mk:
                print_keyword_score_comparison_latex(
                    keyword_scores_by_mk, structure=structure,
                    caption="Cross-model keyword-F1 by factor group with keyword string matching.",
                    label="tab:cross-model-kw-f1",
                )
        tex_output = tex_buf.getvalue()
        if tex_output.strip():
            tex_lp = os.path.splitext(lp)[0] + ".tex"
            with open(tex_lp, "w", encoding="utf-8") as f:
                f.write(tex_output)
            logger.info("📝 LaTeX comparison table written to %s", tex_lp)
