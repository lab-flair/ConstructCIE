"""Shared ACTrainer mixin classes for the AC (Accident Classification) task.

E2EACMixin  — for joint entity+event models (DyGIEpp, OneIE, Degree, AMRIE).
              Overrides predict() to convert tuple-based events → IE format.
              predictAC() drives two-stage predict_ac_hierarchical (defined in
              this module).

EAEACMixin  — for argument-extraction models (EEQA, RCEE, TagPrime, …).
              predictAC() drives predict_ac_eae_hierarchical (defined in this
              module). AC_LOAD_FORMAT = "EAE" tells _load_split_data to use EAE
              instances when loading training / dev data for these models.

              Shared utilities exposed for all EAE-based AC trainers:
                * `_role_loss_weights`   — parses config.group_weight (XGear schema).
                * `_seq_weight(args, candidate=None)` — per-sequence weight
                  policy: max role weight in the gold args (AC-C-style) or
                  the candidate role's weight (AC-CR / per-candidate models).
                * `_save_best_and_check_early_stop(...)` — encapsulates the
                  save-best policy (F1 strict-improve, loss tiebreak) and
                  the early-stop counter driven by `config.early_stop_epochs`.
                * Inference mode (`config.ac_predict_mode = "parallel" |
                  "hierarchical"`) is read by `predict_ac_eae_hierarchical`
                  (below) — automatically applies to every trainer that
                  inherits this mixin's `predictAC`. Default = "hierarchical".

The hierarchical-prediction helpers (`predict_ac_hierarchical`,
`predict_ac_eae_hierarchical`, `_enforce_sub_factor_spans`) used to live in
TextEE/utils.py but are only ever called by the supervised AC trainers in
this directory tree. Keeping them next to the mixins narrows utils.py to
data-loading primitives shared across all tasks/models.
"""

import logging
import os
import pprint
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)


def parse_group_weight(config):
    """Parse `config.group_weight` (XGear schema) into a {role: weight} dict.

    Schema:
        "group_weight": [
            {"roles": ["body_part", "object", ...], "weight": 1.0},
            {"roles": ["mgmt_factor", "train_gap", ...], "weight": 5.0},
            ...
        ]
    Roles not listed default to weight 1.0 at lookup time. Empty / missing
    `group_weight` yields an empty dict (uniform weighting → no-op).
    """
    role_to_weight = {}
    groups = getattr(config, "group_weight", None) or []
    for g in groups:
        if not isinstance(g, dict):
            continue
        w = float(g.get("weight", 1.0))
        for r in g.get("roles", []):
            role_to_weight[r] = w
    return role_to_weight


class E2EACMixin:
    """Mixin for E2E-based AC trainers."""

    def _to_ie_format(self, pred):
        """Convert tuple-style events output to standard IE format.

        Input  (all E2E models):
            events: [{"trigger": (s, e, type), "arguments": [(s, e, role), …]}]

        Output:
            entity_mentions: [{"entity_type": role, "start": s, "end": e}]
            event_mentions:  [{"event_type": type, "trigger": {…}, "arguments": […]}]

        Argument roles double as entity_type so predict_ac_hierarchical can
        locate main-factor spans by entity_type.
        """
        entity_mentions = []
        event_mentions  = []
        seen_spans      = set()

        for event in pred.get("events", []):
            trig_start, trig_end, event_type = event["trigger"]
            ie_args = []
            for arg_start, arg_end, role in event.get("arguments", []):
                ie_args.append({"role": role, "start": arg_start, "end": arg_end})
                span_key = (arg_start, arg_end, role)
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    entity_mentions.append({
                        "entity_type": role,
                        "start":       arg_start,
                        "end":         arg_end,
                    })
            event_mentions.append({
                "event_type": event_type,
                "trigger":    {"start": trig_start, "end": trig_end},
                "arguments":  ie_args,
            })

        return {
            "doc_id":          pred["doc_id"],
            "wnd_id":          pred["wnd_id"],
            "tokens":          pred["tokens"],
            "text":            pred.get("text", ""),
            "entity_mentions": entity_mentions,
            "event_mentions":  event_mentions,
        }

    def predict(self, data, **kwargs):
        """E2E predict returning standard IE format (entity_mentions + event_mentions)."""
        raw_preds = self.internal_predict(data, **kwargs)
        return [self._to_ie_format(p) for p in raw_preds]

    def predictAC(self, data, **kwargs):
        """Two-stage hierarchical E2E inference for the AC task."""
        key_map = getattr(self.config, "key_map", {})
        return predict_ac_hierarchical(self, self.__class__, data, self.config, key_map)

    def _ac_dev_eval(self, dev_data):
        """Run predictAC on dev_data, print AC scores, return overall F1 for model selection."""
        from utils import ie_preds_to_flat_ac, e2e_to_flat_ac_gold
        from scorer import compute_AC_scores, print_scores, strict_keys_from_keymap

        key_map   = getattr(self.config, "key_map", {})
        acc_types = ((key_map or {}).get("task", {})
                                    .get("classification", {})
                                    .get("classes", {})
                                    .get("accident_type", []))
        skip_columns = list(getattr(self.config, "skip_columns", None) or [])
        # `split_name` is plumbed in by evaluate_supervised so per-epoch dev
        # tables get labeled (`[split1] k=0`). None falls back to "GLOBAL".
        split_name = getattr(self.config, "split_name", None)

        self.model.eval()
        with torch.no_grad():
            preds_ie = self.predictAC(dev_data)
        self.model.train()

        flat_preds    = ie_preds_to_flat_ac(preds_ie, acc_types)
        flat_gold     = e2e_to_flat_ac_gold(dev_data, acc_types)
        sim_threshold = getattr(self.config, "ac_similarity_threshold", 0.8)
        ac_scores     = compute_AC_scores(flat_preds, flat_gold,
                                          sim_threshold=sim_threshold,
                                          skip_columns=skip_columns,
                                          strict_keys=strict_keys_from_keymap(key_map))
        print_scores(ac_scores, split=split_name, shot=0, stage="val")
        return ac_scores["factors"]["overall"]["f1"]


class EAEACMixin:
    """Mixin for EAE-based AC trainers.

    Training / dev data are loaded as EAE instances (one per event, gold trigger).
    Inference uses two-stage keyword-detected + EAE extraction.
    """
    AC_LOAD_FORMAT = "EAE"

    # default loss weight for any role not listed in config.group_weight
    # (or when group_weight itself is missing / empty).
    DEFAULT_ROLE_WEIGHT = 1.0

    def predictEAE(self, data, **kwargs):
        """Public EAE prediction interface (processes raw instances and returns predictions)."""
        return self.internal_predict(data, **kwargs)

    def predictAC(self, data, **kwargs):
        """Two-stage hierarchical EAE inference for the AC task."""
        key_map = getattr(self.config, "key_map", {})
        return predict_ac_eae_hierarchical(self, self.__class__, data, self.config, key_map)

    def _ac_dev_eval(self, dev_data):
        """Run predictAC on dev_data, print AC scores, return overall F1 for
        model selection. Subclasses (e.g. TagPrimeE2ACTrainer) override
        `predictAC` itself to swap the cascade — so dev eval automatically
        runs the same predict path as test inference."""
        from utils import ie_preds_to_flat_ac, eae_to_flat_ac_gold
        from scorer import compute_AC_scores, print_scores, strict_keys_from_keymap

        key_map   = getattr(self.config, "key_map", {})
        acc_types = ((key_map or {}).get("task", {})
                                    .get("classification", {})
                                    .get("classes", {})
                                    .get("accident_type", []))
        skip_columns = list(getattr(self.config, "skip_columns", None) or [])
        # `split_name` is plumbed in by evaluate_supervised so per-epoch dev
        # tables get labeled (`[split1] k=0`). None falls back to "GLOBAL".
        split_name = getattr(self.config, "split_name", None)

        self.model.eval()
        with torch.no_grad():
            preds_ie = self.predictAC(dev_data)
        self.model.train()

        flat_preds    = ie_preds_to_flat_ac(preds_ie, acc_types)
        flat_gold     = eae_to_flat_ac_gold(dev_data, acc_types)
        sim_threshold = getattr(self.config, "ac_similarity_threshold", 0.8)
        ac_scores     = compute_AC_scores(flat_preds, flat_gold,
                                          sim_threshold=sim_threshold,
                                          skip_columns=skip_columns,
                                          strict_keys=strict_keys_from_keymap(key_map))
        print_scores(ac_scores, split=split_name, shot=0, stage="val")
        return ac_scores["factors"]["overall"]["f1"]

    # ---------------- shared helpers (loss weighting + early stop) -----------

    @property
    def _role_loss_weights(self):
        """{role: weight} dict cached per trainer instance.

        Returns an empty dict when no group_weight is configured. Use
        `_role_weight(role)` for safe lookup with the 1.0 default; this
        property is for callers that need the raw mapping.
        """
        cached = getattr(self, "_cached_role_weights", None)
        if cached is None:
            cached = parse_group_weight(self.config)
            self._cached_role_weights = cached
        return cached

    def _role_weight(self, role):
        """Look up a single role's loss weight.

        Falls back to DEFAULT_ROLE_WEIGHT (1.0) when:
          * `role` is not listed in any group_weight entry, OR
          * the group entry omits the `weight` field, OR
          * config.group_weight itself is missing / empty.
        """
        return self._role_loss_weights.get(role, self.DEFAULT_ROLE_WEIGHT)

    def _seq_weight(self, arguments, candidate=None):
        """Per-sequence loss weight policy.

        AC-CR / per-candidate models (`candidate` given):
            weight = _role_weight(candidate)
        AC-C / single-inference models (`candidate` is None):
            weight = max(_role_weight(arg.role) for arg in arguments)
            sequences with no gold args fall back to DEFAULT_ROLE_WEIGHT.
        """
        if candidate is not None:
            return self._role_weight(candidate)
        if not arguments:
            return self.DEFAULT_ROLE_WEIGHT
        return max(self._role_weight(arg[2]) for arg in arguments)

    def _save_best_and_check_early_stop(self, dev_f, avg_loss, epoch, best_state):
        """Save-best + early-stop driver, with epoch-summary logging.

        `best_state` is a mutable dict the caller maintains across epochs:
            {"ac_f": float, "loss": float, "epoch": int, "epochs_since_improve": int}
        Use `EAEACMixin.make_best_state()` to construct it.

        Save-best policy:
            * first epoch always saves (best_state["epoch"] < 0);
            * any strict F1 improvement saves;
            * F1 ties break on lower train loss.

        Early-stop policy (config.early_stop_epochs > 0; default 0 = disabled):
            * counter resets on strict F1 improvement only;
            * tie-saves do NOT reset (otherwise a tie loop defers stopping
              indefinitely);
            * stop when counter >= early_stop_epochs.

        Logs: per-epoch summary (current F1 + running best) and an
        early-stop message when triggered.

        Returns (should_save, should_stop). Caller still does the actual
        torch.save (so we don't dictate state shape) and decides whether
        to break out of its training loop.
        """
        f1_improved = (best_state["epoch"] < 0) or (dev_f > best_state["ac_f"])
        should_save = (
            f1_improved
            or (dev_f == best_state["ac_f"] and avg_loss < best_state["loss"])
        )
        if should_save:
            best_state["ac_f"]  = dev_f
            best_state["loss"]  = avg_loss
            best_state["epoch"] = epoch
        if f1_improved:
            best_state["epochs_since_improve"] = 0
        else:
            best_state["epochs_since_improve"] += 1

        early_stop_epochs = int(getattr(self.config, "early_stop_epochs", 0) or 0)
        should_stop = (
            early_stop_epochs > 0
            and best_state["epochs_since_improve"] >= early_stop_epochs
        )

        # Per-epoch summary — same format every EAE-AC trainer used to print.
        logger.info(pprint.pformat({"epoch": epoch, "ac_overall_f1": dev_f}))
        logger.info(pprint.pformat({
            "best_epoch":  best_state["epoch"],
            "best_scores": {"ac_f": best_state["ac_f"]},
        }))

        if should_stop:
            logger.info(
                f"[Early stop] no F1 improvement for "
                f"{best_state['epochs_since_improve']} epochs "
                f"(patience={early_stop_epochs}); stopping at epoch {epoch}. "
                f"Best epoch={best_state['epoch']}, "
                f"best F1={best_state['ac_f']:.4f}."
            )
        return should_save, should_stop

    @staticmethod
    def make_best_state():
        """Initial best_state dict for `_save_best_and_check_early_stop`."""
        return {"ac_f": 0.0, "loss": float("inf"),
                "epoch": -1, "epochs_since_improve": 0}

    # ---------------- shared helpers (span match accounting) -----------------

    @staticmethod
    def new_span_match_stats():
        """Initialize a span-match accumulator for a single eval/test pass.

        Trainers call `record_span_match` for every decoded (span, role)
        pair their `_predict_batch` produces, then call
        `log_span_match_stats(split)` at the end of the pass to surface the
        drop summary. Tracks totals AND per-role breakdown so a single role
        with high paraphrase rate can be identified without flooding the log.
        """
        return {
            "n_decoded": 0,
            "n_dropped": 0,
            "decoded_per_role": defaultdict(int),
            "dropped_per_role": defaultdict(int),
        }

    @staticmethod
    def record_span_match(stats, role_type, matched):
        """Update span-match accumulator for one decoded prediction.

        `matched=False` means the trainer's `get_span_idx` returned -1 —
        i.e. the decoded text could not be located verbatim in the passage.
        """
        stats["n_decoded"] += 1
        stats["decoded_per_role"][role_type] += 1
        if not matched:
            stats["n_dropped"] += 1
            stats["dropped_per_role"][role_type] += 1

    @staticmethod
    def log_span_match_stats(stats, split, level=logging.INFO):
        """Log the drop summary at the end of an eval/test pass.

        Quiet when nothing was decoded (n_decoded=0) or nothing dropped
        (n_dropped=0); otherwise prints two lines — overall drop count plus
        per-role breakdown sorted by drop count. The per-role line is the
        actionable one: it identifies which role is paraphrasing rather
        than copying from the passage.

        `level` allows the caller to demote to DEBUG when the same
        `_predict_batch` is invoked many times in a row (e.g. per-doc in
        `predict_ac_eae_hierarchical`), where info-level would flood the log.
        """
        n_decoded = stats["n_decoded"]
        if not n_decoded:
            return
        n_dropped = stats["n_dropped"]
        drop_pct = 100.0 * n_dropped / n_decoded
        logger.log(level,
                   f"[{split}] span-match: dropped {n_dropped}/{n_decoded} "
                   f"decoded spans ({drop_pct:.1f}%) — these were generated "
                   f"text that could not be located verbatim in the passage.")
        if not n_dropped:
            return
        role_lines = ", ".join(
            f"{r}={stats['dropped_per_role'][r]}/{stats['decoded_per_role'][r]}"
            for r in sorted(stats["dropped_per_role"],
                            key=lambda r: -stats["dropped_per_role"][r])
        )
        logger.log(level, f"[{split}] span-match drops per role: {role_lines}")


# ===========================================================================
# AC hierarchical prediction utilities — lifted from utils.py so they live
# next to the mixins that call them. Only supervised AC trainers use these;
# the LLM trainer and standard EAE/ED trainers don't, so keeping these out of
# the project-wide utils.py keeps that module focused on data-loading helpers.
# ===========================================================================

def _enforce_sub_factor_spans(entity_mentions, event_mentions, structure):
    """Drop sub-factor predictions whose span falls outside their parent main-factor span.

    For each role that is a sub-factor (e.g. weather_condition under
    working_condition_factors), we check that its predicted span is contained
    within ANY predicted span of the parent main-factor entity. Predictions
    that violate containment are discarded.

    Multi-fragment safe: a main factor can have several predicted spans in the
    same doc (e.g. wc as both a long narrative AND a short word fragment); a
    sub-factor span passes if it sits inside any of them.
    """
    main_factors_lower = {mf.lower() for mf in structure.get("accident_report", [])}

    # sub_factor_role -> parent_main_factor_role
    sub_to_parent = {}
    for mf in structure.get("accident_report", []):
        for sf in structure.get(mf, []):
            sub_to_parent[sf.lower()] = mf.lower()

    if not sub_to_parent:
        return entity_mentions, event_mentions

    # Collect ALL predicted spans per main-factor role (not just the last —
    # the prior version's dict overwrite silently dropped earlier fragments
    # and made multi-fragment main factors fail containment for everything
    # except their final span).
    parent_spans = {}
    for ent in entity_mentions:
        role = ent.get("entity_type", "").lower()
        if role in main_factors_lower:
            parent_spans.setdefault(role, []).append((ent["start"], ent["end"]))

    def _within_parent(role, start, end):
        parent = sub_to_parent.get(role.lower())
        if parent is None:
            return True  # not a sub-factor, keep unconditionally
        spans = parent_spans.get(parent)
        if not spans:
            return False  # parent not predicted, discard sub-factor
        return any(start >= ps and end <= pe for ps, pe in spans)

    filtered_entities = [
        ent for ent in entity_mentions
        if _within_parent(ent.get("entity_type", ""), ent["start"], ent["end"])
    ]

    filtered_events = []
    for evt in event_mentions:
        new_args = [
            arg for arg in evt.get("arguments", [])
            if _within_parent(arg.get("role", ""), arg["start"], arg["end"])
        ]
        evt_copy = dict(evt)
        evt_copy["arguments"] = new_args
        filtered_events.append(evt_copy)

    return filtered_entities, filtered_events


def predict_ac_hierarchical(trainer, trainer_class, eval_data, config, key_map):
    """Flat single-pass inference for the AC task (E2E models).

    The training data uses a single flat event per document (acc_type trigger +
    all factor/sub-factor roles as direct arguments).  After prediction, sub-factor
    spans that fall outside their parent main-factor span are discarded.

    Args:
        trainer:        instantiated trainer (already has a loaded checkpoint)
        trainer_class:  the trainer's class (unused, kept for API compatibility)
        eval_data:      list of E2E instances as returned by load_AC_supervised_data
        config:         model config namespace (unused, kept for API compatibility)
        key_map:        parsed key_mapping.json dict (needs "structure" for span enforcement)

    Returns:
        list of prediction dicts (same structure as trainer.predict output).
    """
    # acc_type is given from upstream classification — E2E model only needs to
    # detect main_factor trigger spans and predict sub_factor arguments.
    structure = key_map.get("structure", {})
    preds = trainer.predict(eval_data)
    # Trainers return only {doc_id, wnd_id, tokens, entity_mentions, event_mentions};
    # `text` is dropped. Stitch it back so ie_preds_to_flat_ac can recover verbatim
    # source spans via generate_offset_map instead of " ".join(tokens).
    eval_text_by_doc = {
        d.get("doc_id"): d.get("text") for d in eval_data if d.get("text")
    }
    for pred, ev in zip(preds, eval_data):
        ents, evts = _enforce_sub_factor_spans(
            pred.get("entity_mentions", []),
            pred.get("event_mentions",  []),
            structure,
        )
        pred["entity_mentions"] = ents
        pred["event_mentions"]  = evts
        pred["text"] = ev.get("text") or eval_text_by_doc.get(pred.get("doc_id"), "")
    return preds


def predict_ac_eae_hierarchical(trainer, trainer_class, eval_data, config, key_map):
    """EAE inference for the AC task. Supports two modes via `config.ac_predict_mode`:

        "parallel"     (default) — Level-1 (acc_type → main_factors) and Level-2
                                   (main_factor → sub_factors) run independently
                                   in a single batched predictEAE call. Sub_factor
                                   recall does not depend on Level-1 success.
        "hierarchical"           — Cascade. Level-1 runs first; Level-2 only runs
                                   for main_factor TYPES the model actually emitted
                                   at Level-1. Two predictEAE calls. Strict structural
                                   gating: when a main_factor isn't predicted, none
                                   of its sub_factors are extracted either.

    The dataset's events form two levels:
      Level-1 (acc_type level)    : trigger = acc_type (fall, electrocution, …)
                                    arguments = main_factor spans
      Level-2 (main_factor level) : trigger = main_factor (work_ctx, …)
                                    arguments = sub_factor spans

    Returns document-level dicts (entity_mentions + event_mentions).
    """
    structure       = key_map.get("structure", {})
    skip_columns    = {c for c in (getattr(config, "skip_columns", None) or [])
                       if isinstance(c, str)}
    main_factors    = [mf for mf in structure.get("accident_report", []) if mf not in skip_columns]
    main_factor_set = set(main_factors)
    mode            = getattr(config, "ac_predict_mode", "hierarchical")
    if mode not in ("parallel", "hierarchical"):
        raise ValueError(
            f"ac_predict_mode must be 'parallel' or 'hierarchical', got {mode!r}")
    # Log the active mode once per process (re-fires only if mode changes mid-run).
    if getattr(predict_ac_eae_hierarchical, "_logged_mode", None) != mode:
        logger.debug(f"[predict_ac_eae_hierarchical] ac_predict_mode = {mode!r}")
        predict_ac_eae_hierarchical._logged_mode = mode

    # Group eval instances by document
    doc_instances = defaultdict(list)
    doc_order     = []
    for inst in eval_data:
        dk = (inst["doc_id"], inst["wnd_id"])
        if dk not in doc_instances:
            doc_order.append(dk)
        doc_instances[dk].append(inst)

    final_predictions = {}

    def _make_inst(doc_id, wnd_id, tokens, text, trigger):
        return {
            "doc_id":     doc_id,
            "wnd_id":     wnd_id,
            "tokens":     tokens,
            "text":       text,
            "trigger":    trigger,
            "arguments":  [],
            "extra_info": {"entity_mentions": [], "event_mentions": []},
        }

    def _parse_arg(arg):
        if isinstance(arg, (list, tuple)):
            a_s, a_e, role = arg[0], arg[1], arg[2]
            arg_text = arg[3] if len(arg) > 3 else None
        else:
            a_s, a_e, role = arg["start"], arg["end"], arg["role"]
            arg_text = arg.get("text")
        return a_s, a_e, role, arg_text

    def _add_extra_info(insts):
        try:
            trainer_class.add_extra_info_fn(insts, insts, config)
        except Exception:
            pass

    from tqdm import tqdm
    for doc_key in tqdm(doc_order, desc=f"AC inference [{mode}]", unit="doc", ncols=100):
        instances   = doc_instances[doc_key]
        doc_id, wnd_id = doc_key
        orig_tokens = instances[0].get("tokens", [])
        orig_text   = instances[0].get("text", "")

        # Collect every trigger type / text seen in this doc's gold instances.
        # Trigger types not in main_factor_set are acc_types.
        trig_text_by_type = {}
        acc_types_seen    = []

        def _record_trigger(trig):
            if isinstance(trig, (tuple, list)) and len(trig) >= 3:
                t = trig[2]
                trig_text_by_type.setdefault(
                    t, trig[3] if len(trig) >= 4 else "")
                if t not in main_factor_set and t not in acc_types_seen:
                    acc_types_seen.append(t)

        for inst in instances:
            _record_trigger(inst.get("trigger"))
            for event in inst.get("events", []) or []:
                _record_trigger(event.get("trigger"))

        # Build Level-1 instances (one per acc_type seen).
        acc_type_insts = [
            _make_inst(doc_id, f"{wnd_id}__{at}", orig_tokens, orig_text,
                       (0, 1, at, trig_text_by_type.get(at, "")))
            for at in acc_types_seen
        ]

        doc_entity_mentions = []
        doc_event_mentions  = []

        # Main_factors with no sub-factors (e.g. `object_involved`) have no
        # useful Level-2 event: the BIO/CLS heads have nothing to extract or
        # classify there. Drop them so we don't waste forwards on a degenerate
        # event. oi spans still surface via Level-1 (where main_factor spans
        # are emitted as arguments under the acc_type event).
        level2_main_factors = [mf for mf in main_factors if structure.get(mf, [])]

        # =================== MODE: PARALLEL ================================
        # Build all Level-2 instances upfront (one per main_factor type with
        # sub-factors), batch them with Level-1 in a SINGLE predictEAE call.
        if mode == "parallel":
            factor_insts = [
                _make_inst(doc_id, f"{wnd_id}__{mf}", orig_tokens, orig_text,
                           (0, 1, mf, trig_text_by_type.get(mf, "")))
                for mf in level2_main_factors
            ]
            all_insts = acc_type_insts + factor_insts
            _add_extra_info(all_insts)

            preds        = trainer.predictEAE(all_insts) if all_insts else []
            n_acc        = len(acc_type_insts)
            preds_level1 = preds[:n_acc]
            preds_level2 = preds[n_acc:]
            level2_types = list(level2_main_factors)

        # =================== MODE: HIERARCHICAL ============================
        # Run Level-1 first; only query Level-2 for main_factor types Level-1
        # actually emitted AND that have sub-factors worth extracting.
        else:  # mode == "hierarchical"
            _add_extra_info(acc_type_insts)
            preds_level1 = trainer.predictEAE(acc_type_insts) if acc_type_insts else []

            predicted_mf_types = []
            level2_eligible = set(level2_main_factors)
            for pred in preds_level1:
                for arg in pred.get("arguments", []):
                    _, _, role, _ = _parse_arg(arg)
                    if (role in main_factor_set
                            and role in level2_eligible
                            and role not in predicted_mf_types):
                        predicted_mf_types.append(role)

            factor_insts = [
                _make_inst(doc_id, f"{wnd_id}__{mf}", orig_tokens, orig_text,
                           (0, 1, mf, trig_text_by_type.get(mf, "")))
                for mf in predicted_mf_types
            ]
            if factor_insts:
                _add_extra_info(factor_insts)
                preds_level2 = trainer.predictEAE(factor_insts)
            else:
                preds_level2 = []
            level2_types = predicted_mf_types

        # ---- Emit Level-1 → main_factor entity_mentions / event_mentions ----
        for at, pred in zip(acc_types_seen, preds_level1):
            ie_args = []
            for arg in pred.get("arguments", []):
                a_s, a_e, role, arg_text = _parse_arg(arg)
                if role not in main_factor_set:
                    continue
                ie_args.append({"role": role, "start": a_s, "end": a_e})
                ent = {"entity_type": role, "start": a_s, "end": a_e}
                if arg_text is not None:
                    ent["text"] = arg_text
                doc_entity_mentions.append(ent)
            if ie_args:
                doc_event_mentions.append({
                    "event_type": at,
                    "trigger":    {"start": 0, "end": len(orig_tokens)},
                    "arguments":  ie_args,
                })

        # ---- Emit Level-2 → sub_factor entity_mentions / event_mentions ----
        for mf, pred in zip(level2_types, preds_level2):
            sub_factor_set = {s for s in structure.get(mf, []) if s not in skip_columns}
            ie_args = []
            for arg in pred.get("arguments", []):
                a_s, a_e, role, arg_text = _parse_arg(arg)
                if sub_factor_set and role not in sub_factor_set:
                    continue
                if role in skip_columns:
                    continue
                ie_args.append({"role": role, "start": a_s, "end": a_e})
                ent = {"entity_type": role, "start": a_s, "end": a_e}
                if arg_text is not None:
                    ent["text"] = arg_text
                doc_entity_mentions.append(ent)
            if ie_args:
                doc_event_mentions.append({
                    "event_type": mf,
                    "trigger":    {"start": 0, "end": len(orig_tokens)},
                    "arguments":  ie_args,
                })

        # ---- Hierarchical mode: enforce sub-factor span containment ----
        # In "hierarchical" mode the cascade gates Level-2 by Level-1's emitted
        # types; we also enforce that each sub-factor span lives inside ANY of
        # its parent main-factor's predicted spans. "parallel" mode keeps the
        # flat emit (no containment) since Level-2 there runs independently of
        # Level-1 and parent spans may not exist.
        # Done BEFORE cls merge so virtual cls entity_mentions (sentinel
        # offsets -1, -1) aren't subjected to the containment check.
        if mode == "hierarchical":
            doc_entity_mentions, doc_event_mentions = _enforce_sub_factor_spans(
                doc_entity_mentions, doc_event_mentions, structure)

        # ---- Merge cls predictions emitted by the shared cls head ----
        # Each per-instance pred carries `cls_predictions: {factor: label}`.
        # Route each cls factor to the event the cls head was TRAINED on (the
        # factor's parent main-factor in the schema). In C-mode the cls head
        # is supervised at the Level-2 event for the parent main-factor — e.g.
        # `construction_trade` only sees gold targets when the prompt is
        # "<sep> Working Circumstances", because that's the only event whose
        # args carry construction_trade as a cls arg. Reading the prediction
        # from Level-1 (the previous first-non-empty default) means the head
        # is queried on `"<sep> Fall"`-prompt features it was never trained
        # on, collapsing construction_trade F1 to ~majority-class baseline.
        # Falls back to first-non-empty for factors with no parent in the
        # schema (e.g. `accident_type` under E2 tasks — top-level cls).
        parent_of_cls = {}
        for mf in structure.get("accident_report", []):
            for sf in structure.get(mf, []):
                parent_of_cls[sf] = mf

        all_preds = list(preds_level1) + list(preds_level2)
        preds_by_event_type = {}
        for pred in all_preds:
            trig = pred.get("trigger")
            et = trig[2] if isinstance(trig, (tuple, list)) and len(trig) >= 3 else None
            if et is not None:
                preds_by_event_type.setdefault(et, []).append(pred)

        all_cls_factors = set()
        for pred in all_preds:
            all_cls_factors.update((pred.get("cls_predictions") or {}).keys())

        doc_cls = {}
        for f in all_cls_factors:
            if f in skip_columns:
                continue
            # Try the parent-event prediction first (head was trained here).
            parent = parent_of_cls.get(f)
            if parent is not None:
                for pred in preds_by_event_type.get(parent, []):
                    label = (pred.get("cls_predictions") or {}).get(f)
                    if label:
                        doc_cls[f] = label
                        break
            if f in doc_cls:
                continue
            # Fallback: first non-empty across all events.
            for pred in all_preds:
                label = (pred.get("cls_predictions") or {}).get(f)
                if label:
                    doc_cls[f] = label
                    break

        # Sub-cls factors → virtual entity_mentions with sentinel offsets.
        # `ie_preds_to_flat_ac` picks up `text` directly without slicing
        # source tokens, which is correct for class labels that aren't
        # substrings of the document.
        for f, label in doc_cls.items():
            if f == "accident_type":
                continue  # handled separately via event_type override
            doc_entity_mentions.append({
                "entity_type": f,
                "start":       -1,
                "end":         -1,
                "text":        label,
            })

        # For E2 tasks the model predicts accident_type; inject a virtual
        # root event_mention so `ie_preds_to_flat_ac` picks it up as the
        # doc's `accident_type` (it reads the first event_type that matches
        # the configured acc_type vocab). Insert at the front so it wins
        # over any gold-acc-type Level-1 event still in the list.
        if "accident_type" in doc_cls:
            doc_event_mentions.insert(0, {
                "event_type": doc_cls["accident_type"],
                "trigger":    {"start": 0, "end": len(orig_tokens)},
                "arguments":  [],
            })

        final_predictions[doc_key] = {
            "doc_id":          doc_id,
            "wnd_id":          wnd_id,
            "tokens":          orig_tokens,
            "text":            orig_text,
            "entity_mentions": doc_entity_mentions,
            "event_mentions":  doc_event_mentions,
        }

    return [final_predictions[k] for k in doc_order if k in final_predictions]


# ===========================================================================
# E2 cascade — accident_type is PREDICTED by the cls head, not given as input.
# The full Level-0-probe → trigger-patch → Level-1+2 cascade lives on
# `TagPrimeE2ACTrainer.predictE2AC` (see TextEE/models/TagPrime/E2ACtrainer.py).
# Selected via `PREDICT_METHOD[task]="predictE2AC"` for the E2 task family in
# evaluate_supervised. Only the shared label-normalizer is kept here.
# ===========================================================================

def _normalize_acc_type(raw):
    """Raw class label → internal trigger-type form.

    "caught-in/between" → "caught_in_between"
    "struck-by"         → "struck_by"

    Mirrors the normalization in convert_AC_to_EAE so the predicted label
    can be used as a trigger type in patterns / event_type_tags lookups.
    """
    if not raw:
        return None
    return raw.lower().replace("-", "_").replace("/", "_")
