"""TagPrime E2 (end-to-end) AC trainer.

Used for E2AC / E2AC1 / E2ACH / E2ACH1 tasks where `accident_type` is NOT
given as input and must be predicted by the cls head from the doc text.

Inherits everything from `TagPrimeACTrainer` (training pipeline, model
architecture, extraction logic). Overrides `predictAC` to run a two-pass
cascade:

  1. Per doc, run one Level-0 probe forward with trigger="accident_report".
     `patterns[dataset]["accident_report"]` lists "accident_type" as its
     sole CR candidate, so under CR the prompt is
         </s> Accident Report </s> Accident Type </s>
     and the cls head's `accident_type` column is supervised by this
     forward. Under C the prompt is `</s> Accident Report </s>` and the
     cls head reads accident_type off the pooled vector.

  2. Patch each eval instance's trigger tuple — substitute the gold
     acc_type at position [2] with the model's prediction. Then delegate
     to `predict_ac_eae_hierarchical`, which now builds Level-1 prompts
     conditioned on the PREDICTED acc_type (no gold leak).

Per-epoch dev evaluation (`_ac_dev_eval`) calls `self.predictAC` and
therefore also runs the cascade — dev F1 reflects exactly how the model
scores at test, with the model's own acc_type prediction conditioning
Level-1 extraction.
"""
import logging
from collections import defaultdict

from ..ACtrainer import predict_ac_eae_hierarchical, _normalize_acc_type
from .ACtrainer import TagPrimeACTrainer

logger = logging.getLogger(__name__)


class TagPrimeE2ACTrainer(TagPrimeACTrainer):

    def predictAC(self, data, **kwargs):
        key_map         = getattr(self.config, "key_map", {})
        main_factor_set = set(key_map.get("structure", {}).get("accident_report", []))

        # Group instances per doc so we run one probe forward per doc.
        by_doc = defaultdict(list)
        doc_order = []
        for inst in data:
            dk = (inst["doc_id"], inst["wnd_id"])
            if dk not in by_doc:
                doc_order.append(dk)
            by_doc[dk].append(inst)

        rewritten = []
        for dk in doc_order:
            insts = by_doc[dk]
            doc_id, wnd_id = dk
            orig_tokens = insts[0].get("tokens", [])
            orig_text   = insts[0].get("text", "")

            # Stage 0: Level-0 probe — trigger="accident_report".
            # patterns[dataset]["accident_report"] = ["accident_type"], so
            # under CR the cls head's accident_type column is supervised by
            # this one forward; under C the same pool feeds all cls factors
            # but accident_type is the relevant one here.
            probe = {
                "doc_id":     doc_id,
                "wnd_id":     f"{wnd_id}__acctype_probe",
                "tokens":     orig_tokens,
                "text":       orig_text,
                "trigger":    (0, len(orig_tokens), "accident_report", orig_text),
                "arguments":  [],
                "extra_info": {"entity_mentions": [], "event_mentions": []},
            }
            try:
                self.__class__.add_extra_info_fn([probe], [probe], self.config)
            except Exception:
                pass
            probe_preds = self.predictEAE([probe])
            probe_cls = (probe_preds[0].get("cls_predictions") or {}) if probe_preds else {}
            predicted_at = _normalize_acc_type(probe_cls.get("accident_type"))

            # Stage 1 prep: patch each instance's trigger so Level-1 prompts
            # use the predicted acc_type. Level-2 (main_factor) triggers and
            # the accident_report trigger itself are left untouched.
            def _patch_trigger(trig):
                if not isinstance(trig, (tuple, list)) or len(trig) < 3:
                    return trig
                if trig[2] in main_factor_set or trig[2] == "accident_report":
                    return trig
                if predicted_at is None:
                    return trig
                new_trig = list(trig)
                new_trig[2] = predicted_at
                return tuple(new_trig)

            for inst in insts:
                new_inst = dict(inst)
                new_inst["trigger"] = _patch_trigger(inst.get("trigger"))
                if inst.get("events"):
                    new_inst["events"] = [
                        {**e, "trigger": _patch_trigger(e.get("trigger"))}
                        for e in inst["events"]
                    ]
                rewritten.append(new_inst)

        # Stage 1+2: delegate to the base hierarchical predictor with the
        # rewritten instances. Level-1 prompts now see the predicted
        # acc_type at trigger[2] (no gold leak); Level-2 follows as usual.
        return predict_ac_eae_hierarchical(
            self, self.__class__, rewritten, self.config, key_map)
