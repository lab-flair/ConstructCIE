"""LLM E2AC trainer — end-to-end AC (predicts accident_type itself).

Difference vs AC/ACH:
  * AC / ACH — `accident_type` is given as input context (read by
                `accident_collate_fn` off the gold record) and used
                directly to look up the typed few-shot cache slot. The
                model never predicts it.
  * E2AC     — `accident_type` is predicted FIRST from the document text
                (Stage 0), and the predicted value drives the few-shot
                cache lookup for every subsequent extraction stage. The
                prediction also lands in the final flat-AC output so it
                gets scored (E2AC's `skip_columns` deliberately omits
                `accident_type`).

Everything else (main + sub factor extraction, prompts, parsing) is the
same as `LLMACTrainer`. Only the per-batch acc_type sourcing changes.
"""
import logging

import tqdm
from torch.utils.data import DataLoader

from .ACTrainer import LLMACTrainer, accident_collate_fn

logger = logging.getLogger(__name__)


class LLME2ACTrainer(LLMACTrainer):

    def internal_predict(self, eval_data, few_shot_cache={}):
        eval_batch_num = (len(eval_data) // self.config.eval_batch_size
                          + (len(eval_data) % self.config.eval_batch_size != 0))
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100)

        self._reset_parser_counters()
        predictions = []
        prompt_logs_dict = {}

        dataloader = DataLoader(
            eval_data,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=accident_collate_fn,
        )
        main_categories = self.schema["structure"]["accident_report"]

        # Definitive classification keys (from key_map)
        classification_keys = set()
        if ("classification" in self.schema.get("task", {})
                and "classes" in self.schema["task"]["classification"]):
            classification_keys = set(
                self.schema["task"]["classification"]["classes"].keys())

        # Split main factors by extraction vs classification
        main_cls_keys = [k for k in main_categories if k in classification_keys]
        main_ext_keys = [k for k in main_categories if k not in main_cls_keys]

        for batch in dataloader:
            progress.update(1)
            batch_extracted_factors = [{} for _ in range(len(batch.batch_id))]

            # ==========================================
            # STAGE 0 (E2AC-only): Predict accident_type from doc text.
            # ==========================================
            # We don't know the true acc_type yet — look up the "all"
            # cache bucket for prompt examples. The predicted value seeds
            # `predicted_acc_types`, which is threaded into Stage 1/2 in
            # place of `batch.batch_acc_type` so subsequent few-shot
            # lookups use the model's own type prediction.
            stage0_targets = [
                (i, doc_id, "all", text, ["accident_type"], None,
                 "STAGE 0: PREDICT accident_type")
                for i, (doc_id, text) in enumerate(
                    zip(batch.batch_id, batch.batch_acc_txt))
            ]
            stage0_results = self._run_classification_batch(
                stage0_targets, few_shot_cache, prompt_logs_dict)

            predicted_acc_types = ["all"] * len(batch.batch_id)
            for idx, _, parsed in stage0_results:
                val = parsed.get("accident_type")
                if val:
                    s = str(val).strip()
                    if s and s.lower() not in ("none", "null", "n/a", "unknown", ""):
                        predicted_acc_types[idx] = s
                        batch_extracted_factors[idx]["accident_type"] = s
            self._record_stage0_fallback(
                total=len(batch.batch_id),
                fallback=sum(1 for t in predicted_acc_types if t == "all"))

            # ==========================================
            # STAGE 1: Main Factor Prompting (using predicted acc_type)
            # ==========================================
            ext_targets, cls_targets = [], []
            for i, (doc_id, text, acc_type) in enumerate(zip(
                    batch.batch_id, batch.batch_acc_txt, predicted_acc_types)):
                if main_ext_keys:
                    ext_targets.append((i, doc_id, acc_type, text,
                                        main_ext_keys, None,
                                        "STAGE 1: EXTRACT MAIN"))
                if main_cls_keys:
                    cls_targets.append((i, doc_id, acc_type, text,
                                        main_cls_keys, None,
                                        "STAGE 1: CLASSIFY MAIN"))

            ext_results = self._run_extraction_batch(
                ext_targets, few_shot_cache, prompt_logs_dict)
            cls_results = self._run_classification_batch(
                cls_targets, few_shot_cache, prompt_logs_dict)
            for idx, _, parsed in ext_results + cls_results:
                batch_extracted_factors[idx].update(parsed)

            # ==========================================
            # STAGE 2: Subfactor Prompting (still using predicted acc_type)
            # ==========================================
            ext_targets, cls_targets = [], []
            for i, extracted_factors in enumerate(batch_extracted_factors):
                doc_id = batch.batch_id[i]
                acc_type = predicted_acc_types[i]
                for parent_key, parent_text in extracted_factors.items():
                    if (not parent_text
                            or str(parent_text).lower()
                                in ('none', 'null', 'n/a', '', 'unknown')):
                        continue
                    sub_keys = self.schema["structure"].get(parent_key, [])
                    cls_keys = [k for k in sub_keys if k in classification_keys]
                    ext_keys = [k for k in sub_keys if k not in cls_keys]
                    if ext_keys:
                        ext_targets.append((i, doc_id, acc_type, parent_text,
                                            ext_keys, parent_key,
                                            f"STAGE 2: EXTRACT {parent_key}"))
                    if cls_keys:
                        cls_targets.append((i, doc_id, acc_type, parent_text,
                                            cls_keys, parent_key,
                                            f"STAGE 2: CLASSIFY {parent_key}"))

            parsed_subfactors = {
                (i, p_key): {}
                for i in range(len(batch.batch_id))
                for p_key in main_categories
            }
            ext_results = self._run_extraction_batch(
                ext_targets, few_shot_cache, prompt_logs_dict)
            cls_results = self._run_classification_batch(
                cls_targets, few_shot_cache, prompt_logs_dict)
            for idx, p_key, parsed in ext_results + cls_results:
                parsed_subfactors[(idx, p_key)].update(parsed)

            # ==========================================
            # STAGE 3: Assembly into Flat Dictionary
            # ==========================================
            for idx in range(len(batch.batch_id)):
                doc_id = batch.batch_id[idx]
                text = batch.batch_acc_txt[idx]
                acc_type = predicted_acc_types[idx]

                try:
                    fmt_id = float(doc_id)
                except (TypeError, ValueError):
                    fmt_id = doc_id

                prediction = {
                    "accident_report": [text],
                    "id": fmt_id,
                    "accident_type": [acc_type],
                }

                def add_factors(key, val_str):
                    if (not val_str
                            or str(val_str).strip().lower()
                                in ('none', 'null', '', 'unknown')):
                        return
                    fragments = [
                        f.strip()
                        for f in str(val_str).split(';')
                        if f.strip() and f.strip().lower()
                            not in ('none', 'null', 'unknown')
                    ]
                    if fragments:
                        prediction.setdefault(key, [])
                        for f in fragments:
                            if f not in prediction[key]:
                                prediction[key].append(f)

                for parent_key in main_categories:
                    parent_text = batch_extracted_factors[idx].get(parent_key)
                    add_factors(parent_key, parent_text)
                    for sub_key, sub_text in parsed_subfactors.get(
                            (idx, parent_key), {}).items():
                        add_factors(sub_key, sub_text)

                predictions.append(prediction)

        progress.close()
        self._log_parser_summary()
        prompt_logs = list(prompt_logs_dict.values())
        prompt_logs.sort(
            key=lambda x: (str(x.get("doc_id", "")), str(x.get("stage", ""))))
        return predictions, prompt_logs
