"""LLM E2AC1 trainer — end-to-end AC1 (per-factor + predicts accident_type).

Combines:
  * AC1's one-prompt-per-individual-factor extraction strategy
    (`LLMAC1Trainer`), and
  * E2AC's Stage 0 that predicts `accident_type` from the document text
    and uses that prediction (instead of the gold-given value) to drive
    every subsequent few-shot cache lookup.

Pipeline:
  Stage 0  — classify accident_type from doc text (cache bucket: "all")
  Stage 1  — for each main factor, ONE prompt at a time using the
             predicted acc_type
  Stage 2  — for each sub-factor of a populated parent, ONE prompt at a
             time using the predicted acc_type
  Stage 3  — flat-AC dict assembly (predicted accident_type is included
             so the scorer grades it when E2AC1's `skip_columns` omits it)

Inherits from `LLMAC1Trainer` so the per-factor cache builder and
skip-vector algorithm carry over; we only override `internal_predict`.
"""
import logging

import tqdm
from torch.utils.data import DataLoader

from .AC1Trainer import LLMAC1Trainer
from .ACTrainer import accident_collate_fn

logger = logging.getLogger(__name__)


class LLME2AC1Trainer(LLMAC1Trainer):

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

        classification_keys = set()
        if ("classification" in self.schema.get("task", {})
                and "classes" in self.schema["task"]["classification"]):
            classification_keys = set(
                self.schema["task"]["classification"]["classes"].keys())

        for batch in dataloader:
            progress.update(1)
            batch_extracted_factors = [{} for _ in range(len(batch.batch_id))]

            # ==========================================
            # STAGE 0 (E2AC1-only): Predict accident_type from doc text.
            # ==========================================
            # "all" cache bucket since we don't know acc_type yet.
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
            # STAGE 1: one main factor at a time, using predicted acc_type
            # ==========================================
            for k in main_categories:
                k_targets = [
                    (i, doc_id, acc_type, text, [k], None,
                     f"STAGE 1: {'CLASSIFY' if k in classification_keys else 'EXTRACT'} {k}")
                    for i, (doc_id, text, acc_type) in enumerate(zip(
                        batch.batch_id, batch.batch_acc_txt, predicted_acc_types))
                ]
                runner = (self._run_classification_batch if k in classification_keys
                          else self._run_extraction_batch)
                for idx, _, parsed in runner(k_targets, few_shot_cache, prompt_logs_dict):
                    batch_extracted_factors[idx].update(parsed)

            # ==========================================
            # STAGE 2: one sub-factor at a time, using predicted acc_type
            # Only documents where the parent was successfully extracted are included.
            # ==========================================
            parsed_subfactors = {
                (i, pk): {}
                for i in range(len(batch.batch_id))
                for pk in main_categories
            }

            for parent_key in main_categories:
                for sk in self.schema["structure"].get(parent_key, []):
                    sk_targets = []
                    for i, extracted_factors in enumerate(batch_extracted_factors):
                        parent_text = extracted_factors.get(parent_key)
                        if (not parent_text
                                or str(parent_text).lower()
                                    in ('none', 'null', 'n/a', '', 'unknown')):
                            continue
                        sk_targets.append((
                            i, batch.batch_id[i], predicted_acc_types[i],
                            parent_text, [sk], parent_key,
                            f"STAGE 2: {'CLASSIFY' if sk in classification_keys else 'EXTRACT'} {parent_key}.{sk}"
                        ))
                    if not sk_targets:
                        continue
                    runner = (self._run_classification_batch if sk in classification_keys
                              else self._run_extraction_batch)
                    for idx, p_key, parsed in runner(
                            sk_targets, few_shot_cache, prompt_logs_dict):
                        parsed_subfactors[(idx, p_key)].update(parsed)

            # ==========================================
            # STAGE 3: assemble flat prediction dict
            # ==========================================
            for idx in range(len(batch.batch_id)):
                try:
                    fmt_id = float(batch.batch_id[idx])
                except (ValueError, TypeError):
                    fmt_id = batch.batch_id[idx]

                prediction = {
                    "accident_report": [batch.batch_acc_txt[idx]],
                    "id": fmt_id,
                    "accident_type": [predicted_acc_types[idx]],
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
