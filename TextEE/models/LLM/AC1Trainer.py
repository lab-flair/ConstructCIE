import random
import logging
import tqdm
from torch.utils.data import DataLoader
from .ACTrainer import LLMACTrainer, SYSTEM_PROMPT, accident_collate_fn

logger = logging.getLogger(__name__)


class LLMAC1Trainer(LLMACTrainer):
    """AC trainer that issues one prompt per factor while preserving the
    hierarchical structure: main factors are extracted from the full accident
    text first; sub-factors are extracted from their parent's text only after
    the parent has been found.
    """

    # ------------------------------------------------------------------
    # Few-shot cache keyed by individual factor
    # ------------------------------------------------------------------

    def build_few_shot_cache(self, few_shot_data, few_shot_size):
        if not few_shot_data or few_shot_size <= 0:
            return {}

        def get_val(ex, key):
            val = ex.get(key)
            if not val:
                return None
            s = str(val).strip()
            return s if s.lower() not in ['none', 'null', 'n/a', 'unknown', ''] else None

        def build_for_key(key, text_source_key, pool):
            alias = self.schema["alias"].get(key, key)
            if not pool:
                return ""
            samples = random.sample(pool, min(few_shot_size, len(pool)))
            parts = []
            for i, ex in enumerate(samples):
                text = (get_val(ex, text_source_key)
                        or get_val(ex, "accident_report")
                        or get_val(ex, "text") or "")
                val = get_val(ex, key)
                parts.append(f"Example {i+1}:\nText: {text}\n{alias}: {val}")
            return "\n\n".join(parts) + "\n\n"

        def _build_for_key_with_fallback(key, text_source_key, typed_data, all_data):
            """Per-individual-factor selection via layered top-up.

            Fills `few_shot_size` slots by walking 2 layers in priority
            order, adding records not already selected:
              1. typed acctype, `key` populated
              2. all acctype, `key` populated

            Sub-factors only ever appear with their parent populated (data
            invariant — a sub value can't exist without its parent value),
            so a parent-populated filter would be redundant. Within each
            layer, records are shuffled before consumption so we don't
            always pick the same prefix.

            `all_data is None` when we're already building the "all"
            bucket; in that case layer 2 is skipped.
            """
            layers = [[ex for ex in typed_data if get_val(ex, key)]]
            if all_data is not None and all_data is not typed_data:
                layers.append([ex for ex in all_data if get_val(ex, key)])

            selected, seen = [], set()
            for layer in layers:
                if len(selected) >= few_shot_size:
                    break
                shuffled = layer[:]
                random.shuffle(shuffled)
                for ex in shuffled:
                    eid = id(ex)
                    if eid in seen:
                        continue
                    seen.add(eid)
                    selected.append(ex)
                    if len(selected) >= few_shot_size:
                        break

            return build_for_key(key, text_source_key, selected)

        main_categories = self.schema["structure"]["accident_report"]

        def _build_sub_cache(typed_data, all_data=None):
            sub = {}
            for k in main_categories:
                sub[k] = _build_for_key_with_fallback(
                    k, "accident_report", typed_data, all_data)
            for parent_key in main_categories:
                for sk in self.schema["structure"].get(parent_key, []):
                    sub[sk] = _build_for_key_with_fallback(
                        sk, parent_key, typed_data, all_data)
            return sub

        cache = {}
        acc_types = set(ex.get("accident_type") for ex in few_shot_data if ex.get("accident_type"))
        for acc_type in acc_types:
            typed = [ex for ex in few_shot_data if ex.get("accident_type") == acc_type]
            cache[acc_type] = _build_sub_cache(typed, all_data=few_shot_data)
        cache["all"] = _build_sub_cache(few_shot_data, all_data=None)
        return cache

    # ------------------------------------------------------------------
    # AC1 skip vector: independent per-factor counts
    # ------------------------------------------------------------------

    @staticmethod
    def _skip_algorithm(few_shot_data, few_shot_size, structure, min_count=None):
        """Stateless AC1 skip algorithm — overrides the AC pool logic.

        AC1 issues one prompt per factor (no joint extraction), so each factor
        is counted independently — a factor is added to skip columns iff its
        own populated-example count is below `min_count` (or `few_shot_size`
        if `min_count` is None — legacy callers).
        """
        if not few_shot_data or few_shot_size <= 0:
            return []
        threshold = min_count if min_count is not None else few_shot_size

        def get_val(ex, k):
            v = ex.get(k)
            if not v:
                return None
            s = str(v).strip()
            return s if s.lower() not in ['none', 'null', 'n/a', 'unknown', ''] else None

        main_categories = (structure or {}).get("accident_report", []) or []
        all_keys = list(main_categories)
        for parent_key in main_categories:
            all_keys.extend((structure or {}).get(parent_key, []) or [])

        skip = []
        for k in all_keys:
            n = sum(1 for ex in few_shot_data if get_val(ex, k))
            if n < threshold:
                skip.append(k)
        return skip

    # ------------------------------------------------------------------
    # Batch runners: use per-factor cache key (keys[0])
    # ------------------------------------------------------------------

    def _run_extraction_batch(self, targets, few_shot_cache, prompt_logs_dict):
        if not targets:
            return []
        prompts = [
            self.build_extraction_prompt(text, keys, parent_key,
                                         few_shot_cache.get(acc_type, few_shot_cache.get("all", {})).get(keys[0], ""))
            for _, _, acc_type, text, keys, parent_key, _ in targets
        ]
        preds = self.prompt_batch(prompts, system_prompt=SYSTEM_PROMPT,
                                  max_new_tokens=self.config.max_output_length)
        results = []
        for (idx, doc_id, _, _, keys, parent_key, stage), prompt, pred in zip(targets, prompts, preds):
            parsed, fmt_ok = self.parse_kv_output(pred, keys, prompt_query=prompt, doc_id=doc_id)
            if getattr(self, 'debug', False):
                self.log_interaction(doc_id, prompt, pred, stage, prompt_logs_dict, parsed, fmt_ok)
            results.append((idx, parent_key, parsed))
        return results

    def _run_classification_batch(self, targets, few_shot_cache, prompt_logs_dict):
        if not targets:
            return []
        prompts = [
            self.build_classification_prompt(text, keys, parent_key,
                                              few_shot_cache.get(acc_type, few_shot_cache.get("all", {})).get(keys[0], ""))
            for _, _, acc_type, text, keys, parent_key, _ in targets
        ]
        preds = self.prompt_batch(prompts, system_prompt=SYSTEM_PROMPT,
                                  max_new_tokens=self.config.max_output_length)
        results = []
        for (idx, doc_id, _, _, keys, parent_key, stage), prompt, pred in zip(targets, prompts, preds):
            parsed, fmt_ok = self.parse_kv_output(pred, keys, prompt_query=prompt, doc_id=doc_id)
            if getattr(self, 'debug', False):
                self.log_interaction(doc_id, prompt, pred, stage, prompt_logs_dict, parsed, fmt_ok)
            results.append((idx, parent_key, parsed))
        return results

    # ------------------------------------------------------------------
    # internal_predict: one target per factor, hierarchy preserved
    # ------------------------------------------------------------------

    def internal_predict(self, eval_data, few_shot_cache={}):
        eval_batch_num = (len(eval_data) // self.config.eval_batch_size
                          + (len(eval_data) % self.config.eval_batch_size != 0))
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100)

        self._reset_parser_counters()
        predictions = []
        prompt_logs_dict = {}

        dataloader = DataLoader(eval_data, batch_size=self.config.eval_batch_size,
                                shuffle=False, collate_fn=accident_collate_fn)
        main_categories = self.schema["structure"]["accident_report"]

        classification_keys = set()
        if "classification" in self.schema.get("task", {}) and "classes" in self.schema["task"]["classification"]:
            classification_keys = set(self.schema["task"]["classification"]["classes"].keys())

        for batch in dataloader:
            progress.update(1)

            batch_extracted_factors = [{} for _ in range(len(batch.batch_id))]

            # ==========================================
            # STAGE 1: one factor at a time → eval_batch_size prompts per call
            # ==========================================
            for k in main_categories:
                k_targets = [
                    (i, doc_id, acc_type, text, [k], None,
                     f"STAGE 1: {'CLASSIFY' if k in classification_keys else 'EXTRACT'} {k}")
                    for i, (doc_id, text, acc_type) in enumerate(
                        zip(batch.batch_id, batch.batch_acc_txt, batch.batch_acc_type))
                ]
                runner = (self._run_classification_batch if k in classification_keys
                          else self._run_extraction_batch)
                for idx, _, parsed in runner(k_targets, few_shot_cache, prompt_logs_dict):
                    batch_extracted_factors[idx].update(parsed)

            # ==========================================
            # STAGE 2: one sub-factor at a time → ≤ eval_batch_size prompts per call
            # Only documents where the parent was successfully extracted are included.
            # ==========================================
            parsed_subfactors = {(i, pk): {} for i in range(len(batch.batch_id)) for pk in main_categories}

            for parent_key in main_categories:
                for sk in self.schema["structure"].get(parent_key, []):
                    sk_targets = []
                    for i, extracted_factors in enumerate(batch_extracted_factors):
                        parent_text = extracted_factors.get(parent_key)
                        if not parent_text or str(parent_text).lower() in ['none', 'null', 'n/a', '', 'unknown']:
                            continue
                        sk_targets.append((
                            i, batch.batch_id[i], batch.batch_acc_type[i], parent_text, [sk], parent_key,
                            f"STAGE 2: {'CLASSIFY' if sk in classification_keys else 'EXTRACT'} {parent_key}.{sk}"
                        ))
                    if not sk_targets:
                        continue
                    runner = (self._run_classification_batch if sk in classification_keys
                              else self._run_extraction_batch)
                    for idx, p_key, parsed in runner(sk_targets, few_shot_cache, prompt_logs_dict):
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
                    "accident_type": [batch.batch_acc_type[idx]],
                }

                def add_factors(key, val_str):
                    if not val_str or str(val_str).strip().lower() in ['none', 'null', '', 'unknown']:
                        return
                    frags = [f.strip() for f in str(val_str).split(';')
                             if f.strip() and f.strip().lower() not in ['none', 'null', 'unknown']]
                    if frags:
                        prediction.setdefault(key, [])
                        for f in frags:
                            if f not in prediction[key]:
                                prediction[key].append(f)

                for parent_key in main_categories:
                    add_factors(parent_key, batch_extracted_factors[idx].get(parent_key))
                    for sub_key, sub_text in parsed_subfactors.get((idx, parent_key), {}).items():
                        add_factors(sub_key, sub_text)

                predictions.append(prediction)

        progress.close()
        self._log_parser_summary()

        prompt_logs = sorted(prompt_logs_dict.values(),
                             key=lambda x: (str(x.get("doc_id", "")), str(x.get("stage", ""))))
        return predictions, prompt_logs
