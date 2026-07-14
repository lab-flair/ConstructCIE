import re
import torch
import logging
import tqdm
import random
from collections import namedtuple
from torch.utils.data import DataLoader
from .LLMBase import LLMBaseTrainer

# Strips leading list / bullet / numbering / markdown markers from a parsed
# alias string. Examples of inputs that should all reduce to "weather":
#   "1. weather"   "2) weather"   "- weather"   "* weather"   "• weather"
#   "**weather**"  "1. **weather**"
_LEAD_MARKER_RE = re.compile(r"^[\s\-\*•]*(?:\d+[.\)]\s*)?[\s\-\*•]*")

logger = logging.getLogger(__name__)

# --- Separated Prompts with Instruction (System) Support ---

SYSTEM_PROMPT = """You are a strict safety accident report analyzer.
Your response must ONLY be a list of key-value pairs formatted as "Alias: Value".
If information is missing, output "None"."""

EXTRACTION_PROMPT = """Task: Extract the exact, concise text spans from the Input Text that describe the following factors.
{factor_descriptions}

Your output MUST use the exact Alias provided below as the Key.
Format: "Alias: Extracted_Span"
Example: If the factor is "- object: ...", your output MUST be "object: bucket truck".
If there are multiple distinct occurrences for a single factor, separate them with a semicolon (;).
Example: "object: bucket truck; safety harness"
If a factor is not explicitly described, output "None".

Output ONLY a single line per factor in the format "Alias: Extracted_Span".

{examples}Question:
Text: {input_text}"""

CLASSIFICATION_PROMPT = """Task: Classify the following factors based on the input text into exactly one of the provided allowed classes:
{factor_descriptions}

Your output MUST use the exact Alias provided above as the Key. 
Format: "Alias: Selected_Class"
Example: If the factor is "- task: ...", your output must start with "task: " followed by the class.

{examples}Question:
Text: {input_text}"""

AccidentBatch_fields = ['batch_id', 'batch_wnd_id', 'batch_acc_txt', 'batch_acc_type']
AccidentBatch = namedtuple('AccidentBatch', field_names=AccidentBatch_fields, defaults=[None] * len(AccidentBatch_fields))

def accident_collate_fn(batch):
    acc_types = []
    for instance in batch:
        if "accident_type" in instance:
            acc_types.append(instance["accident_type"])
        elif "event_mentions" in instance and len(instance["event_mentions"]) > 0:
            acc_types.append(instance["event_mentions"][0].get("event_type", "accident_report"))
        else:
            acc_types.append("accident_report")

    def get_str(inst, keys):
        for k in keys:
            if k in inst:
                return inst[k]
        return ""

    return AccidentBatch(
        batch_id=[get_str(instance, ["id", "doc_id"]) for instance in batch],
        batch_wnd_id=[get_str(instance, ["wnd_id", "id", "doc_id"]) for instance in batch],
        batch_acc_txt=[get_str(instance, ["accident_report", "text"]) for instance in batch],
        batch_acc_type=acc_types
    )


class LLMACTrainer(LLMBaseTrainer):
    def __init__(self, config, type_set=None):
        super().__init__(config, type_set)
        
        self.schema = getattr(config, 'key_map', None)
        self.descriptions = getattr(config, 'col_defs', None)
        
        assert self.schema is not None, "schema must be provided either as an argument or within the config object."
        assert self.descriptions is not None, "factor_descriptions must be provided either as an argument or within the config object."

    def build_few_shot_cache(self, few_shot_data, few_shot_size):
        if not few_shot_data or few_shot_size <= 0:
            return {}

        main_categories = self.schema["structure"]["accident_report"]
        main_cls_keys = [k for k in main_categories if "classification" in self.schema.get("task", {}) and k in self.schema["task"]["classification"]["classes"]]
        main_ext_keys = [k for k in main_categories if k not in main_cls_keys]

        def get_val(ex, key):
            val = ex.get(key)
            if not val:
                return None
            str_val = str(val).strip()
            return str_val if str_val.lower() not in ['none', 'null', 'n/a', 'unknown', ''] else None

        def _pool(data, keys, require_parent=None):
            base = data if require_parent is None else [ex for ex in data if get_val(ex, require_parent)]
            strict = [ex for ex in base if all(get_val(ex, k) for k in keys)]
            return strict if len(strict) >= few_shot_size else [ex for ex in base if any(get_val(ex, k) for k in keys)]

        def _build(keys, text_key, pool):
            if not pool:
                return ""
            samples = random.sample(pool, min(few_shot_size, len(pool)))
            str_list = []
            for i, ex in enumerate(samples):
                text = get_val(ex, text_key) or get_val(ex, "accident_report") or get_val(ex, "text") or ""
                lines = [f"Example {i+1}:", f"Text: {text}"]
                for k in keys:
                    val = get_val(ex, k)
                    if val:
                        lines.append(f"{self.schema['alias'].get(k, k)}: {val}")
                str_list.append("\n".join(lines))
            return "\n\n".join(str_list) + "\n\n"

        def _build_with_fallback(typed_data, all_data, keys, text_key,
                                 require_parent=None):
            """Per-factor-group selection via layered top-up.

            Fills `few_shot_size` slots by walking layers in priority order
            and adding records not already selected:
              1. typed acc_type, ALL group keys populated  (strict)
              2. typed acc_type, ANY group key populated   (broaden within acctype)
              3. all acctype, ALL group keys populated     (strict cross-acctype)
              4. all acctype, ANY group key populated      (any cross-acctype)

            Earlier layers are exhausted before later ones — the prompt
            prefers richer same-acctype demos and only borrows from across
            acctypes when needed. Within each consumed layer the records
            are picked in random order so we don't always pull the same
            prefix.

            `all_data is None` when we're already building the "all"
            bucket — there's no further fallback layer.
            """
            def _base(data):
                return (data if require_parent is None
                        else [ex for ex in data if get_val(ex, require_parent)])

            layers = []
            typed_base = _base(typed_data)
            layers.append([ex for ex in typed_base if all(get_val(ex, k) for k in keys)])
            layers.append([ex for ex in typed_base if any(get_val(ex, k) for k in keys)])
            if all_data is not None and all_data is not typed_data:
                all_base = _base(all_data)
                layers.append([ex for ex in all_base if all(get_val(ex, k) for k in keys)])
                layers.append([ex for ex in all_base if any(get_val(ex, k) for k in keys)])

            selected = []
            seen = set()
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

            return _build(keys, text_key, selected)

        def _build_sub_cache(typed_data, all_data=None):
            sub = {}
            if main_ext_keys:
                sub["accident_report"] = _build_with_fallback(
                    typed_data, all_data, main_ext_keys, "accident_report")
            if main_cls_keys:
                sub["acc_txt_cls"] = _build_with_fallback(
                    typed_data, all_data, main_cls_keys, "accident_report")
            for parent_key in main_categories:
                sub_keys = self.schema["structure"].get(parent_key, [])
                if not sub_keys:
                    continue
                sub_cls_keys = [k for k in sub_keys if "classification" in self.schema.get("task", {}) and k in self.schema["task"]["classification"]["classes"]]
                sub_ext_keys = [k for k in sub_keys if k not in sub_cls_keys]
                if sub_ext_keys:
                    sub[parent_key] = _build_with_fallback(
                        typed_data, all_data, sub_ext_keys, parent_key,
                        require_parent=parent_key)
                if sub_cls_keys:
                    sub[f"{parent_key}_cls"] = _build_with_fallback(
                        typed_data, all_data, sub_cls_keys, parent_key,
                        require_parent=parent_key)
            return sub

        cache = {}
        acc_types = set(ex.get("accident_type") for ex in few_shot_data if ex.get("accident_type"))
        for acc_type in acc_types:
            typed = [ex for ex in few_shot_data if ex.get("accident_type") == acc_type]
            cache[acc_type] = _build_sub_cache(typed, all_data=few_shot_data)
        cache["all"] = _build_sub_cache(few_shot_data, all_data=None)
        return cache

    @staticmethod
    def _skip_algorithm(few_shot_data, few_shot_size, structure, min_count=None):
        """Stateless implementation — callable without instantiating the trainer.

        AC prompts extract multiple factors at once:
          - One prompt extracts all main factors from acc_txt — pooled count
            is the sum of populated mains across the few-shot examples.
          - For each parent main factor, one prompt extracts all its sub-factors
            from the parent's text — pooled count is the sum of populated subs.

        If a pool's summed count is below `min_count` (or `few_shot_size` if
        `min_count` is None — legacy callers), every member of that pool is
        added to skip columns. Independent decisions per pool.
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
        skip = []

        main_total = sum(
            sum(1 for ex in few_shot_data if get_val(ex, m))
            for m in main_categories
        )
        if main_total < threshold:
            skip.extend(main_categories)

        for parent_key in main_categories:
            sub_keys = (structure or {}).get(parent_key, []) or []
            if not sub_keys:
                continue
            sub_total = sum(
                sum(1 for ex in few_shot_data if get_val(ex, sk))
                for sk in sub_keys
            )
            if sub_total < threshold:
                skip.extend(sub_keys)
        return skip

    def compute_skip_columns(self, few_shot_data, few_shot_size, min_count=None):
        return self._skip_algorithm(
            few_shot_data, few_shot_size, self.schema.get("structure", {}),
            min_count=min_count)

    def save_predictions(self, predictions, path):
        """Write a list of flat-AC prediction dicts to `path` as JSON-lines.

        For ACH-family tasks (ACH/ACH1/JHE/IHE), the original input was
        nested constee format; convert predictions back to the nested shape
        via `convert_predictions("AC", "ACH", ...)` so the on-disk pred file
        mirrors the dataset's native layout.
        """
        import json
        from argparse import Namespace
        task = getattr(self.config, "task", "")
        if task in ("ACH", "ACH1", "JHE", "IHE"):
            try:
                from TextEE.utils import convert_predictions
            except ImportError:
                from utils import convert_predictions
            cfg = Namespace(key_map=getattr(self.config, "key_map", {}))
            predictions = convert_predictions(predictions, "AC", "ACH", cfg)
        with open(path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    def process_data(self, eval_data, few_shot_data=None, few_shot_size=0):
        # Sweeps through data and joins lists into semicolon-separated strings
        def flatten_instance(inst):
            for k, v in inst.items():
                if isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v):
                    valid_frags = [str(x).strip() for x in v if str(x).strip() and str(x).strip().lower() not in ['none', 'null', 'n/a', 'unknown','']]
                    inst[k] = "; ".join(valid_frags) if valid_frags else ""

        for instance in eval_data:
            flatten_instance(instance)
            
        if few_shot_data:
            for instance in few_shot_data:
                flatten_instance(instance)
                
        few_shot_cache = self.build_few_shot_cache(few_shot_data, few_shot_size)
        return eval_data, few_shot_cache

    def build_extraction_prompt(self, text, keys, parent_key=None, examples_str=""):
        desc_lines = []
        for k in keys:
            alias = self.schema["alias"].get(k, k)
            desc = self.descriptions.get(k, "")
            desc_lines.append(f"- {alias}: {desc}")
            
        return EXTRACTION_PROMPT.format(
            factor_descriptions="\n".join(desc_lines),
            examples=examples_str,
            input_text=text
        )

    def build_classification_prompt(self, text, keys, parent_key=None, examples_str=""):
        desc_lines = []
        for k in keys:
            alias = self.schema["alias"].get(k, k)
            desc = self.descriptions.get(k, "")
            classes = ", ".join(self.schema["task"]["classification"]["classes"][k])
            desc_lines.append(f"- {alias}: {desc} (Allowed Classes: {classes})")
            
        return CLASSIFICATION_PROMPT.format(
            factor_descriptions="\n".join(desc_lines),
            examples=examples_str,
            input_text=text
        )
    
    def parse_kv_output(self, generated_text, key_list, prompt_query="[Prompt Query Not Provided]", doc_id="Unknown"):
        text = generated_text.strip()
        text = text.strip("'\"`").strip()
        
        if "Answer:" in text:
            actual_data = text.split("Answer:")[-1].strip()
        else:
            actual_data = text

        alias_to_key = {self.schema["alias"].get(k, k).lower(): k for k in key_list}
        result = {k: None for k in key_list}
        
        format_recognized = False 
        lines = [line.strip() for line in actual_data.split('\n') if line.strip()]

        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                # Strip list / bullet / numbering / markdown wrappers so the
                # alias matches even when the model formats output as
                # "1. Working Circumstances: ..." or "- **Weather**: ...".
                alias_str = parts[0].strip().lower()
                alias_str = _LEAD_MARKER_RE.sub("", alias_str).strip("*_`").strip()
                val_str = parts[1].strip().strip("'\"").strip()

                target_key = None
                if alias_str in alias_to_key:
                    target_key = alias_to_key[alias_str]
                elif len(key_list) == 1:
                    target_key = key_list[0]

                if target_key:
                    format_recognized = True
                    # First-valid wins: when the model regurgitates few-shot
                    # examples ("1. severity: A\n2. severity: B\n…"), the
                    # first line is its best answer for the actual question;
                    # later lines are noise and shouldn't overwrite it.
                    if (result[target_key] is None
                            and val_str.lower() not in ['none', 'null', 'n/a', '', 'unknown']):
                        result[target_key] = val_str
            else:
                val_str = line.strip().strip("'\"").strip()
                if val_str:
                    format_recognized = True
                    target_key = key_list[0]
                    if (result[target_key] is None
                            and val_str.lower() not in ['none', 'null', 'n/a', '']):
                        result[target_key] = val_str

        return result, format_recognized

    # --- PARSER FAILURE TRACKING ---
    # Reset at the start of each `internal_predict` call (one call = one
    # split). Every parser invocation in the batch runners updates these
    # counters; `_log_parser_summary` emits a single INFO line per split so
    # users can see format-error rates without needing --debug.

    def _reset_parser_counters(self):
        self._parser_total    = 0
        self._parser_failures = 0
        # Per-stage breakdown — keyed on the `stage_name` string passed to
        # _run_*_batch (e.g. "STAGE 1: EXTRACT MAIN"). Helps localize where
        # the model's format breaks down.
        self._parser_failures_by_stage = {}
        # E2AC/E2AC1 only: count how many docs ended Stage 0 without a
        # usable predicted accident_type and thus fell back to the "all"
        # cache bucket. Format errors AND parsed-but-unknown/null both
        # land here; the parser counter above only catches the former.
        self._stage0_total    = 0
        self._stage0_fallback = 0

    def _record_parse_result(self, stage_name, format_recog):
        self._parser_total += 1
        if not format_recog:
            self._parser_failures += 1
            self._parser_failures_by_stage[stage_name] = \
                self._parser_failures_by_stage.get(stage_name, 0) + 1

    def _record_stage0_fallback(self, total, fallback):
        self._stage0_total    += total
        self._stage0_fallback += fallback

    def _log_parser_summary(self):
        total  = getattr(self, "_parser_total", 0)
        failed = getattr(self, "_parser_failures", 0)
        if total == 0:
            return
        pct = (failed / total) * 100.0
        logger.warning(f"[parser] {failed}/{total} ({pct:.1f}%) prompts had unrecognized output format")
        for stage, count in sorted(self._parser_failures_by_stage.items(),
                                   key=lambda kv: -kv[1]):
            logger.warning(f"[parser]   {stage}: {count} failure(s)")
        s0_total = getattr(self, "_stage0_total", 0)
        if s0_total > 0:
            s0_fb = getattr(self, "_stage0_fallback", 0)
            s0_pct = (s0_fb / s0_total) * 100.0
            logger.warning(
                f"[stage0] {s0_fb}/{s0_total} ({s0_pct:.1f}%) docs fell back "
                f"to 'all' (parser failure OR value in none/null/n-a/unknown)")

    # --- BATCH RUNNER HELPERS ---

    def _run_extraction_batch(self, targets, few_shot_cache, prompt_logs_dict):
        if not targets:
            return []

        prompts = []
        for _, _, acc_type, text, keys, parent_key, _ in targets:
            cache_key = parent_key if parent_key else "accident_report"
            type_cache = few_shot_cache.get(acc_type, few_shot_cache.get("all", {}))
            examples_str = type_cache.get(cache_key, "")
            prompts.append(self.build_extraction_prompt(text, keys, parent_key, examples_str))

        preds = self.prompt_batch(prompts, system_prompt=SYSTEM_PROMPT, max_new_tokens=self.config.max_output_length)

        results = []
        for (idx, doc_id, _, _, keys, parent_key, stage_name), prompt, pred in zip(targets, prompts, preds):
            parsed_dict, format_recog = self.parse_kv_output(pred, keys, prompt_query=prompt, doc_id=doc_id)
            self._record_parse_result(stage_name, format_recog)
            if getattr(self, 'debug', False):
                self.log_interaction(doc_id, prompt, pred, stage_name, prompt_logs_dict, parsed_dict, format_recog)
            results.append((idx, parent_key, parsed_dict))

        return results

    def _run_classification_batch(self, targets, few_shot_cache, prompt_logs_dict):
        if not targets:
            return []

        prompts = []
        for _, _, acc_type, text, keys, parent_key, _ in targets:
            cache_key = f"{parent_key}_cls" if parent_key else "acc_txt_cls"
            type_cache = few_shot_cache.get(acc_type, few_shot_cache.get("all", {}))
            examples_str = type_cache.get(cache_key, "")
            prompts.append(self.build_classification_prompt(text, keys, parent_key, examples_str))

        preds = self.prompt_batch(prompts, system_prompt=SYSTEM_PROMPT, max_new_tokens=self.config.max_output_length)

        results = []
        for (idx, doc_id, _, _, keys, parent_key, stage_name), prompt, pred in zip(targets, prompts, preds):
            parsed_dict, format_recog = self.parse_kv_output(pred, keys, prompt_query=prompt, doc_id=doc_id)
            self._record_parse_result(stage_name, format_recog)
            if getattr(self, 'debug', False):
                self.log_interaction(doc_id, prompt, pred, stage_name, prompt_logs_dict, parsed_dict, format_recog)
            results.append((idx, parent_key, parsed_dict))

        return results

    # --- REFACTORED INTERNAL PREDICT ---

    def internal_predict(self, eval_data, few_shot_cache={}):
        eval_batch_num = len(eval_data) // self.config.eval_batch_size + (len(eval_data) % self.config.eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100)

        self._reset_parser_counters()
        predictions = []
        prompt_logs_dict = {}

        dataloader = DataLoader(eval_data, batch_size=self.config.eval_batch_size, shuffle=False, collate_fn=accident_collate_fn)
        main_categories = self.schema["structure"]["accident_report"]

        # 1. EXTRACT DEFINITIVE CLASSIFICATION KEYS
        classification_keys = set()
        if "classification" in self.schema.get("task", {}) and "classes" in self.schema["task"]["classification"]:
            classification_keys = set(self.schema["task"]["classification"]["classes"].keys())

        # 2. SEPARATE MAIN FACTORS
        main_cls_keys = [k for k in main_categories if k in classification_keys]
        main_ext_keys = [k for k in main_categories if k not in main_cls_keys]

        for batch in dataloader:
            progress.update(1)

            batch_extracted_factors = [{} for _ in range(len(batch.batch_id))]

            # ==========================================
            # STAGE 1: Main Factor Prompting
            # ==========================================
            ext_targets, cls_targets = [], []
            for i, (doc_id, text, acc_type) in enumerate(zip(batch.batch_id, batch.batch_acc_txt, batch.batch_acc_type)):
                if main_ext_keys:
                    ext_targets.append((i, doc_id, acc_type, text, main_ext_keys, None, "STAGE 1: EXTRACT MAIN"))
                if main_cls_keys:
                    cls_targets.append((i, doc_id, acc_type, text, main_cls_keys, None, "STAGE 1: CLASSIFY MAIN"))

            ext_results = self._run_extraction_batch(ext_targets, few_shot_cache, prompt_logs_dict)
            cls_results = self._run_classification_batch(cls_targets, few_shot_cache, prompt_logs_dict)

            for idx, _, parsed_dict in ext_results + cls_results:
                batch_extracted_factors[idx].update(parsed_dict)

            # ==========================================
            # STAGE 2: Subfactor Prompting
            # ==========================================
            ext_targets, cls_targets = [], []
            for i, extracted_factors in enumerate(batch_extracted_factors):
                doc_id = batch.batch_id[i]
                acc_type = batch.batch_acc_type[i]
                for parent_key, parent_text in extracted_factors.items():
                    if not parent_text or parent_text.lower() in ['none', 'null', 'n/a', '', 'unknown']:
                        continue

                    sub_keys = self.schema["structure"].get(parent_key, [])
                    cls_keys = [k for k in sub_keys if k in classification_keys]
                    ext_keys = [k for k in sub_keys if k not in cls_keys]

                    if ext_keys:
                        ext_targets.append((i, doc_id, acc_type, parent_text, ext_keys, parent_key, f"STAGE 2: EXTRACT {parent_key}"))
                    if cls_keys:
                        cls_targets.append((i, doc_id, acc_type, parent_text, cls_keys, parent_key, f"STAGE 2: CLASSIFY {parent_key}"))

            parsed_subfactors = {(i, p_key): {} for i in range(len(batch.batch_id)) for p_key in main_categories}
            
            ext_results = self._run_extraction_batch(ext_targets, few_shot_cache, prompt_logs_dict)
            cls_results = self._run_classification_batch(cls_targets, few_shot_cache, prompt_logs_dict)
            
            for idx, p_key, parsed_dict in ext_results + cls_results:
                parsed_subfactors[(idx, p_key)].update(parsed_dict)

            # ==========================================
            # STAGE 3: Assembly into Flat Dictionary
            # ==========================================
            for idx in range(len(batch.batch_id)):
                doc_id = batch.batch_id[idx]
                text = batch.batch_acc_txt[idx]
                acc_type = batch.batch_acc_type[idx]

                try:
                    fmt_id = float(doc_id)
                except ValueError:
                    fmt_id = doc_id
                
                prediction = {
                    "accident_report": [text],
                    "id": fmt_id,
                    "accident_type": [acc_type]
                }

                def add_factors(key, val_str):
                    if not val_str or str(val_str).strip().lower() in ['none', 'null', '', 'unknown']:
                        return
                    
                    # --- SPLIT BY SEMICOLON HERE FOR FINAL RESULTS ---
                    fragments = [f.strip() for f in str(val_str).split(';') if f.strip() and f.strip().lower() not in ['none', 'null', 'unknown']]
                    
                    if fragments:
                        if key not in prediction:
                            prediction[key] = []
                        for f in fragments:
                            if f not in prediction[key]:
                                prediction[key].append(f)

                # Process Main Factors
                for parent_key in main_categories:
                    parent_text = batch_extracted_factors[idx].get(parent_key)
                    add_factors(parent_key, parent_text)

                    # Process Subfactors
                    for sub_key, sub_text in parsed_subfactors.get((idx, parent_key), {}).items():
                        add_factors(sub_key, sub_text)
                
                predictions.append(prediction)

        progress.close()
        self._log_parser_summary()

        prompt_logs = list(prompt_logs_dict.values())
        prompt_logs.sort(key=lambda x: (str(x.get("doc_id", "")), str(x.get("stage", ""))))

        return predictions, prompt_logs
    

    def construct_training_example(self, instance):
        input_text = instance['input_text'] 
        parent_key = instance.get('parent_key', None)
        target_values = instance.get('target_values', {})
        keys = list(target_values.keys())
        
        is_classification = False
        if parent_key is not None and keys:
            if "classification" in self.schema.get("task", {}) and keys[0] in self.schema["task"]["classification"]["classes"]:
                is_classification = True

        if is_classification:
            prompt_str = self.build_classification_prompt(input_text, keys, parent_key, examples_str="")
        else:
            prompt_str = self.build_extraction_prompt(input_text, keys, parent_key, examples_str="")
            
        response_lines = []
        for k in keys:
            alias = self.schema["alias"].get(k, k)
            val = target_values.get(k)
            if not val:
                val = "None"
            response_lines.append(f"{alias}: {val}")
            
        response_str = "\n".join(response_lines)
            
        return prompt_str, response_str