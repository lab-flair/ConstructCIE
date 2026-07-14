import torch
import csv
import os
import logging
import tqdm
import random
from collections import namedtuple, defaultdict
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    # New imports for Llama 3.2 Vision
    MllamaForConditionalGeneration,
    AutoProcessor
)
from torch.utils.data import DataLoader
from ..trainer import BasicTrainer
from .pattern import patterns
from .util import get_span_idx
from .LLMBase import LLMBaseTrainer

logger = logging.getLogger(__name__)

# --- Configuration ---
PROMPT_TEMPLATE = """You are an argument extractor designed to check for the presence of arguments regarding specific
roles for an event in a sentence.
Task Description: Identify all arguments related to the role {role_list} in the sentence.
These arguments should have the semantic role corresponding to the given event trigger by the word
span between [t] and [/t]. Follow the the format of below examples. Your answer should only
contain the answer string and nothing else.
The event of interest is {event_type}. {event_description}
{examples}
Question
Text: {input_text}"""


# --- Helper: Trigger Marker ---
def mark_trigger(text, trigger_start, trigger_end):
    return text[:trigger_start] + " [t] " + text[trigger_start:trigger_end] + " [/t] " + text[trigger_end:]

def get_arg_objects(text, valid_roles):
    """
    Parses LLM output. 
    """
    outputs = []
    lines = text.strip().split('\n')
    
    for line in lines:
        if ':' not in line:
            continue
        parts = line.split(':', 1)
        role = parts[0].strip()
        arg_text = parts[1].strip()
        
        if role in valid_roles and arg_text:
            outputs.append((arg_text, role))
    return outputs

EAEBatch_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_tokens', 'batch_text', 'batch_piece_idxs', 'batch_token_start_idxs', 
                   'batch_trigger', 'batch_arguments', 'batch_input', 'batch_target']
EAEBatch = namedtuple('EAEBatch', field_names=EAEBatch_fields, defaults=[None] * len(EAEBatch_fields))

def EAE_collate_fn(batch):
    return EAEBatch(
        batch_doc_id=[instance.get("doc_id") for instance in batch],
        batch_wnd_id=[instance.get("wnd_id") for instance in batch],
        batch_tokens=[instance.get("tokens") for instance in batch], 
        batch_text=[instance.get("text") for instance in batch], 
        batch_piece_idxs=[instance.get("piece_idxs", []) for instance in batch], 
        batch_token_start_idxs=[instance.get("token_start_idxs", []) for instance in batch], 
        batch_trigger=[instance.get("trigger") for instance in batch], 
        batch_arguments=[instance.get("arguments") for instance in batch], 
    )

def get_token_info(tokens, tokenizer):
    """
    Generates piece_idxs and token_start_idxs for a list of words.
    """
    pieces = [tokenizer.tokenize(t) for t in tokens]
    token_lens = [len(p) for p in pieces]
    pieces = [p for piece in pieces for p in piece]
    piece_idxs = tokenizer.convert_tokens_to_ids(pieces)
    
    # Verify alignment
    assert sum(token_lens) == len(piece_idxs)
    token_start_idxs = [sum(token_lens[:_]) for _ in range(len(token_lens))] + [sum(token_lens)]

    return {
        "piece_idxs": piece_idxs,
        "token_start_idxs": token_start_idxs
    }

class LLMEAETrainer(LLMBaseTrainer):
    def __init__(self, config, type_set=None):
        super().__init__(config, type_set)
        # self.tokenizer = None
        # self.processor = None  # Added for Vision models
        # self.model = None
        # self.model_id = None
        # self.is_vision_model = False # Flag to switch behaviors
        # self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
    # --- Generates string for one event type ---
    def get_few_shot_str(self, event_type, train_data, few_shot_size, valid_roles):
        if not train_data or few_shot_size <= 0:
            return ""

        candidates = [ex for ex in train_data if ex['trigger'][2] == event_type]
        k = min(few_shot_size, len(candidates))
        if k == 0:
            return ""
            
        selected_examples = random.sample(candidates, k)
        formatted_examples = []
        for i, ex in enumerate(selected_examples):
            ex_text = mark_trigger(ex['text'], ex['trigger'][0], ex['trigger'][1])
            args_by_role = defaultdict(list)
            for arg in ex['arguments']:
                role_name = arg[2]
                arg_text = arg[3]
                args_by_role[role_name].append(arg_text)
            
            lines = [f"Example {i+1}", f"Text: {ex_text}"]
            for role in valid_roles:
                val = ", ".join(args_by_role.get(role, []))
                lines.append(f"{role}: {val}")
            
            formatted_examples.append("\n".join(lines))
        
        return "\n".join(formatted_examples)

    def build_prompt(self, text, trigger, event_type, examples_str=""):
        dataset_patterns = patterns.get(self.config.dataset, {})
        event_schema = dataset_patterns.get(event_type, {})
        
        valid_roles = event_schema.get("valid roles", [])
        role_list = ", ".join(valid_roles)
        description = event_schema.get("event description", "")
        
        trig_s, trig_e, _ = trigger[:3]
        marked_text = mark_trigger(text, trig_s, trig_e)

        return PROMPT_TEMPLATE.format(
            role_list=role_list,
            event_type=event_type,
            event_description=description,
            examples=examples_str,
            input_text=marked_text
        )
    
    def build_few_shot_cache(self, eval_data, few_shot_data, few_shot_size):
        if not few_shot_data or few_shot_size <= 0:
            return {}

        logger.info("Building few-shot cache...")
        needed_events = set(inst['trigger'][2] for inst in eval_data)
        
        train_by_type = defaultdict(list)
        for ex in few_shot_data:
            train_by_type[ex['trigger'][2]].append(ex)
            
        cache = {}
        for event_type in needed_events:
            candidates = train_by_type.get(event_type, [])
            k = min(few_shot_size, len(candidates))
            if k == 0:
                cache[event_type] = ""
                continue
            
            selected_examples = random.sample(candidates, k)
            dataset_patterns = patterns.get(self.config.dataset, {})
            event_schema = dataset_patterns.get(event_type, {})
            valid_roles = event_schema.get("valid roles", [])
            
            formatted_examples = []
            for i, ex in enumerate(selected_examples):
                ex_text = mark_trigger(ex['text'], ex['trigger'][0], ex['trigger'][1])
                args_by_role = defaultdict(list)
                for arg in ex['arguments']:
                    role_name = arg[2]
                    arg_text = arg[3]
                    args_by_role[role_name].append(arg_text)
                
                lines = [f"Example {i+1}", f"Text: {ex_text}"]
                for role in valid_roles:
                    val = ", ".join(args_by_role.get(role, []))
                    lines.append(f"{role}: {val}")
                
                formatted_examples.append("\n".join(lines))
            
            cache[event_type] = "\n".join(formatted_examples)
            
        return cache

    def process_data(self, eval_data, few_shot_data=None, few_shot_size = 0):
        for instance in eval_data:
            if "piece_idxs" not in instance:
                # self.tokenizer works for both Vision (via processor) and Standard models
                info = get_token_info(instance["tokens"], self.tokenizer)
                instance["piece_idxs"] = info["piece_idxs"]
                instance["token_start_idxs"] = info["token_start_idxs"]
        
        few_shot_cache = self.build_few_shot_cache(eval_data, few_shot_data, few_shot_size)
        return eval_data, few_shot_cache
    
    def internal_predict(self, eval_data, few_shot_cache={}):
        eval_batch_num = len(eval_data) // self.config.eval_batch_size + (len(eval_data) % self.config.eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100)
        
        predictions = []
        dataloader = DataLoader(eval_data, batch_size=self.config.eval_batch_size, 
                                shuffle=False, collate_fn=EAE_collate_fn)

        for batch in dataloader:
            progress.update(1)
            
            batch_prompts = []
            for i, text in enumerate(batch.batch_text):
                trigger = batch.batch_trigger[i]
                event_type = trigger[2] 
                
                full_prompt_str = self.build_prompt(
                    text, 
                    trigger, 
                    event_type, 
                    examples_str=few_shot_cache.get(event_type,"")
                )
                batch_prompts.append(full_prompt_str)

            pred_texts = self.prompt_batch(batch_prompts, max_new_tokens=self.config.max_output_length)

            for i, pred_text in enumerate(pred_texts):
                doc_id = batch.batch_doc_id[i]
                wnd_id = batch.batch_wnd_id[i]
                tokens = batch.batch_tokens[i]
                text = batch.batch_text[i]
                piece_idxs = batch.batch_piece_idxs[i]
                token_start_idxs = batch.batch_token_start_idxs[i]
                trigger = batch.batch_trigger[i]
                event_type = trigger[2]
                
                dataset_patterns = patterns.get(self.config.dataset, {})
                event_schema = dataset_patterns.get(event_type, {})
                valid_roles = event_schema.get("valid roles", [])

                pred_objects = get_arg_objects(pred_text, valid_roles)
                
                pred_arguments = []
                for span, role_type in pred_objects:
                    sid, eid = get_span_idx(piece_idxs, token_start_idxs, span, self.tokenizer, trigger_span=trigger[:2])
                    if sid == -1:
                        continue
                    pred_arguments.append((sid, eid, role_type, span))
                
                prediction = {
                    "doc_id": doc_id, "wnd_id": wnd_id, "tokens": tokens, "text": text, 
                    "trigger": trigger, "arguments": pred_arguments
                }
                predictions.append(prediction)

        progress.close()
        return predictions
    
    def predict(self, data, **kwargs):
        # We check self.model to ensure load_model was called
        assert self.model is not None, "Model not loaded. Call load_model() first."
        
        internal_data, few_shot_cache = self.process_data(
            data, 
            few_shot_size=int(kwargs.get("few_shot_size", 0)), 
            few_shot_data=kwargs.get("few_shot_data", None)
        )
        predictions = self.internal_predict(internal_data, few_shot_cache)
        return predictions
    
    # --- NEW: Implementation of the Abstract Hook ---
    def construct_training_example(self, instance):
        """
        Formats the EAE task data into Prompt and Target strings.
        """
        trigger = instance['trigger']
        event_type = trigger[2]
        
        # 1. Get Examples (None if Eval/Zero-Shot)
        # examples_str = "" if is_eval else few_shot_cache.get(event_type, "")

        # 2. Build Prompt
        prompt_str = self.build_prompt(
            instance['text'], 
            trigger, 
            event_type, 
            examples_str=""
        )
        
        # 3. Build Target Response
        dataset_patterns = patterns.get(self.config.dataset, {})
        valid_roles = dataset_patterns.get(event_type, {}).get("valid roles", [])
        
        args_by_role = defaultdict(list)
        for arg in instance['arguments']:
            args_by_role[arg[2]].append(arg[3])
        
        target_lines = []
        for role in valid_roles:
            if role in args_by_role:
                val = ", ".join(args_by_role[role])
                target_lines.append(f"{role}: {val}")
        
        response_str = "\n".join(target_lines)
        
        return prompt_str, response_str