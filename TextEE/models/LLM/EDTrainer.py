import torch
import logging
import tqdm
import random
from collections import defaultdict, namedtuple
from torch.utils.data import DataLoader
from .LLMBase import LLMBaseTrainer
from .pattern import patterns
from .util import get_span_idx

logger = logging.getLogger(__name__)

# --- Configuration: Prompt Template from Image ---
ED_PROMPT_TEMPLATE = """You are an event extractor designed to check for the presence of a specific event in a sentence and to locate the corresponding event trigger.
Task Description: Identify all triggers related to the event of interest in the sentence. A trigger is the key word in the sentence that most explicitly conveys the occurrence of the event. If yes, please answer 'Yes, the event trigger is [trigger] in the text.'; otherwise, answer 'No.'
The event of interest is {event_type}. {event_description}
{examples}
Question
Text: {input_text}"""

# --- Helpers ---
def get_ed_objects(text):
    """
    Parses LLM output for ED.
    Expected: "Yes, the event trigger is [trigger] in the text." or "No."
    """
    text = text.strip()
    if text.lower().startswith("no"):
        return None
    
    # Parse: "Yes, the event trigger is <trigger> in the text."
    # We look for the substring between "trigger is " and " in the text"
    lower_text = text.lower()
    start_marker = "trigger is "
    end_marker = " in the text"
    
    if start_marker in lower_text:
        start_idx = lower_text.find(start_marker) + len(start_marker)
        # We try to find the end marker after the start
        end_idx = lower_text.find(end_marker, start_idx)
        
        if end_idx != -1:
            # Extract from the original text to preserve case
            trigger_text = text[start_idx:end_idx].strip()
            # Cleanup punctuation if the model included quotes
            return trigger_text.strip("'").strip('"')
            
    return None

class LLMEDTrainer(LLMBaseTrainer):
    def __init__(self, config, type_set=None):
        super().__init__(config, type_set)

    # --- 1. Prompt Construction ---
    def build_prompt(self, text, event_type, examples_str=""):
        dataset_patterns = patterns.get(self.config.dataset, {})
        event_schema = dataset_patterns.get(event_type, {})
        description = event_schema.get("event description", "")
        
        return ED_PROMPT_TEMPLATE.format(
            event_type=event_type,
            event_description=description,
            examples=examples_str,
            input_text=text
        )

    # --- 2. Few-Shot Logic ---
    def get_few_shot_str(self, event_type, train_data, few_shot_size):
        if not train_data or few_shot_size <= 0:
            return ""
        
        # We collect positive examples for this specific event type
        candidates = [ex for ex in train_data if ex['trigger'][2] == event_type]
        k = min(few_shot_size, len(candidates))
        if k == 0:
            return ""
            
        selected_examples = random.sample(candidates, k)
        formatted_examples = []
        
        for i, ex in enumerate(selected_examples):
            # Format: Example i \n Text: ... \n Answer: Yes...
            trigger_text = ex['text'][ex['trigger'][0]:ex['trigger'][1]]
            lines = [
                f"Examples {i+1}",
                f"Text: {ex['text']}",
                f"Answer: Yes, the event trigger is {trigger_text} in the text."
            ]
            formatted_examples.append("\n".join(lines))
            
        return "\n".join(formatted_examples)

    def build_few_shot_cache(self, eval_data, few_shot_data, few_shot_size):
        if not few_shot_data or few_shot_size <= 0:
            return {}
            
        # For ED, we cache few-shot strings per Event Type
        # (Unlike EAE, we don't know the event type of the input, 
        # but we iterate over types during prediction)
        
        dataset_patterns = patterns.get(self.config.dataset, {})
        # Get all possible event types from the dataset definition
        all_event_types = list(dataset_patterns.keys())
        
        cache = {}
        for event_type in all_event_types:
            cache[event_type] = self.get_few_shot_str(event_type, few_shot_data, few_shot_size)
            
        return cache

    def process_data(self, data, few_shot_data=None, few_shot_size=0):
        # Build cache based on ALL types in the dataset patterns (since we might query any of them)
        few_shot_cache = self.build_few_shot_cache(data, few_shot_data, few_shot_size)
        return data, few_shot_cache

    # --- 3. Training Hook (Implementation of Abstract Method) ---
    def construct_training_example(self, instance):
        """
        Constructs a training instance for Event Detection.
        Assumption: 'instance' in train_data is a POSITIVE example 
        containing a valid 'trigger' field: (start, end, event_type).
        """
        trigger = instance['trigger']
        event_type = trigger[2]
        text = instance['text']
        
        # 1. Get Examples (Zero-shot if eval)
        # examples_str = "" if is_eval else few_shot_cache.get(event_type, "")
        examples_str = ""
        
        # 2. Build Prompt
        prompt_str = self.build_prompt(text, event_type, examples_str)
        
        # 3. Build Target (Positive Answer)
        trigger_text = text[trigger[0]:trigger[1]]
        response_str = f"Yes, the event trigger is {trigger_text} in the text."
        
        return prompt_str, response_str

    # --- 4. Prediction Logic ---
    def internal_predict(self, eval_data, few_shot_cache={}):
        """
        For ED, we must check every text against every possible event type 
        (or a subset if specified).
        """
        predictions = []
        
        # Get list of all event types to check
        dataset_patterns = patterns.get(self.config.dataset, {})
        all_event_types = list(dataset_patterns.keys())
        
        # We process one document at a time, checking all event types
        # (Optimization: You could batch (doc, type) pairs, but simple loops are safer for now)
        progress = tqdm.tqdm(eval_data, desc="Predicting Events", ncols=100)
        
        for instance in progress:
            text = instance['text']
            doc_id = instance.get('doc_id')
            wnd_id = instance.get('wnd_id')
            
            found_triggers = []
            
            # Prepare batch of prompts for this SINGLE sentence (one prompt per event type)
            prompts = []
            types_in_batch = []
            
            for event_type in all_event_types:
                prompt = self.build_prompt(
                    text, 
                    event_type, 
                    examples_str=few_shot_cache.get(event_type, "")
                )
                prompts.append(prompt)
                types_in_batch.append(event_type)
            
            # Run batch inference for this sentence
            # (If all_event_types is large, you might want to chunk this)
            responses = self.prompt_batch(prompts, max_new_tokens=50)
            
            for i, response in enumerate(responses):
                event_type = types_in_batch[i]
                trigger_text = get_ed_objects(response)
                
                if trigger_text:
                    # Locate the trigger in the text
                    # (Simple string find; for robustness, use token alignment if available)
                    start = text.find(trigger_text)
                    if start != -1:
                        end = start + len(trigger_text)
                        # Store prediction
                        found_triggers.append((start, end, event_type, trigger_text))
            
            predictions.append({
                "doc_id": doc_id,
                "wnd_id": wnd_id,
                "text": text,
                "triggers": found_triggers
            })
            
        return predictions

    def predict(self, data, **kwargs):
        assert self.model is not None, "Model not loaded. Call load_model() first."
        # Standard processing
        internal_data, few_shot_cache = self.process_data(
            data, 
            kwargs.get("few_shot_data"), 
            int(kwargs.get("few_shot_size", 0))
        )
        return self.internal_predict(internal_data, few_shot_cache)