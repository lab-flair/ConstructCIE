import torch
import csv
import os
import logging
import tqdm
import random
import re
from collections import namedtuple, defaultdict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    MllamaForConditionalGeneration,  # Llama 3.2 Vision
    AutoProcessor,
    Trainer,
    TrainingArguments
)
from torch.utils.data import DataLoader, Dataset
from ..trainer import BasicTrainer
from .pattern import patterns
from .util import get_span_idx
import gc

logger = logging.getLogger(__name__)

# Simple wrapper for raw list data
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class LLMBaseTrainer(BasicTrainer):
    def __init__(self, config, type_set=None):
        super().__init__(config, type_set)
        self.tokenizer = None
        self.processor = None  # Added for Vision models
        self.model = None
        self.model_id = None
        self.is_vision_model = False # Flag to switch behaviors
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        # --- NEW: Global Debug and Logging State ---
        self.debug = getattr(config, 'debug', False)
        self._seen_logs = set()
        self._seen_errors = set()
    
    def log_interaction(self, doc_id, prompt, response, stage, log_dict, parsed_dict=None, format_recognized=None):
        """
        Stores the prompt, raw response, and optional parsed result in the dictionary to avoid 
        dataloader padding duplicates. If debug mode is enabled, it also prints 
        deduplicated logs and parser warnings to the console.
        """
        # 1. Always store in the dictionary to build the JSON output
        log_entry = {
            "doc_id": doc_id,
            "stage": stage,
            "prompt": prompt,
            "response": response
        }
        
        # Only add parser fields to the JSON if they were provided
        if parsed_dict is not None:
            log_entry["parsed_result"] = parsed_dict
        if format_recognized is not None:
            log_entry["format_recognized"] = format_recognized
            
        log_dict[(doc_id, stage)] = log_entry
        
        # 2. Terminal logging logic (guarded by debug flag and cache)
        # if not getattr(self, 'debug', False):
        #     return
            
        # Deduplication hash
        log_hash = hash((prompt, response, stage))
        
        if not hasattr(self, '_seen_logs'):
            self._seen_logs = set()
            
        if log_hash in self._seen_logs:
            return # Skip printing if we've already logged this exact interaction
            
        self._seen_logs.add(log_hash)

        # Cap the cache size to prevent memory bloat during long runs
        if len(self._seen_logs) > 1000:
            self._seen_logs.clear()

        # Build the console message
        msg_lines = [
            f"\n{'='*80}",
            f"🆔 DOCUMENT ID: {doc_id}",
            f"🏷️  STAGE: {stage}",
            f"{'-'*80}",
            f"📝 PROMPT:\n{prompt}",
            f"{'-'*80}",
            f"🤖 RAW RESPONSE:\n{repr(response)}",
            f"{'-'*80}"
        ]

        log_func = logger.info

        # Append parser success/failure to the message ONLY if parser args were provided
        if format_recognized is not None:
            if not format_recognized:
                msg_lines.insert(1, "🚨 [DEBUG] PARSER FAILURE")
                msg_lines.append("❌ PARSER RESULT: Failed to recognize format.")
                log_func = logger.warning
            else:
                clean_dict = {k: v for k, v in parsed_dict.items() if v is not None} if parsed_dict else {}
                msg_lines.append(f"✅ PARSER MAPPED: {clean_dict}")

        msg_lines.append(f"{'='*80}\n")
        
        # Output to console
        log_func("\n".join(msg_lines))
        
    def load_model(self, checkpoint:str):
        # === PREVENT LEAKS ON RELOAD ===
        if self.model is not None:
            logger.info("Unloading previous model to free VRAM...")
            del self.model
            del self.tokenizer
            del self.processor
            gc.collect()
            torch.cuda.empty_cache()
            self.model = None
        # ===============================
        
        logger.info(f"Loading model from {checkpoint}")
        self.model_id = checkpoint.split("/")[-1].lower()

        # --- Detect Llama 3.2 Vision Models ---
        if "llama-3.2" in self.model_id and "vision" in self.model_id:
            logger.info("Detected Llama 3.2 Vision model. Using MllamaForConditionalGeneration.")
            self.is_vision_model = True
            self.model = MllamaForConditionalGeneration.from_pretrained(
                checkpoint,
                torch_dtype=self.dtype,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(checkpoint)
            self.tokenizer = self.processor.tokenizer
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        else:
            # --- Standard Loading for Text-Only Models (Qwen 3, 0  Zephyr, Mistral, Llama 2) ---
            logger.info("Detected standard text-only model architecture.")
            self.is_vision_model = False
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.tokenizer.padding_side = "left" 
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype=self.dtype,
                device_map="auto",
            )

    def format_prompt(self, user_prompt, system_prompt=None):
        """
        Formats the prompt string into the model-specific template.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # 1. Logic for Vision / Processor-based Models (Llama 3.2 Vision, Gemma 4)
        if self.is_vision_model:
            messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})
            return self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # 2. Logic for Standard Text Models
        messages.append({"role": "user", "content": user_prompt})

        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            if "qwen" in self.model_id:
                kwargs["enable_thinking"] = False  # Qwen3 native thinking mode switch
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except Exception as e:
                logger.warning(f"apply_chat_template failed: {e}. Falling back to manual formatting.")

        # Manual Fallbacks
        if "qwen" in self.model_id:
            sys_block = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
            return f"{sys_block}<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        elif "gemma" in self.model_id:
            sys_block = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n" if system_prompt else ""
            return f"{sys_block}<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"

        elif "llama" in self.model_id or "mixtral" in self.model_id or "zephyr" in self.model_id:
            sys_block = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n" if system_prompt else ""
            return f"<s>[INST] {sys_block}{user_prompt} [/INST]"

        return user_prompt

    def prompt_batch(self, prompts, system_prompt=None, max_new_tokens=512):
        formatted_prompts = [self.format_prompt(p, system_prompt) for p in prompts]

        # Vision path uses the processor (so the chat template is the Mllama
        # one, not the tokenizer's text-only one), but inputs still include
        # `input_ids` / `attention_mask` when called with images=None. That
        # lets us slice generated tokens by prompt length — same as the
        # text-only path below — instead of guessing where the prompt ended
        # via string splits on "assistant" / "Output".
        if self.is_vision_model:
            inputs = self.processor(
                text=formatted_prompts,
                images=None,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
            decoder = self.processor

        else:
            inputs = self.tokenizer(formatted_prompts, return_tensors="pt",
                                    padding=True, truncation=True).to(self.model.device)

            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=gen_config)
            decoder = self.tokenizer

        responses = []
        for i, output in enumerate(outputs):
            prompt_length = inputs['input_ids'][i].shape[0]
            generated_tokens = output[prompt_length:]
            generated_text = decoder.decode(generated_tokens, skip_special_tokens=True)

            # Remove thinking blocks
            generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
            # Strip end-of-sequence markers (removing them preserves content before them)
            for eos_marker in ["<end_of_turn>", "</s>", "<|eot_id|>"]:
                generated_text = generated_text.replace(eos_marker, "").strip()
            # Strip prefix markers that the model may echo (take content after)
            for prefix_marker in ["[/INST]", "Output:", "Answer:"]:
                if prefix_marker in generated_text:
                    generated_text = generated_text.split(prefix_marker)[-1].strip()
            # "assistant" can appear in content — only strip if echoed at the very start
            if generated_text.startswith("assistant"):
                generated_text = generated_text[len("assistant"):].strip()

            responses.append(generated_text)

        return responses
    
    # --- Generates string for one event type ---
    def get_few_shot_str(self):
        raise NotImplementedError

    def build_prompt(self):
        raise NotImplementedError
    
    def build_few_shot_cache(self, eval_data, few_shot_data, few_shot_size):
        raise NotImplementedError

    def process_data(self, eval_data, few_shot_data=None, few_shot_size = 0):
        raise NotImplementedError
    
    def internal_predict(self, eval_data, few_shot_cache={}):
        raise NotImplementedError

        
    def predict(self, data, **kwargs):
        # We check self.model to ensure load_model was called
        assert self.model is not None, "Model not loaded. Call load_model() first."
        
        internal_data, few_shot_cache = self.process_data(
            data, 
            few_shot_size=int(kwargs.get("few_shot_size", 0)), 
            few_shot_data=kwargs.get("few_shot_data", None)
        )
        # Will naturally return (predictions, prompt_logs) if internal_predict provides it
        return self.internal_predict(internal_data, few_shot_cache)
    
    # --- Abstract Method for Subclasses ---
    def construct_training_example(self, instance):
        """
        Must be implemented by the subclass (e.g., LLMEAETrainer).
        
        Args:
            instance: A single data dictionary.
            
        Returns:
            tuple: (prompt_string, target_response_string)
        """
        raise NotImplementedError("Subclasses must implement construct_training_example")

    # --- Generalized Train Method ---
    def train(self, train_data, eval_data, output_dir, num_epochs=3, batch_size=4, learning_rate=2e-5, gradient_accumulation_steps=1, few_shot_size=0, few_shot_data=None, system_prompt=None):
        """
        Fine-tunes the model using Hugging Face Trainer.
        """
        assert self.model is not None, "Model not loaded. Call load_model() first."
        
        # 1. Process Data & Build Cache
        all_data = train_data + (eval_data if eval_data else [])
        _, few_shot_cache = self.process_data(all_data, few_shot_data, few_shot_size)
        
        logger.info(f"Starting HF Training. Train size: {len(train_data)}, Eval size: {len(eval_data) if eval_data else 0}")

        # 2. Create Datasets
        train_dataset = SimpleDataset(train_data)
        eval_dataset = SimpleDataset(eval_data) if eval_data else None

        # 3. Define Data Collator
        def data_collator(features):
            input_ids_list = []
            labels_list = []
            attention_masks_list = []

            for instance in features:
                # --- CALL SUBCLASS HOOK ---
                prompt_str, response_str = self.construct_training_example(instance)
                
                # --- GENERIC TOKENIZATION & MASKING WITH SYSTEM PROMPT ---
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                    messages_prompt = []
                    if system_prompt: messages_prompt.append({"role": "system", "content": system_prompt})
                    messages_prompt.append({"role": "user", "content": prompt_str})
                    
                    messages_full = messages_prompt.copy()
                    messages_full.append({"role": "assistant", "content": response_str})
                    
                    text_prompt = self.tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
                    text_full = self.tokenizer.apply_chat_template(messages_full, tokenize=False)
                else:
                    sys_str = f"System: {system_prompt}\n" if system_prompt else ""
                    text_prompt = sys_str + f"User: {prompt_str}\nAnswer:\n"
                    text_full = text_prompt + response_str + self.tokenizer.eos_token

                tokenized_full = self.tokenizer(text_full, truncation=True, max_length=self.config.max_output_length + 1024)
                tokenized_prompt = self.tokenizer(text_prompt, truncation=True, add_special_tokens=False)

                input_ids = torch.tensor(tokenized_full["input_ids"])
                labels = input_ids.clone()
                
                prompt_len = len(tokenized_prompt["input_ids"])
                safe_prompt_len = min(prompt_len, len(labels))
                labels[:safe_prompt_len] = -100
                
                input_ids_list.append(input_ids)
                labels_list.append(labels)
                attention_masks_list.append(torch.tensor(tokenized_full["attention_mask"]))

            input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels_batch = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
            attention_masks_batch = torch.nn.utils.rnn.pad_sequence(attention_masks_list, batch_first=True, padding_value=0)

            return {
                "input_ids": input_ids_batch,
                "attention_mask": attention_masks_batch,
                "labels": labels_batch
            }

        # 4. Training Arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=10,
            evaluation_strategy="epoch" if eval_data else "no",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True if eval_data else False,
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
        )

        # 5. Initialize and Run Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

        # 6. Save
        logger.info(f"Saving final model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)