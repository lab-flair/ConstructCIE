import torch
import logging
from vllm import LLM, SamplingParams
from ..trainer import BasicTrainer

logger = logging.getLogger(__name__)

class vLLMBaseTrainer(BasicTrainer):
    def __init__(self, config, type_set=None):
        super().__init__(config, type_set)
        self.model = None
        self.tokenizer = None
        self.model_id = None
        self.is_vision_model = False
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = "bfloat16"
            logger.info(f"BF16 support detected. Loading {self.model_id} in 'bfloat16'.")
        else:
            self.dtype = "float16"
            logger.info(f"BF16 not supported. Loading {self.model_id} in 'float16'.")

        
        
    def load_model(self, checkpoint:str):
        logger.info(f"Loading vLLM model from {checkpoint}")
        self.model_id = checkpoint.split("/")[-1].lower()
        # --- 2. Initialize vLLM Engine ---
        # tensor_parallel_size=torch.cuda.device_count() automatically uses all visible GPUs.
        # This works for both Zephyr (usually 1 GPU) and Llama 3.2 (scales to multiple if needed).
        try:
            self.model = LLM(
                model=checkpoint,
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True,
                dtype=self.dtype,
                max_num_seqs=self.config.eval_batch_size,
                # enforce_eager=True # Uncomment if Llama 3.2 Vision gives CUDA graph errors
            )
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            raise e

        self.tokenizer = self.model.get_tokenizer()

        # --- 3. Detect Vision Variant ---
        # Llama 3.2 Vision models have stricter template requirements than standard text models
        if "llama-3.2" in self.model_id and "vision" in self.model_id:
            self.is_vision_model = True
            logger.info("Detected Llama 3.2 Vision model. Using structured content formatting.")
        else:
            self.is_vision_model = False
            logger.info("Detected Standard Text model (Zephyr/Llama-Text). Using standard formatting.")

    def format_prompt(self, prompt):
        """
        Handles the differences in Chat Templates between Zephyr and Llama 3.2 Vision.
        """
        # A. Logic for Llama 3.2 Vision (Requires list of dicts)
        if self.is_vision_model:
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]
            # The tokenizer handles the specific <|start_header_id|> tags for Llama 3
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # B. Logic for Zephyr / Llama 3.2 Text (Expects string)
        messages = [{"role": "user", "content": prompt}]
        
        # Use the model's native template (Zephyr uses <|user|>, Llama 3 uses <|begin_of_text|>)
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # C. Fallback for older models without chat templates
        return f"<s>[INST] {prompt} [/INST]"

    def prompt_batch(self, prompts, max_new_tokens=256):
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")

        # 1. Format prompts based on the model type
        formatted_prompts = [self.format_prompt(p) for p in prompts]
        
        # 2. Set vLLM Sampling Parameters
        # temperature=0.0 is equivalent to do_sample=False (Greedy Decoding)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            ignore_eos=False
        )
        
        # 3. Generate (vLLM handles batching and PagedAttention internally)
        outputs = self.model.generate(formatted_prompts, sampling_params)
        
        # 4. Extract Text
        responses = []
        for output in outputs:
            # vLLM automatically removes the input prompt from the output
            generated_text = output.outputs[0].text
            responses.append(generated_text.strip())
            
        return responses

    # --- Abstract Methods ---
    def get_few_shot_str(self):
        raise NotImplementedError

    def build_prompt(self):
        raise NotImplementedError
    
    def build_few_shot_cache(self, eval_data, few_shot_data, few_shot_size):
        raise NotImplementedError

    def process_data(self, eval_data, few_shot_data=None, few_shot_size=0):
        raise NotImplementedError
    
    def internal_predict(self, eval_data, few_shot_cache={}):
        raise NotImplementedError

    def predict(self, data, **kwargs):
        internal_data, few_shot_cache = self.process_data(
            data, 
            few_shot_size=int(kwargs.get("few_shot_size", 0)), 
            few_shot_data=kwargs.get("few_shot_data", None)
        )
        predictions = self.internal_predict(internal_data, few_shot_cache)
        return predictions