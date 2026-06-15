import gc

import transformers, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

from config import OPT_PROMPT_MODEL_CACHE_PATH
from helper import device, MEPO_MODEL, OPTIM_PROMPT_INSTRUCTION_PATH

load_dotenv()
mepo_hf = os.environ.get("HF_TOKEN")
print(mepo_hf)



class Helper:
    @staticmethod
    def read_txt(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().strip()

    @staticmethod
    def count_tokens(text, tokenizer):
        return len(tokenizer.encode(text))
            
class MePOModel:
    def __init__(self):
        # load prompt instruction
        self.po_prompt_ins = Helper.read_txt(OPTIM_PROMPT_INSTRUCTION_PATH)
    
        print("Loading PO model...")
        self.model, self.tokenizer = self.load_model_and_tokenizer(MEPO_MODEL)

        try:
            self.model = torch.compile(self.model)
        except:
            print("torch.compile not supported, skipping...")

        self.cache = {}

    def load_model_and_tokenizer(self, 
        model_path,
        device_map="auto",
        cache_dir=OPT_PROMPT_MODEL_CACHE_PATH,
        token=mepo_hf,
        inference_precision= os.getenv("INFERENCE_PRECISION")
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        infer_precision = inference_precision.lower().strip()
        if infer_precision not in ["fp16", "4bit"]:
            raise ValueError("Inference precision must be either 'fp16' or '4bit'")
        
        tokenizer_ = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            token=token,
            legacy=False,
            truncation_side='left',
            padding_side='left'
        )
        # Fix pad token
        if tokenizer_.pad_token is None:
            tokenizer_.pad_token = tokenizer_.eos_token
        
        if infer_precision == "fp16":
            model_ = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map=device_map,
                cache_dir=cache_dir,
                token=token,
                torch_dtype="auto"
            )
        elif infer_precision == "4bit":
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            model_ = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map=device_map,
                cache_dir=cache_dir,
                token=token,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
            )
        else:
            raise ValueError("Inference precision must be either 'fp16' or '4bit'")
        
        model_.config.return_dict = True
        model_.config.pad_token_id = tokenizer_.pad_token_id
        model_.eval()
        
        print("="*80)
        print(f"Loading base LLM")
        print("="*80)
        print(f"Loaded model: {model_path}")
        print(f"Precision mode: {inference_precision}")
        
        return model_, tokenizer_
    
    @torch.no_grad()
    def generate_response(self, user_query):        
        torch.set_grad_enabled(False)
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": user_query}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device, non_blocking=True)
    
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True 
        )
    
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
    
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def generate_batch(self, prompts):
        messages_batch = [
            [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": p}
            ]
            for p in prompts
        ]
    
        # tokenize batch
        inputs = self.tokenizer.apply_chat_template(    
            messages_batch,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True
        ).to(self.model.device)
    
        input_len = inputs["input_ids"].shape[1]
    
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
        # cắt phần input
        output_tokens = outputs[:, input_len:]
    
        # decode batch
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    
    #Language Models are Few-Shot Learners -> temp = 0.7
    def generate_paraphrase_batch(self, prompts):
        messages_batch = [
            [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": p}
            ]
            for p in prompts
        ]
    
        # tokenize batch
        inputs = self.tokenizer.apply_chat_template(
            messages_batch,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True
        ).to(self.model.device)
    
        input_len = inputs["input_ids"].shape[1]
    
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            use_cache=True,
            temperature=0.9, # lan 1: 0.7, lan 2: 0.9 (thử tăng độ đa dạng)
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        # cắt phần input
        output_tokens = outputs[:, input_len:]
    
        # decode batch
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    def inference(self, user_query):
        if user_query in self.cache:
            return self.cache[user_query]

        po_qs_input = self.po_prompt_ins.replace("S_P", user_query)

        result = self.generate_response(po_qs_input)

        self.cache[user_query] = result
        return result
    
    def clean_up(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
