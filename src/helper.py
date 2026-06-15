import enum
import json
import os

import torch
from dotenv import load_dotenv

from config import MODEL_CACHE_PATH, SEED

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
INFERENCE_PRECISION = os.getenv("INFERENCE_PRECISION")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BPO_MODEL = "THUDM/BPO"
MEPO_MODEL= "zixiaozhu/MePO"


MINILM_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
GEMMA_EMBEDDING_MODEL = "google/embeddinggemma-300m"
embedding_models = [MINILM_EMBEDDING_MODEL, GEMMA_EMBEDDING_MODEL]

distance_thresholds = {
    f'{MINILM_EMBEDDING_MODEL}': 0.05,
    # f'{QWEN_EMBEDDING_MODEL}': 0.1,
    f'{GEMMA_EMBEDDING_MODEL}': 0.14
}

DEEPSEEK = "deepseek-chat"
CHATGPT = "gpt-5.4-mini"
evaluator_models = [DEEPSEEK, CHATGPT]

LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
VICUNA_7B = "lmsys/vicuna-7b-v1.3"
GEMMA3 = "google/gemma-3-4b-it"
VICUNA_13B = "lmsys/vicuna-13b-v1.5"
GENERIC_LLM_REWRITER = "Qwen/Qwen3-32B"
base_llm_models = [VICUNA_7B, LLAMA2_7B, GEMMA3, VICUNA_13B]

DOLLY_EVAL = "instruction-following/dolly_eval.json"
VICUNA_EVAL = "instruction-following/vicuna_eval.json"
BPO_EVAL = "instruction-following/bpo_test.json"
SELF_INSTRUCT_EVAL = "instruction-following/self_instruct_eval.json"
DEMO_EVAL = "instruction-following/demo_eval.json"
evaluation_datasets = [VICUNA_EVAL, DOLLY_EVAL, BPO_EVAL, SELF_INSTRUCT_EVAL]

BBH = "bbh"
ARC_C = "arc_challenge"
ARC_E = "arc_easy"
GSM8K = "gsm8k"
PIQA = "piqa"
downstream_tasks = [BBH, ARC_C, ARC_E, GSM8K, PIQA]

mepo_folder_name = "rbpo_mepo"
mismatch_folder_name = "mismatch"
consistency_folder_name = "consistency"
eval_folder_name = "evaluation"
downstream_folder_name = "downstream"

experiment_file_name = "experiment.txt"

BPO = "bpo"
RBPO = "rbpo"
MEPO = "mepo"
RMEPO = "rmepo"
METHOD = [BPO, RBPO, MEPO, RMEPO]

# Only using for downstream benchmark
temp_po_models = {
    f'{RBPO}': 0.7,
    f'{RMEPO}': 0.7
}

M = 10 # prompt optimization iterations
# DISTANCE_THRESHOLD=0.05
IMP_ENC=0.5

OPTIM_PROMPT_INSTRUCTION_PATH = "optimize_prompt_instruction.txt"

class AnswerKey(enum.Enum):
    BBH = "target"
    PIQA = "label"
    ARC = "answer"
    GSM8K = "answer"

def clean_name(path_or_id: str):
    name = path_or_id.split("/")[-1]        
    name = name.split(":")[0]            
    return os.path.splitext(name)[0]     

def create_combined_name(model_path: str, dataset: str, evaluator: str):
    model_name = clean_name(model_path)
    dataset_name = clean_name(dataset)
    evaluator_name = clean_name(evaluator)
    
    model_abbr = model_name.split("-")[0].split("_")[0].lower()
    dataset_abbr = dataset_name.split("-")[0].split("_")[0].lower()
    evaluator_abbr = evaluator_name.split("-")[0].split("_")[0].lower()
    
    return f"{model_abbr}_{dataset_abbr}_{evaluator_abbr}"

def convert_analysis_path_to_figure(path: str, suffix: str = "ori_rbpo") -> str:
    norm_path = os.path.normpath(path)
    parts = norm_path.split(os.sep)

    try:
        analysis_idx = parts.index("analysis")

        model = parts[analysis_idx + 1].split("-")[0]
        eval_name = parts[analysis_idx + 2].split("-")[0]
        judge = parts[analysis_idx + 3].split("-")[0]

    except (ValueError, IndexError):
        raise ValueError("Path không đúng cấu trúc src/analysis/...")

    return os.path.join(
        "src",
        "figure",
        f"{model}_{eval_name}_{judge}_{suffix}"
    )

def dataset_processing(file_path: list):
    import os
    os.makedirs(eval_folder_name, exist_ok=True)
    for base_model in base_llm_models:
        for data_path in evaluation_datasets:
            for evaluator in evaluator_models:
                
                file_name = create_combined_name(base_model, data_path, evaluator)
                
                with open(data_path, "r", encoding="utf-8") as f:
                    data =  json.load(f)
                # Chuyển đổi sang định dạng chuẩn: id, ori_prompt, bpo_prompt, bpo_res, rbpo_prompt, rbpo_res, category, expected_response
                
                results = []
                for idx, item in enumerate(data, start=1):
                    results.append({
                        "id": item.get("id") or item.get("question_id") or item.get("idx") or idx,  # giữ nguyên id nếu đã có, nếu không có thì để None
                        "ori_prompt": item.get("instruction", None) or item.get("prompt", None) or item.get("text", None),
                        "context": item.get("context", None),
                        "bpo_prompt": item.get("optimized_prompt", None),
                        "category": item.get("category", None),
                        "expected_response": item.get("output", None) or item.get("good_res", None) or item.get("response", None)
                    })
                
                with open(f"{eval_folder_name}/{file_name}.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
def print_file_names(file_path: str):
    results = []        
    for base_model in base_llm_models:
        for data_path in evaluation_datasets:
            for evaluator in evaluator_models:
                file_name = create_combined_name(base_model, data_path, evaluator)
                results.append(file_name)
    print("\n".join(results))
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
        
def read_json(file_path):
    """Reads a JSON file and returns the parsed data."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data
    return data

def load_model_and_tokenizer(
    model_path,
    device_map="auto",
    cache_dir=MODEL_CACHE_PATH,
    token=HF_TOKEN,
    inference_precision=INFERENCE_PRECISION
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    inference_precision = inference_precision.lower().strip()
    
    if inference_precision not in ["fp16", "4bit"]:
        raise ValueError("precision must be either 'fp16' or '4bit'")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        token=token,
        legacy=False,
        truncation_side='left',
        padding_side='left'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if inference_precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            cache_dir=cache_dir,
            token=token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    elif inference_precision == "4bit":
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            cache_dir=cache_dir,
            token=token,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        )
    
    model.config.return_dict = True
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    
    print("="*80)
    print(f"Loading base LLM")
    print("="*80)
    print(f"Loaded model: {model_path}")
    print(f"Precision mode: {inference_precision}")

    return model, tokenizer

def set_global_seed(seed = SEED):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # dataset_processing(evaluation_datasets)
    print_file_names(experiment_file_name)
