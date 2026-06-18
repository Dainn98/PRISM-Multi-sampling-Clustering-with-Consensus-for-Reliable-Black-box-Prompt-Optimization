import gc
import json
import os
import shutil
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from config import MODEL_CACHE_PATH, prompt_template_vicuna
from helper import (
    DOLLY_EVAL,
    GEMMA_EMBEDDING_MODEL,
    SELF_INSTRUCT_EVAL,
    VICUNA_7B,
    base_llm_models,
    clean_name,
    create_combined_name,
    device,
    embedding_models,
    eval_folder_name,
    evaluation_datasets,
    evaluator_models,
    load_model_and_tokenizer,
)
from utils import generate_batch


print("===== STEP 3.3: Selected Prompt Response Generation =====")
torch.cuda.empty_cache()
gc.collect()

BASE_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = BASE_DIR / eval_folder_name

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Match the current generic-prompt run. Change/remove this line if you want all
# embedding models configured in helper.py.
# embedding_models = [GEMMA_EMBEDDING_MODEL, MINI]

PROMPT_RESPONSE_KEYS = [
    ("ori_prompt", "ori_response"),
    ("bpo_prompt", "bpo_response"),
    ("rbpo_prompt", "rbpo_response"),
    # ("generic_prompt", "generic_response"),
    # ("rgeneric_prompt", "rgeneric_response"),
]


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_prompts_for_generation(prompts, item, is_vicuna, is_need_context):
    if not is_vicuna:
        return prompts

    context = item.get("context") if is_need_context else None
    if isinstance(context, str) and context.strip():
        prompts = [
            f"Context:\n{context}\n\nQuestion:\n{prompt}"
            for prompt in prompts
        ]

    return [prompt_template_vicuna.format(prompt) for prompt in prompts]


def generate_generic_responses(
    item,
    model,
    tokenizer,
    is_vicuna,
    is_need_context,
):
    available_pairs = []
    for prompt_key, response_key in PROMPT_RESPONSE_KEYS:
        prompt = item.get(prompt_key)
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        existing_response = item.get(response_key)
        if isinstance(existing_response, str) and existing_response.strip():
            continue

        available_pairs.append((prompt_key, response_key, prompt))

    if not available_pairs:
        return 0

    # Generate once when multiple prompt keys share the same prompt text.
    unique_prompts = list(dict.fromkeys(prompt for _, _, prompt in available_pairs))

    prompts_for_generation = format_prompts_for_generation(
        prompts=unique_prompts,
        item=item,
        is_vicuna=is_vicuna,
        is_need_context=is_need_context,
    )
    responses = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts_for_generation,
        context=item.get("context") if is_need_context else None,
        do_sample=False,
        apply_chat_template=not is_vicuna,
        device=device,
    )
    prompt_to_response = dict(zip(unique_prompts, responses))

    for _, response_key, prompt in available_pairs:
        item[response_key] = prompt_to_response.get(prompt, "")

    return len(available_pairs)


def collect_available_files(model_name, base_model):
    model_dir = EVALUATION_DIR / clean_name(model_name)
    available_files = []

    for data_path in evaluation_datasets:
        for evaluator in evaluator_models:
            file_name = create_combined_name(base_model, data_path, evaluator)
            input_path = model_dir / f"{file_name}.json"
            if input_path.exists():
                available_files.append((data_path, evaluator, input_path))

    return available_files


def main():
    for model_name in embedding_models:
        for base_model in base_llm_models:
            available_files = collect_available_files(model_name, base_model)
            if not available_files:
                print(
                    f"Skipping {base_model} with {model_name}: "
                    "no generic evaluation files found"
                )
                continue

            torch.cuda.empty_cache()
            gc.collect()
            is_vicuna = base_model == VICUNA_7B
            model, tokenizer = load_model_and_tokenizer(
                model_path=base_model,
                cache_dir=MODEL_CACHE_PATH,
                token=hf_token,
            )

            for data_path, evaluator, input_path in available_files:
                is_need_context = data_path in [DOLLY_EVAL, SELF_INSTRUCT_EVAL]
                with input_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                print(
                    f"\nBase model: {base_model} | Dataset: {data_path} | "
                    f"Evaluator: {evaluator} | Loaded {len(data)} samples"
                )

                generated_count = 0
                for item in tqdm(
                    data,
                    desc=f"Generating {input_path.name}",
                ):
                    generated_count += generate_generic_responses(
                        item=item,
                        model=model,
                        tokenizer=tokenizer,
                        is_vicuna=is_vicuna,
                        is_need_context=is_need_context,
                    )

                save_json(input_path, data)
                print(f"Saved: {input_path} | generated={generated_count}")

            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        if os.path.exists(MODEL_CACHE_PATH):
            shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)


if __name__ == "__main__":
    main()
