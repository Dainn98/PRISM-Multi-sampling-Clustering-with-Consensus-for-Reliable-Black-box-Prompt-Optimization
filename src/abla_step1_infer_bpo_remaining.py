import gc
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from config import MODEL_CACHE_PATH, prompt_template_vicuna
from helper import (
    DEEPSEEK,
    DOLLY_EVAL,
    LLAMA2_7B,
    SELF_INSTRUCT_EVAL,
    VICUNA_7B,
    create_combined_name,
    device,
    load_model_and_tokenizer,
)
from utils import generate_batch


print("===== ABLATION: Generate remaining BPO/RBPO responses =====")

BASE_DIR = Path(__file__).resolve().parent
ABLATION_DIR = BASE_DIR / "ablation"
BATCH_SIZE = int(os.getenv("BPO_RESPONSE_BATCH_SIZE", "8"))

evaluation_datasets = [DOLLY_EVAL, SELF_INSTRUCT_EVAL]
evaluator_models = [DEEPSEEK]
base_llm_models = [LLAMA2_7B]

PROMPT_RESPONSE_KEYS = [
    ("bpo_prompt", "bpo_response"),
    ("rbpo_prompt", "rbpo_response"),
]


def is_non_empty_text(value):
    return isinstance(value, str) and bool(value.strip())


def save_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")

    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary_path, path)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def collect_available_files(base_model):
    available_files = []
    for data_path in evaluation_datasets:
        for evaluator in evaluator_models:
            file_name = create_combined_name(base_model, data_path, evaluator)
            input_path = ABLATION_DIR / f"{file_name}.json"
            if input_path.exists():
                available_files.append((data_path, evaluator, input_path))
            else:
                print(f"Skipping missing file: {input_path}")
    return available_files


def count_remaining(data):
    remaining = 0
    missing_prompts = {prompt_key: 0 for prompt_key, _ in PROMPT_RESPONSE_KEYS}

    for item in data:
        for prompt_key, response_key in PROMPT_RESPONSE_KEYS:
            if is_non_empty_text(item.get(response_key)):
                continue
            if is_non_empty_text(item.get(prompt_key)):
                remaining += 1
            else:
                missing_prompts[prompt_key] += 1

    return remaining, missing_prompts


def format_prompts(prompts, item, is_vicuna, is_need_context):
    if not is_vicuna:
        return prompts

    context = item.get("context") if is_need_context else None
    if is_non_empty_text(context):
        prompts = [
            f"Context:\n{context}\n\nQuestion:\n{prompt}"
            for prompt in prompts
        ]
    return [prompt_template_vicuna.format(prompt) for prompt in prompts]


def generate_missing_responses(
    item,
    model,
    tokenizer,
    is_vicuna,
    is_need_context,
):
    pending_pairs = []
    for prompt_key, response_key in PROMPT_RESPONSE_KEYS:
        prompt = item.get(prompt_key)
        response = item.get(response_key)
        if is_non_empty_text(prompt) and not is_non_empty_text(response):
            pending_pairs.append((prompt_key, response_key, prompt.strip()))

    if not pending_pairs:
        return 0

    # If BPO and RBPO selected the same prompt, generate it only once.
    unique_prompts = list(
        dict.fromkeys(prompt for _, _, prompt in pending_pairs)
    )
    prompts_for_generation = format_prompts(
        unique_prompts,
        item,
        is_vicuna,
        is_need_context,
    )
    responses = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts_for_generation,
        batch_size=BATCH_SIZE,
        context=item.get("context") if is_need_context else None,
        do_sample=False,
        apply_chat_template=not is_vicuna,
        device=device,
    )
    if len(responses) != len(unique_prompts):
        raise RuntimeError(
            "Response count does not match prompt count: "
            f"{len(responses)} != {len(unique_prompts)}"
        )

    prompt_to_response = dict(zip(unique_prompts, responses))
    for _, response_key, prompt in pending_pairs:
        item[response_key] = prompt_to_response[prompt]

    return len(pending_pairs)


def main():
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")
    hf_token = os.getenv("HF_TOKEN")

    for base_model in base_llm_models:
        available_files = collect_available_files(base_model)
        pending_files = []

        for data_path, evaluator, input_path in available_files:
            data = load_json(input_path)
            remaining, missing_prompts = count_remaining(data)
            print(
                f"{input_path.name}: remaining={remaining}, "
                f"missing_bpo_prompt={missing_prompts['bpo_prompt']}, "
                f"missing_rbpo_prompt={missing_prompts['rbpo_prompt']}"
            )
            if remaining:
                pending_files.append(
                    (data_path, evaluator, input_path, data, remaining)
                )

        if not pending_files:
            print(f"No remaining BPO/RBPO responses for {base_model}")
            continue

        torch.cuda.empty_cache()
        gc.collect()
        print(f"Loading base response model: {base_model}")
        model, tokenizer = load_model_and_tokenizer(
            model_path=base_model,
            cache_dir=MODEL_CACHE_PATH,
            token=hf_token,
        )
        is_vicuna = base_model == VICUNA_7B

        try:
            for data_path, evaluator, input_path, data, remaining in pending_files:
                is_need_context = data_path in [DOLLY_EVAL, SELF_INSTRUCT_EVAL]
                generated = 0
                description = f"BPO remaining {input_path.name}"

                for item in tqdm(data, desc=description):
                    generated_now = generate_missing_responses(
                        item=item,
                        model=model,
                        tokenizer=tokenizer,
                        is_vicuna=is_vicuna,
                        is_need_context=is_need_context,
                    )
                    if generated_now:
                        generated += generated_now
                        # Checkpoint after every completed item so reruns resume.
                        save_json_atomic(input_path, data)

                _, missing_prompts = count_remaining(data)
                print(
                    f"Saved: {input_path} | generated={generated}/{remaining} | "
                    f"missing_bpo_prompt={missing_prompts['bpo_prompt']} | "
                    f"missing_rbpo_prompt={missing_prompts['rbpo_prompt']}"
                )
        finally:
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
