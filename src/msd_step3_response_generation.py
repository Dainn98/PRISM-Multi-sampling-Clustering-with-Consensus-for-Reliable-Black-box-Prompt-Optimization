import gc
import os
import shutil
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_CACHE_PATH, prompt_template_vicuna
from helper import (
    DOLLY_EVAL,
    HF_TOKEN,
    SELF_INSTRUCT_EVAL,
    VICUNA_7B,
    base_llm_models,
    clean_name,
    create_combined_name,
    device,
    evaluation_datasets,
    evaluator_models,
    LLAMA2_7B,
    GEMMA3,
)
# base_llm_models = [LLAMA2_7B]

from msd_config import (
    MSD_MERGE_ROOT,
    MSD_EMBEDDING_MODELS,
    embedded_json_path,
    load_json,
    parse_seed_args,
    save_json,
    set_seed,
)
from utils import generate_batch


PROMPT_KEYS = [
    "bpo_prompt",
    "rbpo_prompt",
    "mepo_prompt",
    "rmepo_prompt",
    "generic_prompt",
    "rgeneric_prompt",
]

def has_text(value):
    return isinstance(value, str) and bool(value.strip())


def prompt_with_context(prompt, context):
    if has_text(context):
        return f"""Context:
{context}

Question:
{prompt}
"""
    return prompt


def format_generation_prompt(tokenizer, prompt, context, is_vicuna):
    user_prompt = prompt_with_context(prompt, context)
    if is_vicuna:
        return prompt_template_vicuna.format(user_prompt)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful and concise assistant. "
                "Please reply in English only."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def collect_generation_jobs(data, tokenizer, is_vicuna, needs_context, force=False):
    stats = Counter()
    jobs = []
    unique_prompts = []
    prompt_to_index = {}

    for item_index, item in enumerate(data):
        item_has_generation = False
        context = item.get("context") if needs_context else None

        for key in PROMPT_KEYS:
            if key not in item:
                stats["missing_prompt_field"] += 1
                continue

            prompt = item.get(key)
            if not has_text(prompt):
                stats["empty_prompt"] += 1
                continue

            method = key[: -len("_prompt")]
            response_key = f"{method}_response"
            if not force and has_text(item.get(response_key)):
                stats["already_has_response"] += 1
                continue

            formatted_prompt = format_generation_prompt(
                tokenizer,
                prompt,
                context,
                is_vicuna,
            )
            if formatted_prompt not in prompt_to_index:
                prompt_to_index[formatted_prompt] = len(unique_prompts)
                unique_prompts.append(formatted_prompt)

            jobs.append((item_index, response_key, prompt_to_index[formatted_prompt]))
            item_has_generation = True

        if item_has_generation:
            stats["items_changed"] += 1
        else:
            stats["items_without_generation"] += 1

    stats["generated_responses"] = len(jobs)
    stats["unique_prompts_generated"] = len(unique_prompts)
    return jobs, unique_prompts, stats


def generate_file_responses(
    model,
    tokenizer,
    data,
    is_vicuna,
    needs_context,
    batch_size,
    max_new_tokens,
    force=False,
):
    jobs, unique_prompts, stats = collect_generation_jobs(
        data,
        tokenizer,
        is_vicuna,
        needs_context,
        force=force,
    )
    if not jobs:
        return stats

    unique_responses = [None] * len(unique_prompts)
    position = 0
    current_batch_size = min(batch_size, len(unique_prompts))
    progress = tqdm(
        total=len(unique_prompts),
        desc="Generating unique prompts",
        unit="prompt",
    )

    while position < len(unique_prompts):
        batch_prompts = unique_prompts[position : position + current_batch_size]
        try:
            with torch.inference_mode():
                batch_responses = generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    batch_size=current_batch_size,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    apply_chat_template=False,
                    device=device,
                )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if current_batch_size == 1:
                progress.close()
                raise RuntimeError(
                    "CUDA OOM at batch_size=1. Reduce --batch-size workload "
                    "or stop other GPU processes before retrying."
                ) from None
            current_batch_size = max(1, current_batch_size // 2)
            print(f"CUDA OOM: retrying with batch_size={current_batch_size}")
            continue

        for offset, response in enumerate(batch_responses):
            unique_responses[position + offset] = response

        position += len(batch_responses)
        progress.update(len(batch_responses))

    progress.close()

    for item_index, response_key, unique_index in jobs:
        data[item_index][response_key] = unique_responses[unique_index]

    return stats


def print_generation_stats(experiment_name, stats, total_items):
    total_prompt_fields = total_items * len(PROMPT_KEYS)
    skipped_fields = total_prompt_fields - stats["generated_responses"]
    print(
        f"Stats for {experiment_name}: "
        f"items={total_items}, prompt_fields={total_prompt_fields}, "
        f"items_changed={stats['items_changed']}, "
        f"items_without_generation={stats['items_without_generation']}, "
        f"generated_responses={stats['generated_responses']}, "
        f"unique_prompts_generated={stats['unique_prompts_generated']}, "
        f"skipped_fields={skipped_fields}, "
        f"missing_prompt_field={stats['missing_prompt_field']}, "
        f"empty_prompt={stats['empty_prompt']}, "
        f"already_has_response={stats['already_has_response']}"
    )


def run_seed(seed, args):
    set_seed(seed)
    print(f"\n===== MSD STEP 3 | seed={seed} =====")
    seed_stats = Counter()

    for embedding_model_name in MSD_EMBEDDING_MODELS:
        for base_model in base_llm_models:
            torch.cuda.empty_cache()
            gc.collect()

            print(f"Loading base model: {base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                cache_dir=MODEL_CACHE_PATH,
                token=HF_TOKEN,
                torch_dtype="auto",
            ).eval().to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                cache_dir=MODEL_CACHE_PATH,
                token=HF_TOKEN,
                legacy=False,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            is_vicuna = base_model == VICUNA_7B
            for dataset in evaluation_datasets:
                needs_context = dataset in [DOLLY_EVAL, SELF_INSTRUCT_EVAL]
                for evaluator in evaluator_models:
                    experiment_name = create_combined_name(base_model, dataset, evaluator)
                    if experiment_name not in args.experiment_names:
                        continue

                    path = embedded_json_path(
                        seed,
                        embedding_model_name,
                        experiment_name,
                        args.output_root,
                    )
                    if not path.exists():
                        print(f"Missing clustered file, skipping: {path}")
                        continue

                    data = load_json(path)
                    print(
                        f"Generating responses for seed={seed}, "
                        f"embed={clean_name(embedding_model_name)}, "
                        f"experiment={experiment_name}, n={len(data)}, "
                        f"batch_size={args.batch_size}, "
                        f"max_new_tokens={args.max_new_tokens}"
                    )
                    experiment_stats = generate_file_responses(
                        model,
                        tokenizer,
                        data,
                        is_vicuna,
                        needs_context,
                        batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens,
                        force=args.force,
                    )
                    save_json(path, data)
                    print(f"Saved {path}")
                    print_generation_stats(experiment_name, experiment_stats, len(data))
                    seed_stats.update(experiment_stats)

            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            if os.path.exists(MODEL_CACHE_PATH):
                shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)

    print(
        f"\nMSD STEP 3 summary | seed={seed}: "
        f"items_changed={seed_stats['items_changed']}, "
        f"items_without_generation={seed_stats['items_without_generation']}, "
        f"generated_responses={seed_stats['generated_responses']}, "
        f"unique_prompts_generated={seed_stats['unique_prompts_generated']}, "
        f"missing_prompt_field={seed_stats['missing_prompt_field']}, "
        f"empty_prompt={seed_stats['empty_prompt']}, "
        f"already_has_response={seed_stats['already_has_response']}"
    )
    return seed_stats


def main():
    args = parse_seed_args(
        "Generate downstream responses for MSD seed outputs.",
        default_output_root=MSD_MERGE_ROOT,
    )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")

    all_stats = Counter()
    for seed in args.seed_values:
        all_stats.update(run_seed(seed, args))
    print(
        f"\nMSD STEP 3 final summary: "
        f"items_changed={all_stats['items_changed']}, "
        f"items_without_generation={all_stats['items_without_generation']}, "
        f"generated_responses={all_stats['generated_responses']}, "
        f"unique_prompts_generated={all_stats['unique_prompts_generated']}, "
        f"missing_prompt_field={all_stats['missing_prompt_field']}, "
        f"empty_prompt={all_stats['empty_prompt']}, "
        f"already_has_response={all_stats['already_has_response']}"
    )


if __name__ == "__main__":
    main()
