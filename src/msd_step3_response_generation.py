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
)
from msd_config import (
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


def generate_item_responses(model, tokenizer, item, is_vicuna, needs_context, force=False):
    stats = Counter()
    available_keys = []
    for key in PROMPT_KEYS:
        if key not in item:
            stats["missing_prompt_field"] += 1
            continue

        prompt = item.get(key)
        if not isinstance(prompt, str) or not prompt.strip():
            stats["empty_prompt"] += 1
            continue

        response_key = f"{key[: -len('_prompt')]}_response"
        if not force and item.get(response_key):
            stats["already_has_response"] += 1
            continue

        available_keys.append(key)

    if not available_keys:
        stats["items_without_generation"] += 1
        return stats

    unique_prompts = []
    prompt_to_index = {}
    key_to_unique = {}
    for key in available_keys:
        prompt = item[key]
        if prompt not in prompt_to_index:
            prompt_to_index[prompt] = len(unique_prompts)
            unique_prompts.append(prompt)
        key_to_unique[key] = prompt_to_index[prompt]

    generation_prompts = unique_prompts
    if is_vicuna:
        generation_prompts = [prompt_template_vicuna.format(prompt) for prompt in unique_prompts]

    unique_responses = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=generation_prompts,
        context=item.get("context") if needs_context else None,
        do_sample=False,
        apply_chat_template=not is_vicuna,
        device=device,
    )

    for key in available_keys:
        method = key[: -len("_prompt")]
        item[f"{method}_response"] = unique_responses[key_to_unique[key]]
    stats["items_changed"] += 1
    stats["generated_responses"] += len(available_keys)
    stats["unique_prompts_generated"] += len(unique_prompts)
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
                        f"experiment={experiment_name}, n={len(data)}"
                    )
                    progress = tqdm(
                        data,
                        desc=f"Responses seed={seed} {clean_name(embedding_model_name)} {experiment_name}",
                        unit="item",
                    )
                    experiment_stats = Counter()
                    for item in progress:
                        item_stats = generate_item_responses(
                            model,
                            tokenizer,
                            item,
                            is_vicuna,
                            needs_context,
                            force=args.force,
                        )
                        experiment_stats.update(item_stats)
                        if item_stats["items_changed"]:
                            save_json(path, data)
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
    args = parse_seed_args("Generate downstream responses for MSD seed outputs.")
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
