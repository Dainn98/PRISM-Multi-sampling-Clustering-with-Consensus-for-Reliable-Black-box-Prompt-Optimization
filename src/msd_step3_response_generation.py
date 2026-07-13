import gc
import os
import shutil

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
    available_keys = [
        key
        for key in PROMPT_KEYS
        if isinstance(item.get(key), str) and item[key].strip()
        and (force or not item.get(f"{key[: -len('_prompt')]}_response"))
    ]
    if not available_keys:
        return False

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
    return True


def run_seed(seed, args):
    set_seed(seed)
    print(f"\n===== MSD STEP 3 | seed={seed} =====")

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
                    for item in progress:
                        changed = generate_item_responses(
                            model,
                            tokenizer,
                            item,
                            is_vicuna,
                            needs_context,
                            force=args.force,
                        )
                        if changed:
                            save_json(path, data)
                    save_json(path, data)
                    print(f"Saved {path}")

            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            if os.path.exists(MODEL_CACHE_PATH):
                shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)


def main():
    args = parse_seed_args("Generate downstream responses for MSD seed outputs.")
    for seed in args.seed_values:
        run_seed(seed, args)


if __name__ == "__main__":
    main()
