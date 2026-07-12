import gc

import torch, os, shutil

from config import MODEL_CACHE_PATH, prompt_template_optimize
from helper import BPO_MODEL, HF_TOKEN, M, clean_name, device, load_model_and_tokenizer
from msd_config import (
    MSD_EMBEDDING_MODELS,
    ensure_seed_input,
    parse_seed_args,
    save_json,
    set_seed,
)
from utils import generate, generate_batch


def run_seed(seed, args):
    set_seed(seed)
    print(f"\n===== MSD STEP 1-BPO | seed={seed} =====")
    model, tokenizer = load_model_and_tokenizer(
        BPO_MODEL,
        device_map="auto",
        cache_dir=MODEL_CACHE_PATH,
        token=HF_TOKEN,
    )

    for embedding_model_name in MSD_EMBEDDING_MODELS:
        print(f"Writing Step 1-BPO outputs under {clean_name(embedding_model_name)}")
        for experiment_name in args.experiment_names:
            path, data = ensure_seed_input(
                seed,
                experiment_name,
                args.output_root,
                force=args.force,
                embedding_model_name=embedding_model_name,
            )
            print(f"Processing {path}")

            for item in data:
                ori_prompt = item.get("ori_prompt", "")
                if not ori_prompt:
                    continue

                if args.force or not item.get("bpo_prompt"):
                    item["bpo_prompt"] = generate(
                        model,
                        tokenizer,
                        prompt_template_optimize.format(ori_prompt),
                        temperature=0.9,
                        top_p=0.9,
                        apply_chat_template=False,
                        device=device,
                    )

                if args.force or len(item.get("rbpo_paraphrases", [])) < M:
                    prompts = [prompt_template_optimize.format(ori_prompt) for _ in range(M)]
                    item["rbpo_paraphrases"] = generate_batch(
                        model,
                        tokenizer,
                        prompts,
                        temperature=0.9,
                        top_p=0.9,
                        apply_chat_template=False,
                        device=device,
                    )

            save_json(path, data)

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    if os.path.exists(MODEL_CACHE_PATH):
        shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)


def main():
    args = parse_seed_args("Generate BPO/RBPO prompts for independent MSD seeds.")
    for seed in args.seed_values:
        run_seed(seed, args)


if __name__ == "__main__":
    main()
