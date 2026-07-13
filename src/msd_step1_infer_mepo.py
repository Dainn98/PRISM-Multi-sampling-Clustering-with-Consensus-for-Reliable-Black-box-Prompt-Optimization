import gc
import os
import shutil

import torch
from tqdm import tqdm

from config import MODEL_CACHE_PATH
from helper import M, clean_name
from mepo_inference import MePOModel
from msd_config import (
    MSD_EMBEDDING_MODELS,
    ensure_seed_input,
    parse_seed_args,
    save_json,
    set_seed,
)


def run_seed(seed, args):
    set_seed(seed)
    print(f"\n===== MSD STEP 1-MePO | seed={seed} =====")
    model = MePOModel()
    batch_size = args.batch_size

    for embedding_model_name in MSD_EMBEDDING_MODELS:
        print(f"Writing Step 1-MePO outputs under {clean_name(embedding_model_name)}")
        for experiment_name in args.experiment_names:
            path, data = ensure_seed_input(
                seed,
                experiment_name,
                args.output_root,
                embedding_model_name=embedding_model_name,
            )
            print(f"Processing {path}")

            pending_prompts = []
            pending_indices = []
            for index, item in enumerate(data):
                if args.force or not item.get("mepo_prompt"):
                    ori_prompt = item.get("ori_prompt", "")
                    pending_prompts.append(model.po_prompt_ins.replace("S_P", ori_prompt))
                    pending_indices.append(index)

            mepo_batches = range(0, len(pending_prompts), batch_size)
            for start in tqdm(
                mepo_batches,
                desc=f"MePO seed={seed} {experiment_name}",
                unit="batch",
            ):
                outputs = model.generate_batch(pending_prompts[start : start + batch_size])
                for item_index, output in zip(
                    pending_indices[start : start + batch_size],
                    outputs,
                ):
                    data[item_index]["mepo_prompt"] = output
                save_json(path, data)

            for index, item in enumerate(data):
                existing = item.get("rmepo_paraphrases", [])
                if not isinstance(existing, list):
                    existing = []
                missing = M if args.force else M - len(existing)
                if missing <= 0:
                    continue

                item["rmepo_paraphrases"] = [] if args.force else existing
                ori_prompt = item.get("ori_prompt", "")
                paraphrase_prompts = [
                    model.po_prompt_ins.replace("S_P", ori_prompt)
                    for _ in range(missing)
                ]
                for start in tqdm(
                    range(0, len(paraphrase_prompts), batch_size),
                    desc=(
                        f"RMePO seed={seed} {experiment_name} "
                        f"item={index + 1}/{len(data)}"
                    ),
                    unit="batch",
                ):
                    outputs = model.generate_paraphrase_batch(
                        paraphrase_prompts[start : start + batch_size]
                    )
                    item["rmepo_paraphrases"].extend(outputs)
                    save_json(path, data)

            save_json(path, data)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    if os.path.exists(MODEL_CACHE_PATH):
        shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)


def main():
    args = parse_seed_args("Generate MePO/RMePO prompts for independent MSD seeds.")
    args.batch_size = getattr(args, "batch_size", 8)
    for seed in args.seed_values:
        run_seed(seed, args)


if __name__ == "__main__":
    main()
