import gc
import os
import re
import shutil

import torch
from tqdm import tqdm

from config import MODEL_CACHE_PATH
from helper import GENERIC_LLM_REWRITER, M, clean_name, load_model_and_tokenizer
from msd_config import (
    MSD_EMBEDDING_MODELS,
    ensure_seed_input,
    parse_seed_args,
    save_json,
    set_seed,
)


SYSTEM_PROMPT = """
You are a general-purpose prompt rewriter.
Rewrite the user's prompt so it is clearer, more specific, and easier for a
language model to follow. Preserve the original intent and return only the
rewritten prompt.
""".strip()

VARIATION_FOCUSES = [
    "Prioritize unambiguous wording and explicit task structure.",
    "Prioritize concise wording while retaining every requirement.",
    "Prioritize logical ordering of instructions and constraints.",
    "Prioritize clarity about the expected output.",
    "Prioritize preserving subtle intent while removing ambiguity.",
    "Prioritize readability for a general-purpose language model.",
    "Prioritize precise verbs and concrete references.",
    "Prioritize grouping related requirements together.",
    "Prioritize direct language without unnecessary verbosity.",
    "Prioritize faithful reformulation with a distinct sentence structure.",
]


def normalize_rewrite(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def apply_chat_template(tokenizer, messages):
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def rewrite_prompt(model, tokenizer, original_prompt, variation_index=None):
    if variation_index is None:
        variation = "Produce the single best faithful rewrite of the original prompt."
    else:
        focus = VARIATION_FOCUSES[variation_index % len(VARIATION_FOCUSES)]
        variation = f"Produce rewrite candidate {variation_index + 1}. {focus}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{variation}\n\nOriginal prompt:\n{original_prompt}",
        },
    ]
    prompt = apply_chat_template(tokenizer, messages)
    model_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model_device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
    rewritten = normalize_rewrite(tokenizer.decode(generated_ids, skip_special_tokens=True))
    if not rewritten:
        raise ValueError("Generic rewriter returned an empty prompt")
    return rewritten


def run_seed(seed, args):
    set_seed(seed)
    print(f"\n===== MSD STEP 1-Generic | seed={seed} =====")
    model, tokenizer = load_model_and_tokenizer(
        GENERIC_LLM_REWRITER,
        cache_dir=MODEL_CACHE_PATH,
    )

    for embedding_model_name in MSD_EMBEDDING_MODELS:
        print(f"Writing Step 1-Generic outputs under {clean_name(embedding_model_name)}")
        for experiment_name in args.experiment_names:
            path, data = ensure_seed_input(
                seed,
                experiment_name,
                args.output_root,
                embedding_model_name=embedding_model_name,
            )
            print(f"Processing {path}")

            progress = tqdm(
                data,
                desc=f"Generic seed={seed} {experiment_name}",
                unit="item",
            )
            for item in progress:
                original_prompt = item.get("ori_prompt", "")
                if not original_prompt:
                    continue

                if args.force or not item.get("generic_prompt"):
                    progress.set_postfix(stage="generic")
                    item["generic_prompt"] = rewrite_prompt(model, tokenizer, original_prompt)
                    save_json(path, data)

                paraphrases = item.get("generic_paraphrases", [])
                if args.force or len(paraphrases) < M:
                    paraphrases = [] if args.force else paraphrases
                    while len(paraphrases) < M:
                        progress.set_postfix(
                            stage="generic_paraphrases",
                            candidates=f"{len(paraphrases) + 1}/{M}",
                        )
                        paraphrases.append(
                            rewrite_prompt(
                                model,
                                tokenizer,
                                original_prompt,
                                variation_index=len(paraphrases),
                            )
                        )
                        item["generic_paraphrases"] = paraphrases
                        save_json(path, data)

            save_json(path, data)

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    if os.path.exists(MODEL_CACHE_PATH):
        shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)


def main():
    args = parse_seed_args("Generate generic prompt rewrites for MSD seeds.")
    for seed in args.seed_values:
        run_seed(seed, args)


if __name__ == "__main__":
    main()
