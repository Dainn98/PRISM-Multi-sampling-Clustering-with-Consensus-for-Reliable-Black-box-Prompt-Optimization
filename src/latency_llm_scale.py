"""Compare base-LLM response-generation latency across model scales.

Only the response-generation path (tokenization, ``model.generate``, and
decoding) is timed. Model loading and the warm-up run are excluded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_CACHE_PATH, prompt_template_vicuna
from helper import GEMMA3, LLAMA2_7B, VICUNA_13B, VICUNA_7B, set_global_seed


DEFAULT_MODELS = [LLAMA2_7B, VICUNA_7B, GEMMA3, VICUNA_13B]
AUTO_PROMPT_FIELDS = (
    "prism_prompt",
    "rmepo_prompt",
    "optimized_prompt",
    "ori_prompt",
    "instruction",
    "prompt",
    "question",
    "input",
    "query",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="src/testset/instruction-following/dolly_eval.json",
        help="JSON list containing the prompts to benchmark",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Hugging Face model IDs",
    )
    parser.add_argument(
        "--prompt-field",
        default="auto",
        help="Prompt key in each item; 'auto' checks common PRISM/dataset keys",
    )
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="src/extra_results/latency_llm_scale",
    )
    return parser.parse_args()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_prompt(item: dict, prompt_field: str) -> str:
    fields = AUTO_PROMPT_FIELDS if prompt_field == "auto" else (prompt_field,)
    for field in fields:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_sample_id(item: dict, fallback: int) -> str:
    for field in ("id", "idx", "sample_id", "prompt_id"):
        if item.get(field) is not None:
            return str(item[field])
    return str(fallback)


def load_model(model_name: str, token: str | None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_PATH,
        token=token,
        legacy=False,
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_PATH,
        token=token,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def format_prompt(tokenizer, model_name: str, prompt: str, context: str | None) -> str:
    if isinstance(context, str) and context.strip():
        user_content = f"Context:\n{context.strip()}\n\nQuestion:\n{prompt}"
    else:
        user_content = prompt

    # A user-only message works with Llama, Gemma, and modern Vicuna templates.
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if "vicuna" in model_name.lower():
        return prompt_template_vicuna.format(user_content)
    return user_content


@torch.inference_mode()
def generate_response(model, tokenizer, formatted_prompt: str, max_new_tokens: int):
    device = next(model.parameters()).device
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
    ).to(device)
    input_width = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_ids = outputs[0, input_width:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response, int(generated_ids.numel())


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.num_prompts < 1 or args.max_new_tokens < 1 or args.warmup_runs < 0:
        raise SystemExit("num-prompts/max-new-tokens must be positive; warmup-runs >= 0")

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    set_global_seed(args.seed)

    with Path(args.dataset).open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, list):
        raise SystemExit("Dataset must be a JSON list")

    samples = [
        item
        for item in dataset
        if isinstance(item, dict) and get_prompt(item, args.prompt_field)
    ][: args.num_prompts]
    if len(samples) < args.num_prompts:
        raise SystemExit(
            f"Dataset has only {len(samples)} usable prompts; need {args.num_prompts}"
        )

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []

    for model_name in args.models:
        print(f"\nLoading {model_name} ...")
        model = tokenizer = None
        try:
            model, tokenizer = load_model(model_name, token)
            warmup_prompt = format_prompt(
                tokenizer,
                model_name,
                get_prompt(samples[0], args.prompt_field),
                samples[0].get("context"),
            )
            print(f"Warm-up: {args.warmup_runs} run(s), excluded from latency")
            for _ in range(args.warmup_runs):
                generate_response(model, tokenizer, warmup_prompt, args.max_new_tokens)
            synchronize()

            latencies_ms = []
            for index, item in enumerate(samples):
                formatted_prompt = format_prompt(
                    tokenizer,
                    model_name,
                    get_prompt(item, args.prompt_field),
                    item.get("context"),
                )
                synchronize()
                start = time.perf_counter()
                response, output_tokens = generate_response(
                    model, tokenizer, formatted_prompt, args.max_new_tokens
                )
                synchronize()
                latency_ms = (time.perf_counter() - start) * 1000.0
                latencies_ms.append(latency_ms)
                raw_rows.append(
                    {
                        "model": model_name,
                        "sample_id": get_sample_id(item, index),
                        "latency_ms": latency_ms,
                        "latency_s": latency_ms / 1000.0,
                        "output_tokens": output_tokens,
                        "response": response,
                    }
                )
                print(
                    f"  prompt {index + 1}/{len(samples)}: "
                    f"{latency_ms / 1000.0:.3f} s ({output_tokens} tokens)"
                )

            values = np.asarray(latencies_ms, dtype=float)
            mean_ms = float(values.mean())
            std_ms = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary_rows.append(
                {
                    "model": model_name,
                    "num_prompts": len(values),
                    "mean_ms": mean_ms,
                    "std_ms": std_ms,
                    "mean_s": mean_ms / 1000.0,
                    "std_s": std_ms / 1000.0,
                    "mean_plus_minus_std_s": f"{mean_ms / 1000.0:.3f} ± {std_ms / 1000.0:.3f}",
                }
            )
        finally:
            model = None
            tokenizer = None
            release_memory()

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "latency_raw.csv", raw_rows)
    write_csv(output_dir / "latency_summary.csv", summary_rows)
    with (output_dir / "responses.json").open("w", encoding="utf-8") as handle:
        json.dump(raw_rows, handle, ensure_ascii=False, indent=2)

    print("\nBase LLM generation latency (mean ± sample std):")
    for row in summary_rows:
        print(f"  {row['model']}: {row['mean_plus_minus_std_s']} s")
    print(f"Saved results to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
