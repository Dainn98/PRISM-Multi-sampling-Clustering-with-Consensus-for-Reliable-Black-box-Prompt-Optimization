"""Profile latency and GPU-time cost of the Llama-based PRISM pipeline."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence, TypeVar

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_CACHE_PATH, prompt_template_optimize
from helper import (
    IMP_ENC,
    LLAMA2_7B,
    MINILM_EMBEDDING_MODEL,
    distance_thresholds,
    set_global_seed,
)
from utils import make_prompt_template


T = TypeVar("T")
MODULE_COLUMNS = [
    "candidate_generation_ms",
    "embedding_ms",
    "clustering_ms",
    "representative_selection_ms",
    "consensus_scoring_ms",
    "total_prism_ms",
    "base_llm_generation_ms",
    "end_to_end_ms",
]
PROMPT_FIELDS = (
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
        help="JSON list containing id/ori_prompt fields",
    )
    parser.add_argument("--output-dir", default="src/extra_results/latency_llama")
    parser.add_argument("--candidate-counts", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--batch-modes", nargs="+", choices=["sequential", "full"], default=["full"])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--optimizer-model", default=LLAMA2_7B)
    parser.add_argument("--base-model", default=LLAMA2_7B)
    parser.add_argument("--embedding-model", default=MINILM_EMBEDDING_MODEL)
    parser.add_argument("--distance-threshold", type=float)
    parser.add_argument("--creativity-coefficient", type=float, default=IMP_ENC)
    parser.add_argument("--candidate-max-new-tokens", type=int, default=512)
    parser.add_argument("--response-max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--gpu-price-per-hour", type=float)
    parser.add_argument("--currency", default="USD")
    return parser.parse_args()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timer() -> Iterator[Callable[[], float]]:
    synchronize()
    start = time.perf_counter()
    elapsed = [0.0]

    def milliseconds() -> float:
        return elapsed[0]

    try:
        yield milliseconds
    finally:
        synchronize()
        elapsed[0] = (time.perf_counter() - start) * 1000.0


def model_input_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_causal_model(model_name: str, token: str | None) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_PATH,
        token=token,
        torch_dtype="auto",
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_PATH,
        token=token,
        legacy=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def prepare_base_prompt(tokenizer: AutoTokenizer, prompt: str, context: str | None) -> str:
    messages = make_prompt_template(prompt, context=context)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if context:
        return f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
    return prompt


def get_prompt(item: dict) -> str:
    for field in PROMPT_FIELDS:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_sample_id(item: dict, fallback: int) -> str:
    for field in ("id", "idx", "sample_id", "prompt_id"):
        value = item.get(field)
        if value is not None:
            return str(value)
    return str(fallback)


@torch.inference_mode()
def generate_texts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    do_sample: bool,
    batch_mode: str,
    temperature: float | None = None,
    top_p: float | None = None,
) -> tuple[list[str], int, int]:
    results: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    batches = [[prompt] for prompt in prompts] if batch_mode == "sequential" else [list(prompts)]

    for batch in batches:
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model_input_device(model))
        input_width = inputs["input_ids"].shape[1]
        total_input_tokens += int(inputs["attention_mask"].sum().item())
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        outputs = model.generate(**inputs, **kwargs)
        generated = outputs[:, input_width:]
        total_output_tokens += int((generated != tokenizer.pad_token_id).sum().item())
        results.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    return [text.strip() for text in results], total_input_tokens, total_output_tokens


def embed_prompts(
    embedding_model: SentenceTransformer,
    original_prompt: str,
    candidates: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = embedding_model.encode(
        [original_prompt, *candidates],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings[0], embeddings[1:]


def cluster_candidates(candidate_embeddings: torch.Tensor, threshold: float) -> list[list[int]]:
    count = len(candidate_embeddings)
    if count == 1:
        return [[0]]
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    ).fit_predict(candidate_embeddings.detach().cpu().numpy())
    clusters: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(index)
    return list(clusters.values())


def choose_representatives(
    clusters: Sequence[Sequence[int]],
    original_embedding: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> list[int]:
    representatives = []
    for cluster in clusters:
        if len(cluster) == 1:
            representatives.append(cluster[0])
            continue
        similarities = util.cos_sim(
            original_embedding,
            candidate_embeddings[list(cluster)],
        )[0]
        order = torch.argsort(similarities)
        representatives.append(cluster[int(order[len(order) // 2].item())])
    return representatives


def select_by_consensus(
    representatives: Sequence[int],
    original_embedding: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    creativity_coefficient: float,
) -> tuple[int, list[float]]:
    if len(representatives) == 1:
        return representatives[0], [0.0]
    rep_embeddings = candidate_embeddings[list(representatives)]
    pairwise = util.cos_sim(rep_embeddings, rep_embeddings)
    originality = util.cos_sim(rep_embeddings, original_embedding).squeeze(1)
    scores = pairwise.sum(dim=1) - 1.0 - creativity_coefficient * originality
    best = int(torch.argmax(scores).item())
    return representatives[best], [float(value) for value in scores.detach().cpu()]


def run_pipeline(
    item: dict,
    candidate_count: int,
    batch_mode: str,
    optimizer_model: AutoModelForCausalLM,
    optimizer_tokenizer: AutoTokenizer,
    embedding_model: SentenceTransformer,
    base_model: AutoModelForCausalLM,
    base_tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    threshold: float,
) -> dict:
    original_prompt = get_prompt(item)
    if not original_prompt:
        raise ValueError(f"Dataset item has no prompt field among {PROMPT_FIELDS}")
    context = item.get("context")
    optimizer_inputs = [prompt_template_optimize.format(original_prompt)] * candidate_count

    synchronize()
    end_to_end_start = time.perf_counter()

    with timer() as elapsed:
        candidates, candidate_input_tokens, candidate_output_tokens = generate_texts(
            optimizer_model,
            optimizer_tokenizer,
            optimizer_inputs,
            max_new_tokens=args.candidate_max_new_tokens,
            do_sample=True,
            batch_mode=batch_mode,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    candidate_generation_ms = elapsed()

    with timer() as elapsed:
        original_embedding, candidate_embeddings = embed_prompts(
            embedding_model, original_prompt, candidates
        )
    embedding_ms = elapsed()

    with timer() as elapsed:
        clusters = cluster_candidates(candidate_embeddings, threshold)
    clustering_ms = elapsed()

    with timer() as elapsed:
        representatives = choose_representatives(
            clusters, original_embedding, candidate_embeddings
        )
    representative_selection_ms = elapsed()

    with timer() as elapsed:
        best_index, _ = select_by_consensus(
            representatives,
            original_embedding,
            candidate_embeddings,
            args.creativity_coefficient,
        )
    consensus_scoring_ms = elapsed()

    final_prompt = candidates[best_index]
    formatted_prompt = prepare_base_prompt(base_tokenizer, final_prompt, context)
    with timer() as elapsed:
        responses, base_input_tokens, base_output_tokens = generate_texts(
            base_model,
            base_tokenizer,
            [formatted_prompt],
            max_new_tokens=args.response_max_new_tokens,
            do_sample=False,
            batch_mode="full",
        )
    base_llm_generation_ms = elapsed()
    synchronize()
    end_to_end_ms = (time.perf_counter() - end_to_end_start) * 1000.0

    total_prism_ms = (
        candidate_generation_ms
        + embedding_ms
        + clustering_ms
        + representative_selection_ms
        + consensus_scoring_ms
    )
    return {
        "num_clusters": len(clusters),
        "num_representatives": len(representatives),
        "original_prompt_tokens": len(
            optimizer_tokenizer(original_prompt, add_special_tokens=False)["input_ids"]
        ),
        "total_candidate_input_tokens": candidate_input_tokens,
        "total_candidate_output_tokens": candidate_output_tokens,
        "final_prompt_tokens": len(
            base_tokenizer(final_prompt, add_special_tokens=False)["input_ids"]
        ),
        "base_llm_input_tokens": base_input_tokens,
        "base_llm_output_tokens": base_output_tokens,
        "candidate_generation_ms": candidate_generation_ms,
        "embedding_ms": embedding_ms,
        "clustering_ms": clustering_ms,
        "representative_selection_ms": representative_selection_ms,
        "consensus_scoring_ms": consensus_scoring_ms,
        "total_prism_ms": total_prism_ms,
        "base_llm_generation_ms": base_llm_generation_ms,
        "end_to_end_ms": end_to_end_ms,
        "candidate_generation_ms_per_output_token": (
            candidate_generation_ms / candidate_output_tokens
            if candidate_output_tokens
            else np.nan
        ),
        "base_llm_generation_ms_per_output_token": (
            base_llm_generation_ms / base_output_tokens if base_output_tokens else np.nan
        ),
        "response_empty": int(not responses or not responses[0]),
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict], gpu_price: float | None, currency: str) -> list[dict]:
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        if row["status"] == "ok":
            groups.setdefault((int(row["m"]), str(row["batch_mode"])), []).append(row)

    summaries = []
    for (candidate_count, batch_mode), group in sorted(groups.items()):
        # Average repetitions within a prompt before aggregating across prompts.
        prompt_groups: dict[str, list[dict]] = {}
        for row in group:
            prompt_groups.setdefault(str(row["sample_id"]), []).append(row)
        prompt_means = []
        for sample_rows in prompt_groups.values():
            prompt_means.append(
                {
                    column: float(np.mean([float(row[column]) for row in sample_rows]))
                    for column in MODULE_COLUMNS
                }
            )

        summary: dict[str, float | int | str] = {
            "m": candidate_count,
            "batch_mode": batch_mode,
            "num_samples": len(prompt_means),
            "repetitions": len(group) // max(len(prompt_means), 1),
        }
        for column in MODULE_COLUMNS:
            values = np.array([row[column] for row in prompt_means], dtype=float)
            prefix = column.removesuffix("_ms")
            summary[f"{prefix}_mean_ms"] = float(values.mean())
            summary[f"{prefix}_std_ms"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary[f"{prefix}_median_ms"] = float(np.median(values))
            summary[f"{prefix}_p90_ms"] = float(np.percentile(values, 90))
            summary[f"{prefix}_p95_ms"] = float(np.percentile(values, 95))

        total = float(summary["total_prism_mean_ms"])
        for column in MODULE_COLUMNS[:5]:
            prefix = column.removesuffix("_ms")
            summary[f"{prefix}_share_pct"] = (
                float(summary[f"{prefix}_mean_ms"]) / total * 100.0 if total else 0.0
            )
        end_to_end_seconds = float(summary["end_to_end_mean_ms"]) / 1000.0
        summary["gpu_seconds_per_sample"] = end_to_end_seconds
        summary["gpu_hours_per_sample"] = end_to_end_seconds / 3600.0
        summary["gpu_hours_per_dataset"] = (
            float(summary["gpu_hours_per_sample"]) * len(prompt_means)
        )
        summary["gpu_price_per_hour"] = gpu_price if gpu_price is not None else ""
        summary["currency"] = currency if gpu_price is not None else ""
        summary["cost_per_sample"] = (
            float(summary["gpu_hours_per_sample"]) * gpu_price
            if gpu_price is not None
            else ""
        )
        summary["cost_per_dataset"] = (
            float(summary["gpu_hours_per_dataset"]) * gpu_price
            if gpu_price is not None
            else ""
        )
        summaries.append(summary)

    baselines = {
        str(row["batch_mode"]): float(row["total_prism_mean_ms"])
        for row in summaries
        if int(row["m"]) == 1
    }
    for row in summaries:
        baseline = baselines.get(str(row["batch_mode"]))
        row["relative_total_prism_latency"] = (
            float(row["total_prism_mean_ms"]) / baseline if baseline else np.nan
        )
    return summaries


def metadata(args: argparse.Namespace, threshold: float) -> dict:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    )
    return {
        "execution_date_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_memory_bytes": gpu_memory,
        "optimizer_checkpoint": args.optimizer_model,
        "base_llm_checkpoint": args.base_model,
        "embedding_model": args.embedding_model,
        "torch_dtype": "auto",
        "quantization": None,
        "candidate_counts": args.candidate_counts,
        "batch_modes": args.batch_modes,
        "candidate_max_new_tokens": args.candidate_max_new_tokens,
        "response_max_new_tokens": args.response_max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "distance_threshold": threshold,
        "creativity_coefficient": args.creativity_coefficient,
        "repetitions": args.repetitions,
        "warmup_runs": args.warmup_runs,
        "random_seed": args.random_seed,
    }


def main() -> int:
    args = parse_args()
    if any(value < 1 for value in args.candidate_counts):
        print("ERROR: candidate counts must be positive", file=sys.stderr)
        return 2
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    threshold = args.distance_threshold
    if threshold is None:
        threshold = distance_thresholds.get(args.embedding_model)
    if threshold is None:
        print("ERROR: provide --distance-threshold for this embedding model", file=sys.stderr)
        return 2

    dataset_path = Path(args.dataset)
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, list):
        print("ERROR: dataset must be a JSON list", file=sys.stderr)
        return 2
    original_size = len(dataset)
    dataset = [item for item in dataset if isinstance(item, dict) and get_prompt(item)]
    skipped = original_size - len(dataset)
    if skipped:
        print(f"Skipped {skipped} samples without a prompt field")
    dataset = dataset[: args.max_samples] if args.max_samples else dataset
    if not dataset:
        print("ERROR: dataset is empty", file=sys.stderr)
        return 2

    print("Loading optimizer, embedding model, and base LLM...")
    optimizer_model, optimizer_tokenizer = load_causal_model(args.optimizer_model, token)
    embedding_model = SentenceTransformer(
        args.embedding_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_folder=MODEL_CACHE_PATH,
    )
    base_model, base_tokenizer = load_causal_model(args.base_model, token)

    warmup_m = min(args.candidate_counts)
    print(f"Running {args.warmup_runs} warm-up iterations...")
    for warmup_index in range(args.warmup_runs):
        set_global_seed(args.random_seed)
        run_pipeline(
            dataset[warmup_index % len(dataset)],
            warmup_m,
            args.batch_modes[0],
            optimizer_model,
            optimizer_tokenizer,
            embedding_model,
            base_model,
            base_tokenizer,
            args,
            threshold,
        )

    raw_rows: list[dict] = []
    for batch_mode in args.batch_modes:
        for candidate_count in args.candidate_counts:
            print(f"Profiling m={candidate_count}, batch_mode={batch_mode}")
            for sample_index, item in enumerate(dataset):
                sample_id = get_sample_id(item, sample_index)
                for repetition in range(1, args.repetitions + 1):
                    set_global_seed(args.random_seed)
                    prefix = {
                        "sample_id": sample_id,
                        "m": candidate_count,
                        "batch_mode": batch_mode,
                        "batch_size": 1 if batch_mode == "sequential" else candidate_count,
                        "repetition": repetition,
                    }
                    try:
                        row = run_pipeline(
                            item,
                            candidate_count,
                            batch_mode,
                            optimizer_model,
                            optimizer_tokenizer,
                            embedding_model,
                            base_model,
                            base_tokenizer,
                            args,
                            threshold,
                        )
                        raw_rows.append({**prefix, "status": "ok", "error": "", **row})
                    except torch.cuda.OutOfMemoryError as error:
                        raw_rows.append({**prefix, "status": "oom", "error": str(error)})
                        torch.cuda.empty_cache()
                        gc.collect()
                        print(f"  OOM for sample={sample_id}; continuing")
                    except Exception as error:
                        raw_rows.append({**prefix, "status": "error", "error": repr(error)})
                        print(f"  ERROR for sample={sample_id}: {error}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "latency_raw.csv", raw_rows)
    summary_rows = summarize(raw_rows, args.gpu_price_per_hour, args.currency)
    write_csv(output_dir / "latency_summary.csv", summary_rows)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata(args, threshold), handle, indent=2)
    print(f"Saved latency results to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
