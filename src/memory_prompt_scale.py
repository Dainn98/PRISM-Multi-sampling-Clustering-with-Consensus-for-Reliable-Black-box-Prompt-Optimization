"""Measure PRISM memory use as the number of sampled prompts increases.

The embedding model is loaded once and excluded from the reported incremental
memory.  For every original prompt and candidate size ``m``, the complete
clustering-and-selection path is measured repeatedly.  Raw measurements and
mean/sample-standard-deviation summaries are written as CSV files.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

import psutil
import torch
from sentence_transformers import SentenceTransformer

from config import MODEL_CACHE_PATH
from helper import MINILM_EMBEDDING_MODEL, device, distance_thresholds, set_global_seed
from step2_clustering_and_selecting import (
    compute_consensus_score,
    optimize_prompt_selection,
    prompt_clustering,
    representative_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="src/ablation/llama_dolly_deepseek.json",
        help="JSON list containing ori_prompt and candidate lists",
    )
    parser.add_argument("--candidate-key", default="rmepo_paraphrases")
    parser.add_argument(
        "--m-values",
        nargs="+",
        type=int,
        default=[2,5,10,20,50,100, 200, 500, 1000],
        help="Candidate sizes to benchmark",
    )
    parser.add_argument(
        "--candidate-expansion",
        choices=("repeat", "error"),
        default="repeat",
        help=(
            "How to handle m larger than the available candidate list: "
            "cycle existing candidates for a synthetic scalability test, or fail"
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-model", default=MINILM_EMBEDDING_MODEL)
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=None,
        help="Defaults to the threshold configured for the embedding model",
    )
    parser.add_argument(
        "--poll-interval-ms",
        type=float,
        default=1.0,
        help="CPU RSS sampling interval",
    )
    parser.add_argument(
        "--output-dir",
        default="src/extra_results/memory_prompt_scale",
    )
    parser.add_argument("--isolated-child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


class RSSMonitor:
    """Poll process RSS in a background thread, including native allocations."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.process = psutil.Process(os.getpid())
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> int:
        baseline = self.process.memory_info().rss
        self.samples = [baseline]
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return baseline

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self.process.memory_info().rss)
            self._stop.wait(self.interval_s)

    def stop(self) -> tuple[int, float, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        final_rss = self.process.memory_info().rss
        self.samples.append(final_rss)
        return max(self.samples), statistics.fmean(self.samples), final_rss


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def release_temporary_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_prism_selection(
    item: dict,
    candidate_key: str,
    embedding_model: SentenceTransformer,
    m: int,
    distance_threshold: float,
) -> tuple[int, int]:
    clusters, embeddings = prompt_clustering(
        candidate_key, item, embedding_model, m, distance_threshold
    )
    if not clusters or embeddings is None:
        raise ValueError(f"Need at least two candidates for m={m}")
    representatives, single_cluster = representative_selection(
        item, embedding_model, clusters, embeddings
    )
    scores = compute_consensus_score(
        candidate_key,
        item,
        embedding_model,
        clusters,
        embeddings,
        representatives,
        single_cluster,
    )
    optimize_prompt_selection(
        candidate_key, item, clusters, embeddings, representatives, scores
    )
    return len(clusters), len(representatives)


def measure_once(operation: Callable[[], tuple[int, int]], interval_s: float) -> dict:
    release_temporary_memory()
    monitor = RSSMonitor(interval_s)
    baseline_rss = monitor.start()

    if torch.cuda.is_available():
        gpu_baseline_allocated = torch.cuda.memory_allocated()
        gpu_baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
    else:
        gpu_baseline_allocated = gpu_baseline_reserved = 0

    synchronize()
    start = time.perf_counter()
    num_clusters, num_representatives = operation()
    synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    peak_rss, mean_rss, final_rss = monitor.stop()
    if torch.cuda.is_available():
        peak_gpu_allocated = torch.cuda.max_memory_allocated()
        peak_gpu_reserved = torch.cuda.max_memory_reserved()
    else:
        peak_gpu_allocated = peak_gpu_reserved = 0

    mib = 1024**2
    return {
        "num_clusters": num_clusters,
        "num_representatives": num_representatives,
        "latency_ms": elapsed_ms,
        "cpu_baseline_rss_mib": baseline_rss / mib,
        "cpu_mean_rss_mib": mean_rss / mib,
        "cpu_peak_rss_mib": peak_rss / mib,
        "cpu_peak_delta_mib": max(0, peak_rss - baseline_rss) / mib,
        "cpu_final_delta_mib": (final_rss - baseline_rss) / mib,
        "gpu_baseline_allocated_mib": gpu_baseline_allocated / mib,
        "gpu_peak_allocated_mib": peak_gpu_allocated / mib,
        "gpu_peak_allocated_delta_mib": max(
            0, peak_gpu_allocated - gpu_baseline_allocated
        )
        / mib,
        "gpu_baseline_reserved_mib": gpu_baseline_reserved / mib,
        "gpu_peak_reserved_mib": peak_gpu_reserved / mib,
        "gpu_peak_reserved_delta_mib": max(
            0, peak_gpu_reserved - gpu_baseline_reserved
        )
        / mib,
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_items(
    data: Sequence[object],
    candidate_key: str,
    max_m: int,
    num_prompts: int,
    expansion: str,
) -> list[dict]:
    prepared = []
    for value in data:
        if not isinstance(value, dict) or not isinstance(value.get("ori_prompt"), str):
            continue
        candidates = value.get(candidate_key)
        if not isinstance(candidates, list) or len(candidates) < 2:
            continue
        if not all(isinstance(candidate, str) and candidate.strip() for candidate in candidates):
            continue
        if len(candidates) < max_m:
            if expansion == "error":
                continue
            expanded = list(itertools.islice(itertools.cycle(candidates), max_m))
        else:
            expanded = candidates
        item = dict(value)
        item[candidate_key] = expanded
        item["_memory_scale_source_candidates"] = len(candidates)
        prepared.append(item)
        if len(prepared) == num_prompts:
            break
    return prepared


def summarize(raw_rows: Sequence[dict]) -> list[dict]:
    metrics = (
        "latency_ms",
        "cpu_mean_rss_mib",
        "cpu_peak_rss_mib",
        "cpu_peak_delta_mib",
        "gpu_peak_allocated_mib",
        "gpu_peak_allocated_delta_mib",
        "gpu_peak_reserved_mib",
        "gpu_peak_reserved_delta_mib",
    )
    rows = []
    for m in sorted({int(row["m"]) for row in raw_rows}):
        group = [row for row in raw_rows if int(row["m"]) == m]
        summary = {
            "m": m,
            "num_measurements": len(group),
            "num_prompts": len({row["prompt_index"] for row in group}),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(summary)
    return rows


def validate_args(args: argparse.Namespace) -> None:
    if args.num_prompts < 1 or args.repeats < 1 or args.warmup_runs < 0:
        raise SystemExit("num-prompts/repeats must be positive; warmup-runs >= 0")
    if any(m < 2 for m in args.m_values):
        raise SystemExit("Every m value must be at least 2 for clustering")
    if args.poll_interval_ms <= 0:
        raise SystemExit("poll-interval-ms must be positive")


def run_benchmark(args: argparse.Namespace) -> int:
    """Run one process-local benchmark (normally for exactly one value of m)."""
    validate_args(args)

    set_global_seed(args.seed)
    with Path(args.dataset).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise SystemExit("Dataset must be a JSON list")

    max_m = max(args.m_values)
    usable = prepare_items(
        data,
        args.candidate_key,
        max_m,
        args.num_prompts,
        args.candidate_expansion,
    )
    if len(usable) < args.num_prompts:
        raise SystemExit(
            f"Found {len(usable)} usable prompts for max m={max_m}; requested "
            f"{args.num_prompts}. Use --candidate-expansion repeat or provide more "
            "real candidates."
        )
    source_counts = [item["_memory_scale_source_candidates"] for item in usable]
    if any(count < max_m for count in source_counts):
        print(
            "WARNING: m exceeds the available candidates; existing candidates are "
            "cycled. This measures algorithmic memory scaling, not diversity scaling."
        )

    threshold = args.distance_threshold
    if threshold is None:
        threshold = distance_thresholds.get(args.embedding_model)
    if threshold is None:
        raise SystemExit("Pass --distance-threshold for this embedding model")

    print(f"Loading embedding model {args.embedding_model} on {device} ...")
    model = SentenceTransformer(
        args.embedding_model, device=device, cache_folder=MODEL_CACHE_PATH
    )

    # Use the same tiny warm-up in every isolated process. Warming up with the
    # target m could leave a target-sized CPU allocator pool in the baseline
    # and hide exactly the incremental memory that this experiment measures.
    warmup_m = 2
    print(
        f"Warm-up: {args.warmup_runs} run(s) at m={warmup_m}, "
        "excluded from measurements"
    )
    for _ in range(args.warmup_runs):
        run_prism_selection(
            usable[0], args.candidate_key, model, warmup_m, threshold
        )

    raw_rows: list[dict] = []
    interval_s = args.poll_interval_ms / 1000.0
    for m in sorted(set(args.m_values)):
        for prompt_index, item in enumerate(usable):
            for repeat in range(1, args.repeats + 1):
                result = measure_once(
                    lambda item=item, m=m: run_prism_selection(
                        item, args.candidate_key, model, m, threshold
                    ),
                    interval_s,
                )
                row = {
                    "m": m,
                    "prompt_index": prompt_index,
                    "source_candidates": item["_memory_scale_source_candidates"],
                    "candidate_expansion": (
                        "repeat"
                        if m > item["_memory_scale_source_candidates"]
                        else "none"
                    ),
                    "repeat": repeat,
                    **result,
                }
                raw_rows.append(row)
                print(
                    f"m={m:4d} prompt={prompt_index + 1}/{len(usable)} "
                    f"repeat={repeat}/{args.repeats}: "
                    f"CPU peak delta={result['cpu_peak_delta_mib']:.2f} MiB, "
                    f"GPU peak delta={result['gpu_peak_allocated_delta_mib']:.2f} MiB, "
                    f"time={result['latency_ms']:.2f} ms"
                )

    summary_rows = summarize(raw_rows)
    output_dir = Path(args.output_dir)
    suffix = f"_m{args.m_values[0]}" if args.isolated_child else ""
    write_csv(output_dir / f"memory_raw{suffix}.csv", raw_rows)
    write_csv(output_dir / f"memory_summary{suffix}.csv", summary_rows)
    metadata = {
        "dataset": str(Path(args.dataset).resolve()),
        "candidate_key": args.candidate_key,
        "candidate_expansion_policy": args.candidate_expansion,
        "source_candidate_counts": source_counts,
        "synthetic_expansion_used": any(count < max_m for count in source_counts),
        "m_values": sorted(set(args.m_values)),
        "num_prompts": len(usable),
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "embedding_model": args.embedding_model,
        "distance_threshold": threshold,
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "isolated_process": args.isolated_child,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"metadata{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("\nMean incremental peak memory across prompts and repeats:")
    for row in summary_rows:
        print(
            f"  m={row['m']:4d}: "
            f"CPU {row['cpu_peak_delta_mib_mean']:.2f} +/- "
            f"{row['cpu_peak_delta_mib_std']:.2f} MiB; "
            f"GPU {row['gpu_peak_allocated_delta_mib_mean']:.2f} +/- "
            f"{row['gpu_peak_allocated_delta_mib_std']:.2f} MiB"
        )
    print(f"Saved results to {output_dir.resolve()}")
    return 0


def child_command(args: argparse.Namespace, m: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset",
        args.dataset,
        "--candidate-key",
        args.candidate_key,
        "--m-values",
        str(m),
        "--candidate-expansion",
        args.candidate_expansion,
        "--num-prompts",
        str(args.num_prompts),
        "--repeats",
        str(args.repeats),
        "--warmup-runs",
        str(args.warmup_runs),
        "--seed",
        str(args.seed),
        "--embedding-model",
        args.embedding_model,
        "--poll-interval-ms",
        str(args.poll_interval_ms),
        "--output-dir",
        args.output_dir,
        "--isolated-child",
    ]
    if args.distance_threshold is not None:
        command.extend(["--distance-threshold", str(args.distance_threshold)])
    return command


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_isolated(args: argparse.Namespace) -> int:
    """Run every candidate size in a fresh process, then merge its results."""
    validate_args(args)
    m_values = sorted(set(args.m_values))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for m in m_values:
        print(
            f"\n{'=' * 18} independent process: m={m} {'=' * 18}",
            flush=True,
        )
        subprocess.run(child_command(args, m), check=True)
        all_rows.extend(read_csv(output_dir / f"memory_raw_m{m}.csv"))

    summary_rows = summarize(all_rows)
    write_csv(output_dir / "memory_raw.csv", all_rows)
    write_csv(output_dir / "memory_summary.csv", summary_rows)
    metadata = {
        "dataset": str(Path(args.dataset).resolve()),
        "candidate_key": args.candidate_key,
        "candidate_expansion_policy": args.candidate_expansion,
        "m_values": m_values,
        "num_prompts": args.num_prompts,
        "repeats": args.repeats,
        "warmup_runs_per_process": args.warmup_runs,
        "embedding_model": args.embedding_model,
        "distance_threshold": args.distance_threshold,
        "independent_process_per_m": True,
        "note": (
            "Each m was measured in a fresh subprocess. Memory/cache from one "
            "candidate size cannot become the baseline of another candidate size."
        ),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("\nIndependent-process mean incremental peak memory:")
    for row in summary_rows:
        print(
            f"  m={int(row['m']):4d}: "
            f"CPU {float(row['cpu_peak_delta_mib_mean']):.2f} +/- "
            f"{float(row['cpu_peak_delta_mib_std']):.2f} MiB; "
            f"GPU {float(row['gpu_peak_allocated_delta_mib_mean']):.2f} +/- "
            f"{float(row['gpu_peak_allocated_delta_mib_std']):.2f} MiB"
        )
    print(f"Merged results saved to {output_dir.resolve()}")
    return 0


def main() -> int:
    args = parse_args()
    if args.isolated_child:
        if len(set(args.m_values)) != 1:
            raise SystemExit("An isolated child must receive exactly one m value")
        return run_benchmark(args)
    return run_isolated(args)


if __name__ == "__main__":
    raise SystemExit(main())
