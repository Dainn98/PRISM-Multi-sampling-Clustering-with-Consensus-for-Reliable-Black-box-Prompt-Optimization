"""Calculate robustness, confidence intervals, and significance statistics.

Accepted input formats:

1. Long-form CSV with columns:
   comparison,prompt_id,judge_run,result[,seed]
2. One or more repository judge JSON files named ``*_eval_N.json``. For JSON
   input, pass --verify-key so ``<verify_key>_winner`` can be located.

Winner values in repository JSON files are interpreted as 0=Method A win,
1=Method A loss, and 2=tie.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


VALID_RESULTS = {"win", "tie", "loss"}
PREF_VALUE = {"win": 1.0, "tie": 0.5, "loss": 0.0}
DELTA_VALUE = {"win": 1.0, "tie": 0.0, "loss": -1.0}
WINNER_VALUE = {0: "win", 1: "loss", 2: "tie", "0": "win", "1": "loss", "2": "tie"}
DEFAULT_VERIFY_JOBS = {
    "rbpo_bpo": "PRISM_BPO_vs_BPO",
    "rbpo_mepo": "PRISM_BPO_vs_MePO",
    "rmepo_mepo": "PRISM_MePO_vs_MePO",
}
DEFAULT_VERIFY_ROOT = Path("src/evaluation/embeddinggemma-300m/verify")


@dataclass(frozen=True)
class Record:
    comparison: str
    prompt_id: str
    judge_run: int
    result: str
    seed: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", help="CSV/JSON input files or glob patterns")
    parser.add_argument("--output-dir", default="src/extra_results/robust_all")
    parser.add_argument("--verify-key", help="Key prefix used by repository JSON judge files")
    parser.add_argument("--comparison", help="Comparison name for JSON input; defaults to --verify-key")
    parser.add_argument("--expected-runs", type=int, help="Expected judge runs per prompt")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--embedding-label", default="embeddinggemma-300m")
    parser.add_argument(
        "--benchmark-mode",
        choices=["aggregate-complete", "separate"],
        default="aggregate-complete",
        help=(
            "aggregate-complete combines the four instruction-following benchmarks "
            "into one config and keeps only configs with all required benchmarks."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["vicuna", "dolly", "bpo", "self"],
        help="Benchmark labels required when --benchmark-mode=aggregate-complete.",
    )
    return parser.parse_args()


def default_jobs() -> list[tuple[str, str, str, list[str]]]:
    jobs = []
    for verify_key, comparison in DEFAULT_VERIFY_JOBS.items():
        verify_dir = DEFAULT_VERIFY_ROOT / verify_key
        pattern = str(verify_dir / "*_eval_*.json")
        if list(Path().glob(pattern)):
            jobs.append((verify_key, comparison, verify_key, [pattern]))
    return jobs


def expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.exists():
            paths.append(candidate)
            continue
        matches = sorted(Path().glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No files matched: {pattern}")
        paths.extend(matches)
    return list(dict.fromkeys(path.resolve() for path in paths))


def read_csv_records(path: Path) -> list[Record]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"comparison", "prompt_id", "judge_run", "result"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing CSV columns: {sorted(missing)}")
        records = []
        for line_no, row in enumerate(reader, start=2):
            result = (row["result"] or "").strip().lower()
            if result not in VALID_RESULTS:
                raise ValueError(f"{path}:{line_no}: invalid result {result!r}")
            records.append(
                Record(
                    comparison=(row["comparison"] or "").strip(),
                    prompt_id=(row["prompt_id"] or "").strip(),
                    judge_run=int(row["judge_run"]),
                    result=result,
                    seed=(row.get("seed") or "").strip(),
                )
            )
    return records


def infer_run_number(path: Path) -> int:
    match = re.search(r"_eval_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Cannot infer judge run from JSON filename: {path.name}")
    return int(match.group(1))


def infer_experiment_name(path: Path) -> str:
    match = re.match(r"(.+)_eval_\d+\.json$", path.name)
    if not match:
        raise ValueError(f"Cannot infer experiment name from JSON filename: {path.name}")
    return match.group(1)


def json_comparison_name(
    path: Path,
    verify_key: str,
    comparison: str | None,
    split_by_experiment: bool,
) -> str:
    experiment = infer_experiment_name(path)
    if comparison and "{experiment}" in comparison:
        return comparison.format(experiment=experiment, verify_key=verify_key)
    if split_by_experiment:
        prefix = comparison or verify_key
        return f"{prefix}_{experiment}"
    return comparison or verify_key


def read_json_records(
    path: Path,
    verify_key: str,
    comparison: str,
) -> list[Record]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")

    judge_run = infer_run_number(path)
    winner_key = f"{verify_key}_winner"
    records = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: item {index} is not an object")
        if winner_key not in item:
            raise ValueError(f"{path}: item {index} missing {winner_key!r}")
        winner = item[winner_key]
        if winner not in WINNER_VALUE:
            raise ValueError(f"{path}: item {index} has invalid winner {winner!r}")
        prompt_id = item.get("id", index)
        records.append(
            Record(
                comparison=comparison,
                prompt_id=str(prompt_id),
                judge_run=judge_run,
                result=WINNER_VALUE[winner],
            )
        )
    return records


def load_records(paths: Sequence[Path], verify_key: str | None, comparison: str | None) -> list[Record]:
    records: list[Record] = []
    json_experiments = {
        infer_experiment_name(path)
        for path in paths
        if path.suffix.lower() == ".json"
    }
    split_by_experiment = len(json_experiments) > 1
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            records.extend(read_csv_records(path))
        elif suffix == ".json":
            if not verify_key:
                raise ValueError("--verify-key is required for JSON judge files")
            records.extend(
                read_json_records(
                    path,
                    verify_key,
                    json_comparison_name(
                        path,
                        verify_key,
                        comparison,
                        split_by_experiment,
                    ),
                )
            )
        else:
            raise ValueError(f"Unsupported input format: {path}")
    return records


def validate_records(records: Sequence[Record], expected_runs: int | None) -> list[str]:
    if not records:
        raise ValueError("Input contains no records")

    duplicates = [
        key
        for key, count in Counter(
            (r.comparison, r.seed, r.prompt_id, r.judge_run) for r in records
        ).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate comparison/seed/prompt/run records: {duplicates[:5]}")

    warnings: list[str] = []
    grouped: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for record in records:
        if not record.comparison or not record.prompt_id:
            raise ValueError("comparison and prompt_id must be non-empty")
        grouped[(record.comparison, record.seed, record.prompt_id)].add(record.judge_run)

    for key, runs in grouped.items():
        target = expected_runs or max(runs)
        expected = set(range(1, target + 1))
        if runs != expected:
            warnings.append(f"{key}: judge runs={sorted(runs)}, expected={sorted(expected)}")
    return warnings


def metric_row(records: Sequence[Record]) -> dict[str, float | int]:
    counts = Counter(record.result for record in records)
    total = sum(counts.values())
    win_ratio = counts["win"] / total
    tie_ratio = counts["tie"] / total
    loss_ratio = counts["loss"] / total
    return {
        "num_win": counts["win"],
        "num_tie": counts["tie"],
        "num_loss": counts["loss"],
        "win_ratio": win_ratio,
        "tie_ratio": tie_ratio,
        "loss_ratio": loss_ratio,
        "pref": win_ratio + 0.5 * tie_ratio,
        "delta_wr": win_ratio - loss_ratio,
    }


def sample_std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def bootstrap_ci(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    chunk_size = min(iterations, 2_000)
    means: list[np.ndarray] = []
    remaining = iterations
    while remaining:
        current = min(chunk_size, remaining)
        indices = rng.integers(0, len(values), size=(current, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= current
    samples = np.concatenate(means)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def sign_flip_p_value(values: np.ndarray, iterations: int, rng: np.random.Generator) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    observed = abs(float(values.mean()))
    extreme = 0
    remaining = iterations
    while remaining:
        current = min(remaining, 5_000)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(current, len(values)))
        permuted = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
        remaining -= current
    return (extreme + 1) / (iterations + 1)


def mcnemar_exact(results_by_prompt: dict[str, list[str]]) -> tuple[int, int, float]:
    majority = {}
    for prompt_id, results in results_by_prompt.items():
        counts = Counter(results)
        top = counts.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            continue
        majority[prompt_id] = top[0][0]
    wins = sum(result == "win" for result in majority.values())
    losses = sum(result == "loss" for result in majority.values())
    n = wins + losses
    if n == 0:
        return wins, losses, math.nan
    k = min(wins, losses)
    lower_tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return wins, losses, min(1.0, 2.0 * lower_tail)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    valid = [(index, value) for index, value in enumerate(p_values) if math.isfinite(value)]
    ordered = sorted(valid, key=lambda pair: pair[1])
    running = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    return adjusted


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    records: Sequence[Record],
    bootstrap_iterations: int,
    permutation_iterations: int,
    random_seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    run_groups: dict[tuple[str, str, int], list[Record]] = defaultdict(list)
    prompt_groups: dict[tuple[str, str, str], list[Record]] = defaultdict(list)
    for record in records:
        run_groups[(record.comparison, record.seed, record.judge_run)].append(record)
        prompt_groups[(record.comparison, record.seed, record.prompt_id)].append(record)

    run_rows = []
    for (comparison, seed, judge_run), group in sorted(run_groups.items()):
        run_rows.append(
            {"comparison": comparison, "seed": seed, "judge_run": judge_run, **metric_row(group)}
        )

    prompt_rows = []
    for (comparison, seed, prompt_id), group in sorted(prompt_groups.items()):
        counts = Counter(record.result for record in group)
        prompt_rows.append(
            {
                "comparison": comparison,
                "seed": seed,
                "prompt_id": prompt_id,
                "pref_contribution": float(np.mean([PREF_VALUE[r.result] for r in group])),
                "delta_wr_contribution": float(np.mean([DELTA_VALUE[r.result] for r in group])),
                "num_win": counts["win"],
                "num_tie": counts["tie"],
                "num_loss": counts["loss"],
                "unstable": int(len(counts) > 1),
            }
        )

    rng = np.random.default_rng(random_seed)
    summaries = []
    comparisons = sorted({record.comparison for record in records})
    for comparison in comparisons:
        comparison_runs = [row for row in run_rows if row["comparison"] == comparison]
        comparison_prompts = [row for row in prompt_rows if row["comparison"] == comparison]

        # Average repeated seed/judge observations within each prompt before bootstrap.
        per_prompt: dict[str, list[dict]] = defaultdict(list)
        for row in comparison_prompts:
            per_prompt[str(row["prompt_id"])].append(row)
        prompt_pref = np.array(
            [np.mean([row["pref_contribution"] for row in rows]) for rows in per_prompt.values()]
        )
        prompt_delta = np.array(
            [np.mean([row["delta_wr_contribution"] for row in rows]) for rows in per_prompt.values()]
        )

        pref_ci = bootstrap_ci(prompt_pref, bootstrap_iterations, rng)
        delta_ci = bootstrap_ci(prompt_delta, bootstrap_iterations, rng)
        p_value = sign_flip_p_value(prompt_delta, permutation_iterations, rng)

        judge_pref_by_run: dict[int, list[float]] = defaultdict(list)
        judge_delta_by_run: dict[int, list[float]] = defaultdict(list)
        seed_pref: dict[str, list[float]] = defaultdict(list)
        seed_delta: dict[str, list[float]] = defaultdict(list)
        for row in comparison_runs:
            judge_pref_by_run[int(row["judge_run"])].append(float(row["pref"]))
            judge_delta_by_run[int(row["judge_run"])].append(float(row["delta_wr"]))
            if row["seed"]:
                seed_pref[str(row["seed"])].append(float(row["pref"]))
                seed_delta[str(row["seed"])].append(float(row["delta_wr"]))

        results_by_prompt = {
            prompt_id: [
                record.result
                for record in records
                if record.comparison == comparison and record.prompt_id == prompt_id
            ]
            for prompt_id in per_prompt
        }
        majority_wins, majority_losses, mcnemar_p = mcnemar_exact(results_by_prompt)
        unstable_count = sum(
            len(set(results)) > 1 for results in results_by_prompt.values()
        )
        run_pref_values = [float(row["pref"]) for row in comparison_runs]
        run_delta_values = [float(row["delta_wr"]) for row in comparison_runs]
        seed_pref_values = [float(np.mean(values)) for values in seed_pref.values()]
        seed_delta_values = [float(np.mean(values)) for values in seed_delta.values()]

        summaries.append(
            {
                "comparison": comparison,
                "num_prompts": len(per_prompt),
                "num_optimizer_seeds": len(seed_pref),
                "num_judge_runs": len(judge_pref_by_run),
                "win_ratio_mean": float(np.mean([row["win_ratio"] for row in comparison_runs])),
                "tie_ratio_mean": float(np.mean([row["tie_ratio"] for row in comparison_runs])),
                "loss_ratio_mean": float(np.mean([row["loss_ratio"] for row in comparison_runs])),
                "pref_mean": float(prompt_pref.mean()),
                "pref_judge_std": sample_std(
                    [float(np.mean(values)) for values in judge_pref_by_run.values()]
                ),
                "pref_seed_std": sample_std(seed_pref_values),
                "pref_all_run_std": sample_std(run_pref_values),
                "pref_ci_lower": pref_ci[0],
                "pref_ci_upper": pref_ci[1],
                "delta_wr_mean": float(prompt_delta.mean()),
                "delta_wr_judge_std": sample_std(
                    [float(np.mean(values)) for values in judge_delta_by_run.values()]
                ),
                "delta_wr_seed_std": sample_std(seed_delta_values),
                "delta_wr_all_run_std": sample_std(run_delta_values),
                "delta_wr_ci_lower": delta_ci[0],
                "delta_wr_ci_upper": delta_ci[1],
                "permutation_p_value": p_value,
                "mcnemar_exact_p_value": mcnemar_p,
                "majority_win_count": majority_wins,
                "majority_loss_count": majority_losses,
                "method_loss_rate": majority_losses / len(per_prompt) if per_prompt else math.nan,
                "unstable_case_rate": unstable_count / len(per_prompt) if per_prompt else math.nan,
            }
        )

    adjusted = holm_adjust([float(row["permutation_p_value"]) for row in summaries])
    for row, value in zip(summaries, adjusted):
        row["permutation_p_value_holm"] = value
    return run_rows, prompt_rows, summaries


def print_markdown(summary_rows: Sequence[dict]) -> None:
    print("| Comparison | Pref. mean +/- judge std | Pref. 95% CI | DeltaWR mean +/- judge std | DeltaWR 95% CI | Holm p |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        print(
            f"| {row['comparison']} "
            f"| {row['pref_mean']:.3f} +/- {row['pref_judge_std']:.3f} "
            f"| [{row['pref_ci_lower']:.3f}, {row['pref_ci_upper']:.3f}] "
            f"| {row['delta_wr_mean']:.3f} +/- {row['delta_wr_judge_std']:.3f} "
            f"| [{row['delta_wr_ci_lower']:.3f}, {row['delta_wr_ci_upper']:.3f}] "
            f"| {row['permutation_p_value_holm']:.4g} |"
        )


def failure_case_rows(prompt_rows: Sequence[dict]) -> list[dict]:
    rows = []
    for row in prompt_rows:
        counts = {
            "win": int(row["num_win"]),
            "tie": int(row["num_tie"]),
            "loss": int(row["num_loss"]),
        }
        majority_loss = counts["loss"] > max(counts["win"], counts["tie"])
        if majority_loss or int(row["unstable"]):
            rows.append(
                {
                    **row,
                    "majority_loss": int(majority_loss),
                    "failure_reason": ",".join(
                        reason
                        for reason, active in (
                            ("majority_loss", majority_loss),
                            ("judge_instability", bool(row["unstable"])),
                        )
                        if active
                    ),
                }
            )
    return rows


def parse_summary_comparison(comparison: str) -> dict[str, str] | None:
    parts = comparison.split("_")
    if len(parts) < 4:
        return None
    evaluator = parts[-1]
    benchmark = parts[-2]
    base_llm = parts[-3]
    method_key = "_".join(parts[:-3])
    method_map = {
        "PRISM_BPO_vs_BPO": ("PRISM + BPO", "BPO"),
        "PRISM_BPO_vs_MePO": ("PRISM + BPO", "MePO"),
        "PRISM_MePO_vs_MePO": ("PRISM + MePO", "MePO"),
        "PRISM_MEPO_vs_MEPO": ("PRISM + MePO", "MePO"),
        "PRISM_Generic_vs_Generic": ("PRISM + Generic", "Generic"),
    }
    method_a, method_b = method_map.get(
        method_key,
        (method_key.replace("_vs_", " vs "), ""),
    )
    return {
        "method_key": method_key,
        "method_a": method_a,
        "method_b": method_b,
        "base_llm": base_llm,
        "benchmark": benchmark,
        "evaluator": evaluator,
    }


def aggregate_complete_benchmark_records(
    records: Sequence[Record],
    required_benchmarks: Sequence[str],
) -> list[Record]:
    required = set(required_benchmarks)
    groups: dict[tuple[str, str, str], list[tuple[Record, dict[str, str]]]] = defaultdict(list)

    for record in records:
        parsed = parse_summary_comparison(record.comparison)
        if parsed is None:
            continue
        key = (parsed["method_key"], parsed["base_llm"], parsed["evaluator"])
        groups[key].append((record, parsed))

    aggregated: list[Record] = []
    for (method_key, base_llm, evaluator), group in groups.items():
        available = {parsed["benchmark"] for _, parsed in group}
        if not required.issubset(available):
            continue

        comparison = f"{method_key}_{base_llm}_all_{evaluator}"
        for record, parsed in group:
            if parsed["benchmark"] not in required:
                continue
            aggregated.append(
                Record(
                    comparison=comparison,
                    prompt_id=f"{parsed['benchmark']}:{record.prompt_id}",
                    judge_run=record.judge_run,
                    result=record.result,
                    seed=record.seed,
                )
            )

    if not aggregated:
        expected = ",".join(required_benchmarks)
        raise ValueError(f"No configs contain all required benchmarks: {expected}")
    return aggregated


def benchmark_aggregate_rows(
    summary_rows: Sequence[dict],
    embedding_label: str,
) -> list[dict]:
    groups: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    benchmark_order = ["vicuna", "dolly", "bpo", "self"]
    for row in summary_rows:
        parsed = parse_summary_comparison(str(row["comparison"]))
        if parsed is None:
            continue
        if parsed["benchmark"] not in benchmark_order:
            continue
        key = (
            embedding_label,
            parsed["base_llm"],
            parsed["method_a"],
            parsed["method_b"],
            parsed["evaluator"],
        )
        groups[key].append({**row, **parsed})

    output_rows = []
    for (embedding, base_llm, method_a, method_b, evaluator), rows in sorted(groups.items()):
        by_benchmark = {str(row["benchmark"]): row for row in rows}
        pref_values = [
            float(by_benchmark[name]["pref_mean"])
            for name in benchmark_order
            if name in by_benchmark
        ]
        wr_values = [
            float(by_benchmark[name]["delta_wr_mean"])
            for name in benchmark_order
            if name in by_benchmark
        ]
        output = {
            "embedding": embedding,
            "base_llm": base_llm,
            "method_a": method_a,
            "method_b": method_b,
            "evaluator": evaluator,
            "num_benchmarks": len(pref_values),
            "pref_mean_across_benchmarks": float(np.mean(pref_values)) if pref_values else math.nan,
            "pref_std_across_benchmarks": sample_std(pref_values),
            "pref_mean_std": (
                f"{float(np.mean(pref_values)):.3f} +/- {sample_std(pref_values):.3f}"
                if pref_values
                else ""
            ),
            "wr_mean_across_benchmarks": float(np.mean(wr_values)) if wr_values else math.nan,
            "wr_std_across_benchmarks": sample_std(wr_values),
            "wr_mean_std": (
                f"{float(np.mean(wr_values)):.3f} +/- {sample_std(wr_values):.3f}"
                if wr_values
                else ""
            ),
        }
        for benchmark in benchmark_order:
            row = by_benchmark.get(benchmark)
            prefix = f"{benchmark}_eval"
            output[f"{prefix}_a_win"] = float(row["win_ratio_mean"]) if row else ""
            output[f"{prefix}_tie"] = float(row["tie_ratio_mean"]) if row else ""
            output[f"{prefix}_b_win"] = float(row["loss_ratio_mean"]) if row else ""
            output[f"{prefix}_pref"] = float(row["pref_mean"]) if row else ""
            output[f"{prefix}_wr"] = float(row["delta_wr_mean"]) if row else ""
        output_rows.append(output)
    return output_rows


def main() -> int:
    args = parse_args()
    try:
        if args.input:
            jobs = [
                (
                    args.verify_key,
                    args.comparison,
                    "custom",
                    args.input,
                )
            ]
        else:
            jobs = default_jobs()
            if not jobs:
                raise FileNotFoundError(
                    f"No default judge files found under {DEFAULT_VERIFY_ROOT}. "
                    "Pass --input explicitly."
                )

        all_records = []
        for verify_key, comparison, _, patterns in jobs:
            if not verify_key:
                raise ValueError("--verify-key is required for JSON judge files")
            paths = expand_inputs(patterns)
            all_records.extend(load_records(paths, verify_key, comparison))

        records = (
            aggregate_complete_benchmark_records(all_records, args.benchmarks)
            if args.benchmark_mode == "aggregate-complete"
            else all_records
        )
        warnings = validate_records(records, args.expected_runs)
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)

        run_rows, prompt_rows, summaries = analyze(
            records,
            bootstrap_iterations=args.bootstrap_iterations,
            permutation_iterations=args.permutation_iterations,
            random_seed=args.random_seed,
        )
        output_dir = Path(args.output_dir)
        write_csv(output_dir / "judge_run_metrics.csv", run_rows)
        write_csv(output_dir / "prompt_level_metrics.csv", prompt_rows)
        write_csv(output_dir / "statistical_summary.csv", summaries)
        write_csv(
            output_dir / "benchmark_aggregate_summary.csv",
            benchmark_aggregate_rows(summaries, args.embedding_label),
        )
        write_csv(
            output_dir / "failure_cases.csv",
            failure_case_rows(prompt_rows),
            fieldnames=[
                "comparison",
                "seed",
                "prompt_id",
                "pref_contribution",
                "delta_wr_contribution",
                "num_win",
                "num_tie",
                "num_loss",
                "unstable",
                "majority_loss",
                "failure_reason",
            ],
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "validation_warnings.txt").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(warnings))
        print_markdown(summaries)
        print(f"\nSaved results to {output_dir.resolve()}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
