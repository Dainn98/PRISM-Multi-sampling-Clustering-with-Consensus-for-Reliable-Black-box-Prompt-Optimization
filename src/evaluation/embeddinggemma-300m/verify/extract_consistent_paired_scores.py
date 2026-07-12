"""Export final item-level scores from consistency and resolved mismatches.

Run this file from any working directory. Output CSV files are written to
verify/paired_scores_final_ttest/ and can be opened directly in Excel.
"""

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from scipy import stats


VERIFY_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = VERIFY_DIR / "paired_scores_final_ttest"

CRITERIA = (
    "Correctness",
    "Relevance",
    "Completeness",
    "Clarity_Coherence",
    "Usefulness_Helpfulness",
    "Style_Tone",
    "Conciseness",
    "Safety_Compliance",
)

COMPARISONS = {
    "rbpo_bpo": (VERIFY_DIR / "rbpo_bpo", "PRISM(BPO)", "BPO", False),
    "rbpo_mepo": (VERIFY_DIR / "rbpo_mepo", "PRISM(BPO)", "MePO", False),
    "rmepo_mepo": (VERIFY_DIR / "rmepo_mepo", "PRISM(MePO)", "MePO", False),
    # Source order is Generic (response_A) vs RGeneric (response_B). Swap it in
    # the export so method_A is consistently the PRISM method.
    "generic_rgeneric": (
        VERIFY_DIR.parent / "verify_deepseek-chat" / "generic_rgeneric",
        "PRISM(Generic)",
        "Generic",
        True,
    ),
}


def experiment_name(path: Path) -> str:
    for suffix in ("_deepseek_consistency", "_deepseek_mismatch", "_deepseek_final"):
        if path.stem.endswith(suffix):
            return path.stem[: -len(suffix)]
    return path.stem


def split_experiment(name: str) -> tuple[str, str]:
    parts = name.split("_", 1)
    return (parts[0], parts[1] if len(parts) == 2 else "")


def mean_score(scores: dict, path: Path, item_id: object, side: str) -> float:
    missing = [criterion for criterion in CRITERIA if criterion not in scores]
    if missing:
        raise ValueError(f"{path}: item {item_id}, {side} missing {missing}")
    return statistics.fmean(float(scores[criterion]) for criterion in CRITERIA)


def majority_winner(winners: list[int]) -> int:
    counts = Counter(winners)
    highest = max(counts.values())
    leaders = [winner for winner, count in counts.items() if count == highest]
    return leaders[0] if len(leaders) == 1 else 2


def final_winner_map(case_dir: Path, mismatch_path: Path) -> dict:
    experiment = experiment_name(mismatch_path)
    final_path = case_dir / "final" / f"{experiment}_deepseek_final.json"
    if not final_path.exists():
        return {}
    with final_path.open("r", encoding="utf-8") as file:
        final_data = json.load(file)
    return {
        item["id"]: int(item["final_winner"])
        for item in final_data.get("mismatches", [])
    }


def export_winner(winner: int, swap_responses: bool) -> int:
    if swap_responses and winner in (0, 1):
        return 1 - winner
    return winner


def extract_file(
    path: Path,
    case_dir: Path,
    comparison: str,
    method_a: str,
    method_b: str,
    swap_responses: bool,
    source_split: str,
) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    excluded_ids = set(data.get("excluded_ids", []))
    experiment = experiment_name(path)
    target_model, dataset = split_experiment(experiment)
    rows = []

    item_key = "consistent_items" if source_split == "consistency" else "mismatches"
    resolved_winners = final_winner_map(case_dir, path) if source_split == "mismatch" else {}

    for item in data.get(item_key, []):
        item_id = item.get("id")
        if item_id in excluded_ids:
            continue

        evaluations = item.get("llm_evaluations_per_run", [])
        if not evaluations:
            raise ValueError(f"{path}: item {item_id} has no judge evaluation")

        # One run is sufficient for consistent items. Mismatch items use the
        # mean over all judge runs and the separately resolved final winner.
        selected_evaluations = evaluations[:1] if source_split == "consistency" else evaluations
        scores_a = {
            criterion: statistics.fmean(
                float(evaluation["response_A"][criterion])
                for evaluation in selected_evaluations
            )
            for criterion in CRITERIA
        }
        scores_b = {
            criterion: statistics.fmean(
                float(evaluation["response_B"][criterion])
                for evaluation in selected_evaluations
            )
            for criterion in CRITERIA
        }
        if swap_responses:
            scores_a, scores_b = scores_b, scores_a
        score_a = mean_score(scores_a, path, item_id, "response_A")
        score_b = mean_score(scores_b, path, item_id, "response_B")

        raw_winners = [int(winner) for winner in item.get("winners_per_run", [])]
        if source_split == "consistency":
            raw_final_winner = raw_winners[0] if raw_winners else 2
            winner_source = "consistent_judges"
        elif item_id in resolved_winners:
            raw_final_winner = resolved_winners[item_id]
            winner_source = "final_file"
        else:
            raw_final_winner = majority_winner(raw_winners)
            winner_source = "derived_majority_vote"
        final_winner = export_winner(raw_final_winner, swap_responses)

        row = {
            "embedding": VERIFY_DIR.parent.name,
            "comparison": comparison,
            "experiment": experiment,
            "target_model": target_model,
            "dataset": dataset,
            "item_id": item_id,
            "source_split": source_split,
            "score_aggregation": (
                "first_judge_run" if source_split == "consistency" else "mean_all_judge_runs"
            ),
            "judge_runs_used": len(selected_evaluations),
            "final_winner": final_winner,
            "final_winner_label": (
                method_a if final_winner == 0 else method_b if final_winner == 1 else "Tie"
            ),
            "final_winner_source": winner_source,
            "method_A": method_a,
            "method_B": method_b,
        }
        for criterion in CRITERIA:
            row[f"A_{criterion}"] = float(scores_a[criterion])
            row[f"B_{criterion}"] = float(scores_b[criterion])
        row["method_A_mean"] = score_a
        row["method_B_mean"] = score_b
        row["difference_A_minus_B"] = score_a - score_b
        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    try:
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        print(f"Skipped locked output file: {path}")


def summary_rows(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["comparison"], row["experiment"])
        groups.setdefault(key, []).append(row)

    summaries = []
    for (comparison, experiment), group in sorted(groups.items()):
        scores_a = [row["method_A_mean"] for row in group]
        scores_b = [row["method_B_mean"] for row in group]
        differences = [row["difference_A_minus_B"] for row in group]
        n = len(group)
        mean_difference = statistics.fmean(differences)
        sd_difference = statistics.stdev(differences) if n > 1 else 0.0
        if n > 1 and sd_difference > 0:
            test = stats.ttest_rel(scores_a, scores_b)
            standard_error = sd_difference / (n**0.5)
            margin = stats.t.ppf(0.975, n - 1) * standard_error
            t_statistic = float(test.statistic)
            p_value = float(test.pvalue)
            ci_lower = mean_difference - margin
            ci_upper = mean_difference + margin
            cohen_dz = mean_difference / sd_difference
        else:
            t_statistic = 0.0 if mean_difference == 0 else None
            p_value = 1.0 if mean_difference == 0 else None
            ci_lower = mean_difference
            ci_upper = mean_difference
            cohen_dz = 0.0 if mean_difference == 0 else None

        summaries.append(
            {
                "embedding": group[0]["embedding"],
                "comparison": comparison,
                "experiment": experiment,
                "method_A": group[0]["method_A"],
                "method_B": group[0]["method_B"],
                "N": n,
                "mean_A": statistics.fmean(scores_a),
                "sd_A": statistics.stdev(scores_a) if len(group) > 1 else 0.0,
                "mean_B": statistics.fmean(scores_b),
                "sd_B": statistics.stdev(scores_b) if len(group) > 1 else 0.0,
                "mean_difference_A_minus_B": mean_difference,
                "sd_difference": sd_difference,
                "t_statistic": t_statistic,
                "degrees_of_freedom": n - 1,
                "p_value_two_sided": p_value,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "cohen_dz": cohen_dz,
            }
        )
    return summaries


def overall_summary_row(rows: list[dict]) -> dict:
    """Calculate one pooled paired t-test across all datasets and base LLMs."""
    scores_a = [row["method_A_mean"] for row in rows]
    scores_b = [row["method_B_mean"] for row in rows]
    differences = [row["difference_A_minus_B"] for row in rows]
    n = len(rows)
    mean_difference = statistics.fmean(differences)
    sd_difference = statistics.stdev(differences) if n > 1 else 0.0

    if n > 1 and sd_difference > 0:
        test = stats.ttest_rel(scores_a, scores_b)
        standard_error = sd_difference / (n**0.5)
        margin = stats.t.ppf(0.975, n - 1) * standard_error
        t_statistic = float(test.statistic)
        p_value = float(test.pvalue)
        ci_lower = mean_difference - margin
        ci_upper = mean_difference + margin
        cohen_dz = mean_difference / sd_difference
    else:
        t_statistic = 0.0 if mean_difference == 0 else None
        p_value = 1.0 if mean_difference == 0 else None
        ci_lower = mean_difference
        ci_upper = mean_difference
        cohen_dz = 0.0 if mean_difference == 0 else None

    return {
        "embedding": rows[0]["embedding"],
        "comparison": rows[0]["comparison"],
        "scope": "all_datasets_and_base_llms",
        "method_A": rows[0]["method_A"],
        "method_B": rows[0]["method_B"],
        "N": n,
        "mean_A": statistics.fmean(scores_a),
        "sd_A": statistics.stdev(scores_a) if n > 1 else 0.0,
        "mean_B": statistics.fmean(scores_b),
        "sd_B": statistics.stdev(scores_b) if n > 1 else 0.0,
        "mean_difference_A_minus_B": mean_difference,
        "sd_difference": sd_difference,
        "t_statistic": t_statistic,
        "degrees_of_freedom": n - 1,
        "p_value_two_sided": p_value,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "cohen_dz": cohen_dz,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    overall_rows = []

    for comparison, (case_dir, method_a, method_b, swap_responses) in COMPARISONS.items():
        comparison_rows = []
        consistent_count = 0
        mismatch_count = 0
        for path in sorted((case_dir / "consistency").glob("*_consistency.json")):
            rows = extract_file(
                path, case_dir, comparison, method_a, method_b, swap_responses, "consistency"
            )
            consistent_count += len(rows)
            comparison_rows.extend(rows)
        for path in sorted((case_dir / "mismatch").glob("*_mismatch.json")):
            rows = extract_file(
                path, case_dir, comparison, method_a, method_b, swap_responses, "mismatch"
            )
            mismatch_count += len(rows)
            comparison_rows.extend(rows)

        keys = [(row["experiment"], row["item_id"]) for row in comparison_rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{comparison}: duplicate experiment/item_id after final merge")

        write_csv(OUTPUT_DIR / f"{comparison}_paired_scores.csv", comparison_rows)
        if comparison_rows:
            overall = overall_summary_row(comparison_rows)
            write_csv(OUTPUT_DIR / f"{comparison}_overall_ttest.csv", [overall])
            overall_rows.append(overall)
        all_rows.extend(comparison_rows)
        print(
            f"{comparison}: {consistent_count} consistency + {mismatch_count} mismatch "
            f"= {len(comparison_rows)} final paired items"
        )

    write_csv(OUTPUT_DIR / "all_paired_scores.csv", all_rows)
    write_csv(OUTPUT_DIR / "summary_ttest.csv", summary_rows(all_rows))
    write_csv(OUTPUT_DIR / "overall_summary_ttest.csv", overall_rows)
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
