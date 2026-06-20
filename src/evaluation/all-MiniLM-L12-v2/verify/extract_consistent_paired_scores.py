"""Export item-level paired scores from consistent PRISM comparisons.

Run this file from any working directory. Output CSV files are written to
verify/paired_scores/ and can be opened directly in Excel.
"""

import csv
import json
import statistics
from pathlib import Path

from scipy import stats


VERIFY_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = VERIFY_DIR / "paired_scores_ttest"

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
    "rbpo_bpo": (VERIFY_DIR / "rbpo_bpo" / "consistency", "PRISM(BPO)", "BPO", False),
    "rbpo_mepo": (VERIFY_DIR / "rbpo_mepo" / "consistency", "PRISM(BPO)", "MePO", False),
    "rmepo_mepo": (VERIFY_DIR / "rmepo_mepo" / "consistency", "PRISM(MePO)", "MePO", False),
    # Source order is Generic (response_A) vs RGeneric (response_B). Swap it in
    # the export so method_A is consistently the PRISM method.
    "generic_rgeneric": (
        VERIFY_DIR.parent / "verify_deepseek-chat" / "generic_rgeneric" / "consistency",
        "PRISM(Generic)",
        "Generic",
        True,
    ),
}


def experiment_name(path: Path) -> str:
    suffix = "_deepseek_consistency"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def split_experiment(name: str) -> tuple[str, str]:
    parts = name.split("_", 1)
    return (parts[0], parts[1] if len(parts) == 2 else "")


def mean_score(scores: dict, path: Path, item_id: object, side: str) -> float:
    missing = [criterion for criterion in CRITERIA if criterion not in scores]
    if missing:
        raise ValueError(f"{path}: item {item_id}, {side} missing {missing}")
    return statistics.fmean(float(scores[criterion]) for criterion in CRITERIA)


def extract_file(
    path: Path, comparison: str, method_a: str, method_b: str, swap_responses: bool
) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    excluded_ids = set(data.get("excluded_ids", []))
    experiment = experiment_name(path)
    target_model, dataset = split_experiment(experiment)
    rows = []

    for item in data.get("consistent_items", []):
        item_id = item.get("id")
        if item_id in excluded_ids:
            continue

        evaluations = item.get("llm_evaluations_per_run", [])
        if not evaluations:
            raise ValueError(f"{path}: item {item_id} has no judge evaluation")

        # Consistent items have equivalent judge runs, so use judge run 1 only.
        first_evaluation = evaluations[0]
        scores_a = first_evaluation["response_A"]
        scores_b = first_evaluation["response_B"]
        if swap_responses:
            scores_a, scores_b = scores_b, scores_a
        score_a = mean_score(scores_a, path, item_id, "response_A")
        score_b = mean_score(scores_b, path, item_id, "response_B")

        row = {
            "embedding": VERIFY_DIR.parent.name,
            "comparison": comparison,
            "experiment": experiment,
            "target_model": target_model,
            "dataset": dataset,
            "item_id": item_id,
            "judge_run": 1,
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

    for comparison, (consistency_dir, method_a, method_b, swap_responses) in COMPARISONS.items():
        comparison_rows = []
        for path in sorted(consistency_dir.glob("*_consistency.json")):
            comparison_rows.extend(
                extract_file(path, comparison, method_a, method_b, swap_responses)
            )

        write_csv(OUTPUT_DIR / f"{comparison}_paired_scores.csv", comparison_rows)
        if comparison_rows:
            overall = overall_summary_row(comparison_rows)
            write_csv(OUTPUT_DIR / f"{comparison}_overall_ttest.csv", [overall])
            overall_rows.append(overall)
        all_rows.extend(comparison_rows)
        print(f"{comparison}: exported {len(comparison_rows)} paired items")

    write_csv(OUTPUT_DIR / "all_paired_scores.csv", all_rows)
    write_csv(OUTPUT_DIR / "summary_ttest.csv", summary_rows(all_rows))
    write_csv(OUTPUT_DIR / "overall_summary_ttest.csv", overall_rows)
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
