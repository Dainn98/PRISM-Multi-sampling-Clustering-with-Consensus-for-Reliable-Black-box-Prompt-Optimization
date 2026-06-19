import json
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = BASE_DIR / "evaluation"
OUTPUT_PATH = EVALUATION_DIR / "deepseek_three_run_consistency_winrates.txt"
RUNS = 3


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_consistency_files():
    files = []
    for path in EVALUATION_DIR.glob("*/verify/**/consistency/*_consistency.json"):
        files.append(path)
    for path in EVALUATION_DIR.glob(
        "*/verify_deepseek-chat/generic_rgeneric/consistency/*_consistency.json"
    ):
        files.append(path)
    return sorted(set(files))


def describe_file(path):
    relative = path.relative_to(EVALUATION_DIR)
    parts = relative.parts
    embedding = parts[0]
    verify_folder = parts[1]
    if verify_folder == "verify_deepseek-chat":
        source = "verify_deepseek-chat/generic_rgeneric"
        comparison = "generic_rgeneric"
    else:
        source = "verify"
        consistency_index = parts.index("consistency")
        comparison = parts[consistency_index - 1]
        if comparison == "verify":
            comparison = "legacy_rbpo_bpo"

    suffix = "_deepseek_consistency.json"
    filename = path.name
    experiment = filename[: -len(suffix)] if filename.endswith(suffix) else path.stem
    return embedding, source, comparison, experiment


def calculate_file(path):
    data = load_json(path)
    if data.get("check_runs", 0) < RUNS:
        return None, f"{path}: only {data.get('check_runs', 0)} verification runs"

    excluded_ids = set(data.get("excluded_ids", []))
    items = []
    for item in data.get("consistent_items", []):
        if item.get("id") in excluded_ids:
            continue
        winners = item.get("winners_per_run", [])
        if len(winners) >= RUNS and all(winner in (0, 1, 2) for winner in winners[:RUNS]):
            items.append(winners[:RUNS])

    run_rows = []
    for run_index in range(RUNS):
        counts = Counter(winners[run_index] for winners in items)
        total = len(items)
        a_rate = counts[0] / total if total else 0.0
        b_rate = counts[1] / total if total else 0.0
        tie_rate = counts[2] / total if total else 0.0
        run_rows.append(
            {
                "a_win": counts[0],
                "tie": counts[2],
                "b_win": counts[1],
                "a_win_rate": a_rate,
                "tie_rate": tie_rate,
                "b_win_rate": b_rate,
                "preference_a": a_rate + 0.5 * tie_rate,
                "delta_wr": a_rate - b_rate,
            }
        )

    deltas = [row["delta_wr"] for row in run_rows]
    if all(delta > 0 for delta in deltas):
        trend = "ALL_POSITIVE"
    elif all(delta < 0 for delta in deltas):
        trend = "ALL_NEGATIVE"
    elif all(delta == 0 for delta in deltas):
        trend = "ALL_ZERO"
    else:
        trend = "MIXED"

    if not items:
        return None, f"{path}: no consistent samples with {RUNS} winners"

    embedding, source, comparison, experiment = describe_file(path)
    return {
        "embedding": embedding,
        "source": source,
        "comparison": comparison,
        "experiment": experiment,
        "num_consistent": len(items),
        "reported_consistency_rate": data.get("consistency_rate", "N/A"),
        "runs": run_rows,
        "trend": trend,
    }, None


def percentage(value):
    return f"{value * 100:.2f}%"


def format_report(rows, skipped):
    trend_counts = Counter(row["trend"] for row in rows)
    total_comparisons = len(rows)

    lines = [
        "DEEPSEEK-CHAT THREE-RUN CONSISTENCY WIN-RATE REPORT",
        "=" * 78,
        "",
        "Scope:",
        "- src/evaluation/*/verify/**/consistency",
        "- src/evaluation/*/verify_deepseek-chat/generic_rgeneric/consistency",
        "- Only samples in consistent_items are included.",
        "- Winner coding: 0 = Response A win, 1 = Response B win, 2 = tie.",
        "- DeltaWR = A win rate - B win rate.",
        "- Positive DeltaWR favors A; negative DeltaWR favors B.",
        "",
        "Methodological note:",
        "Repeated DeepSeek-Chat judgments measure intra-referee stability. They do not",
        "remove systematic referee bias and are not a substitute for human evaluation",
        "or cross-validation with a different referee model. Because this report uses",
        "consistent_items, item-level winners are identical across all three runs by",
        "construction; the report quantifies the direction and magnitude of that stable",
        "subset.",
        "",
        "SUMMARY OF THREE-RUN DIRECTIONS",
        "-" * 78,
        f"Comparisons analyzed : {total_comparisons}",
        f"Files skipped        : {len(skipped)}",
    ]

    for trend in ("ALL_POSITIVE", "ALL_NEGATIVE", "ALL_ZERO", "MIXED"):
        count = trend_counts[trend]
        rate = count / total_comparisons if total_comparisons else 0.0
        lines.append(f"{trend:<21}: {count:>3} ({percentage(rate)})")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    lines.extend(["", "DIRECTIONS BY SOURCE FOLDER", "-" * 78])
    for source, source_rows in sorted(grouped.items()):
        source_counts = Counter(row["trend"] for row in source_rows)
        source_total = len(source_rows)
        positive_rate = source_counts["ALL_POSITIVE"] / source_total
        negative_rate = source_counts["ALL_NEGATIVE"] / source_total
        zero_rate = source_counts["ALL_ZERO"] / source_total
        mixed_rate = source_counts["MIXED"] / source_total
        lines.extend(
            [
                f"{source} (n={source_total})",
                f"  all positive: {source_counts['ALL_POSITIVE']:>3} ({percentage(positive_rate)})",
                f"  all negative: {source_counts['ALL_NEGATIVE']:>3} ({percentage(negative_rate)})",
                f"  all zero    : {source_counts['ALL_ZERO']:>3} ({percentage(zero_rate)})",
                f"  mixed       : {source_counts['MIXED']:>3} ({percentage(mixed_rate)})",
            ]
        )

    if skipped:
        lines.extend(["", "SKIPPED FILES", "-" * 78])
        lines.extend(f"- {reason}" for reason in skipped)

    for row in rows:
        lines.extend(
            [
                "",
                "=" * 78,
                f"Embedding  : {row['embedding']}",
                f"Source     : {row['source']}",
                f"Comparison : {row['comparison']} (A vs B)",
                f"Experiment : {row['experiment']}",
                f"Consistent : {row['num_consistent']} samples "
                f"(reported rate: {row['reported_consistency_rate']})",
                f"Trend      : {row['trend']}",
                "",
                "Run | A win       | Tie         | B win       | Pref(A) | DeltaWR",
                "----+-------------+-------------+-------------+---------+---------",
            ]
        )
        for run_index, run in enumerate(row["runs"], start=1):
            lines.append(
                f" {run_index}  | "
                f"{run['a_win']:>4} {percentage(run['a_win_rate']):>7} | "
                f"{run['tie']:>4} {percentage(run['tie_rate']):>7} | "
                f"{run['b_win']:>4} {percentage(run['b_win_rate']):>7} | "
                f"{percentage(run['preference_a']):>7} | "
                f"{run['delta_wr']:+.4f}"
            )

    lines.append("")
    return "\n".join(lines)


def main():
    paths = discover_consistency_files()
    if not paths:
        raise FileNotFoundError("No DeepSeek consistency files were found")
    rows = []
    skipped = []
    for path in paths:
        row, reason = calculate_file(path)
        if row is None:
            skipped.append(reason)
        else:
            rows.append(row)
    report = format_report(rows, skipped)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {len(rows)} comparisons ({len(skipped)} skipped) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
