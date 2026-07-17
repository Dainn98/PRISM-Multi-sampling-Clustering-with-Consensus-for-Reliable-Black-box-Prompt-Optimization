import json
import os
import re
from collections import Counter, defaultdict

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from helper import (
    DEEPSEEK,
    base_llm_models,
    clean_name,
    create_combined_name,
    evaluation_datasets,
    evaluator_models,
)
from msd_config import (
    BASE_DIR,
    MSD_MERGE_ROOT,
    MSD_EMBEDDING_MODELS,
    embedded_json_path,
    load_json,
    mean_std_ci,
    parse_seed_args,
    save_json,
    set_seed,
    write_csv,
)


load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = DEEPSEEK
PROMPT_FILE = BASE_DIR / "response_eval_prompt.txt"

EXPECTED_CRITERIA = [
    "Correctness",
    "Relevance",
    "Completeness",
    "Clarity_Coherence",
    "Usefulness_Helpfulness",
    "Style_Tone",
    "Conciseness",
    "Safety_Compliance",
]

CRITERIA_MAP = {
    "Correctness": ["correctness", "accuracy", "factual", "correct", "truthfulness"],
    "Relevance": ["relevance", "relevant", "on_topic", "topicality"],
    "Completeness": ["completeness", "complete", "coverage", "thoroughness"],
    "Clarity_Coherence": ["clarity", "coherence", "readability", "structure"],
    "Usefulness_Helpfulness": ["usefulness", "helpfulness", "useful", "helpful"],
    "Style_Tone": ["style", "tone", "formality", "politeness"],
    "Conciseness": ["conciseness", "concise", "brevity", "verbosity"],
    "Safety_Compliance": ["safety", "compliance", "bias", "harmful", "safe"],
}

VERIFY_KEYS_METHOD = {
    "rbpo_bpo": ["rbpo_prompt", "rbpo_response", "bpo_prompt", "bpo_response"],
    "rbpo_mepo": ["rbpo_prompt", "rbpo_response", "mepo_prompt", "mepo_response"],
    "rmepo_mepo": ["rmepo_prompt", "rmepo_response", "mepo_prompt", "mepo_response"],
    "rgeneric_generic": [
        "rgeneric_prompt",
        "rgeneric_response",
        "generic_prompt",
        "generic_response",
    ],
}


def load_system_prompt():
    namespace = {}
    with PROMPT_FILE.open("r", encoding="utf-8") as f:
        exec(f.read(), namespace)
    return namespace["SYSTEM_PROMPT"]


SYSTEM_PROMPT = load_system_prompt()


def build_user_prompt(item, verify_methods):
    return f"""
Prompt_A:
\"\"\"{item.get(verify_methods[0], "")}\"\"\"

Response_A:
\"\"\"{item.get(verify_methods[1], "")}\"\"\"

Prompt_B:
\"\"\"{item.get(verify_methods[2], "")}\"\"\"

Response_B:
\"\"\"{item.get(verify_methods[3], "")}\"\"\"

Judge Response_A only against Prompt_A and Response_B only against Prompt_B.
Return strict JSON with response_A and response_B scores for these criteria:
{", ".join(EXPECTED_CRITERIA)}.
Scores must be numbers from 0.0 to 1.0.
"""


def call_judge(system_prompt, user_prompt):
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in environment.")
    response = requests.post(
        DEEPSEEK_API_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0,
            "response_format": {"type": "json_object"},
        },
        timeout=120,
    )
    if response.status_code != 200:
        print(f"Status: {response.status_code} | Body: {response.text[:300]}")
        response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def extract_json(raw):
    try:
        json.loads(raw)
        return raw
    except Exception:
        pass
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    return match.group(0).strip() if match else raw.strip()


def map_to_schema(raw_scores):
    mapped = {criteria: None for criteria in EXPECTED_CRITERIA}
    for model_key, value in (raw_scores or {}).items():
        if not isinstance(value, (int, float)):
            continue
        key = model_key.lower()
        for expected, keywords in CRITERIA_MAP.items():
            if mapped[expected] is None and (key == expected.lower() or key in keywords):
                mapped[expected] = float(value)
                break
    values = [value for value in mapped.values() if value is not None]
    fallback = round(sum(values) / len(values), 1) if values else 0.0
    return {key: (fallback if value is None else value) for key, value in mapped.items()}


def is_complete(candidate):
    return (
        isinstance(candidate, dict)
        and all(
            isinstance(candidate.get(side), dict)
            and all(isinstance(candidate[side].get(c), (int, float)) for c in EXPECTED_CRITERIA)
            for side in ["response_A", "response_B"]
        )
    )


def normalize_eval(candidate):
    if is_complete(candidate):
        return candidate
    if isinstance(candidate, dict) and "response_A" in candidate and "response_B" in candidate:
        mapped = {
            "response_A": map_to_schema(candidate.get("response_A")),
            "response_B": map_to_schema(candidate.get("response_B")),
        }
        if is_complete(mapped):
            return mapped
    return {
        "response_A": {criteria: 0.0 for criteria in EXPECTED_CRITERIA},
        "response_B": {criteria: 0.0 for criteria in EXPECTED_CRITERIA},
    }


def decide_winner(scores, threshold=0.01):
    score_a = sum(scores["response_A"].values()) / len(scores["response_A"])
    score_b = sum(scores["response_B"].values()) / len(scores["response_B"])
    diff = score_a - score_b
    if abs(diff) < threshold:
        return 2
    return 0 if diff > 0 else 1


def verify_item(item, verify_key, verify_methods, attempts=3):
    for attempt in range(attempts):
        try:
            raw = call_judge(SYSTEM_PROMPT, build_user_prompt(item, verify_methods))
            parsed = normalize_eval(json.loads(extract_json(raw)))
            return {
                "id": item.get("id"),
                "ori_prompt": item.get("ori_prompt"),
                "context": item.get("context"),
                verify_methods[0]: item.get(verify_methods[0]),
                verify_methods[1]: item.get(verify_methods[1]),
                verify_methods[2]: item.get(verify_methods[2]),
                verify_methods[3]: item.get(verify_methods[3]),
                f"{verify_key}_winner": decide_winner(parsed),
                f"{verify_key}_llm_evaluation": parsed,
            }
        except Exception as error:
            print(f"Judge failed for id={item.get('id')} attempt={attempt + 1}: {error}")
    parsed = normalize_eval({})
    return {
        "id": item.get("id"),
        "ori_prompt": item.get("ori_prompt"),
        f"{verify_key}_winner": decide_winner(parsed),
        f"{verify_key}_llm_evaluation": parsed,
    }


def result_id(item):
    return item.get("id")


def verify_output_path(seed, embedding_model_name, verify_key, experiment_name, run_idx, output_root):
    return (
        embedded_json_path(seed, embedding_model_name, experiment_name, output_root).parent
        / "verify"
        / verify_key
        / f"{experiment_name}_eval_{run_idx}.json"
    )


def run_verification(args):
    for seed in args.seed_values:
        set_seed(seed)
        print(f"\n===== MSD STEP 4 verify | seed={seed} =====")
        for embedding_model_name in MSD_EMBEDDING_MODELS:
            for base_model in base_llm_models:
                for dataset in evaluation_datasets:
                    for evaluator in evaluator_models:
                        experiment_name = create_combined_name(base_model, dataset, evaluator)
                        if experiment_name not in args.experiment_names:
                            continue

                        source_path = embedded_json_path(
                            seed,
                            embedding_model_name,
                            experiment_name,
                            args.output_root,
                        )
                        if not source_path.exists():
                            print(f"Missing response file, skipping: {source_path}")
                            continue
                        data = load_json(source_path)

                        for verify_key, methods in VERIFY_KEYS_METHOD.items():
                            if any(not any(item.get(method) for item in data) for method in methods):
                                print(f"Skipping {verify_key}; missing required fields in {source_path}")
                                continue

                            for run_idx in range(1, args.verify_times + 1):
                                output_path = verify_output_path(
                                    seed,
                                    embedding_model_name,
                                    verify_key,
                                    experiment_name,
                                    run_idx,
                                    args.output_root,
                                )
                                existing_results = []
                                done_ids = set()
                                if output_path.exists() and not args.force:
                                    existing_results = load_json(output_path)
                                    done_ids = {
                                        result_id(item)
                                        for item in existing_results
                                        if result_id(item) is not None
                                        and f"{verify_key}_winner" in item
                                        and f"{verify_key}_llm_evaluation" in item
                                    }
                                    required_ids = {
                                        result_id(item)
                                        for item in data
                                        if result_id(item) is not None
                                    }
                                    if len(required_ids) == len(data) and required_ids <= done_ids:
                                        print(f"Complete, skipping: {output_path}")
                                        continue

                                results = [] if args.force else existing_results
                                progress = tqdm(
                                    data,
                                    desc=(
                                        f"Verify seed={seed} "
                                        f"{clean_name(embedding_model_name)} "
                                        f"{experiment_name} {verify_key} run={run_idx}"
                                    ),
                                    unit="item",
                                )
                                for item in progress:
                                    item_id = result_id(item)
                                    if item_id is not None and item_id in done_ids:
                                        continue
                                    results.append(verify_item(item, verify_key, methods))
                                    if item_id is not None:
                                        done_ids.add(item_id)
                                    if len(results) % args.batch_size == 0:
                                        save_json(output_path, results)
                                save_json(output_path, results)
                                print(f"Saved {output_path}")


def excluded_ids(dataset):
    if "vicuna" in dataset.lower():
        return {81}
    if "dolly" in dataset.lower():
        return {201}
    return set()


def summarize_run(records, verify_key, dataset):
    excluded = excluded_ids(dataset)
    winners = [
        item[f"{verify_key}_winner"]
        for item in records
        if item.get("id") not in excluded and f"{verify_key}_winner" in item
    ]
    total = len(winners)
    counts = Counter(winners)
    return {
        "counted_items": total,
        "a_win_rate": counts[0] / total if total else 0.0,
        "b_win_rate": counts[1] / total if total else 0.0,
        "draw_rate": counts[2] / total if total else 0.0,
        "pref_score": (counts[0] - counts[1]) / total if total else 0.0,
    }


def aggregate(args):
    per_run_rows = []
    per_seed_values = defaultdict(lambda: defaultdict(list))

    for seed in tqdm(args.seed_values, desc="Aggregate seeds", unit="seed"):
        for embedding_model_name in MSD_EMBEDDING_MODELS:
            for base_model in base_llm_models:
                for dataset in evaluation_datasets:
                    for evaluator in evaluator_models:
                        experiment_name = create_combined_name(base_model, dataset, evaluator)
                        if experiment_name not in args.experiment_names:
                            continue
                        for verify_key in VERIFY_KEYS_METHOD:
                            for run_idx in range(1, args.verify_times + 1):
                                path = verify_output_path(
                                    seed,
                                    embedding_model_name,
                                    verify_key,
                                    experiment_name,
                                    run_idx,
                                    args.output_root,
                                )
                                if not path.exists():
                                    continue
                                row = summarize_run(load_json(path), verify_key, dataset)
                                row.update(
                                    {
                                        "seed": seed,
                                        "verify_run": run_idx,
                                        "embedding_model": clean_name(embedding_model_name),
                                        "experiment": experiment_name,
                                        "comparison": verify_key,
                                    }
                                )
                                per_run_rows.append(row)
                                group_key = (
                                    clean_name(embedding_model_name),
                                    experiment_name,
                                    verify_key,
                                    seed,
                                )
                                for metric in ["a_win_rate", "b_win_rate", "draw_rate", "pref_score"]:
                                    per_seed_values[group_key][metric].append(row[metric])

    per_seed_rows = []
    grouped_for_ci = defaultdict(lambda: defaultdict(list))
    for (embed, experiment, comparison, seed), metrics in per_seed_values.items():
        row = {
            "seed": seed,
            "embedding_model": embed,
            "experiment": experiment,
            "comparison": comparison,
        }
        for metric, values in metrics.items():
            row[metric] = sum(values) / len(values)
            grouped_for_ci[(embed, experiment, comparison)][metric].append(row[metric])
        per_seed_rows.append(row)

    aggregate_rows = []
    for (embed, experiment, comparison), metrics in grouped_for_ci.items():
        row = {
            "embedding_model": embed,
            "experiment": experiment,
            "comparison": comparison,
            "n_seeds": len(next(iter(metrics.values()))),
        }
        for metric, values in metrics.items():
            mean, std, ci_low, ci_high = mean_std_ci(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        aggregate_rows.append(row)

    summary_dir = args.output_root / "summary"
    write_csv(summary_dir / "msd_per_verify_run.csv", per_run_rows)
    write_csv(summary_dir / "msd_per_seed_summary.csv", per_seed_rows)
    write_csv(summary_dir / "msd_aggregate_ci95.csv", aggregate_rows)
    save_json(summary_dir / "msd_aggregate_ci95.json", aggregate_rows)
    print(f"Saved MSD summaries to {summary_dir}")


def main():
    args = parse_seed_args(
        "Verify responses and aggregate MSD seed variance/CI.",
        default_output_root=MSD_MERGE_ROOT,
    )
    if not args.aggregate_only:
        run_verification(args)
    aggregate(args)


if __name__ == "__main__":
    main()
