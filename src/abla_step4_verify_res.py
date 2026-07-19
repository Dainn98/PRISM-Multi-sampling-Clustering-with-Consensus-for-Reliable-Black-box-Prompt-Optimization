import json
import os
import re
from collections import Counter
from pathlib import Path

import requests
from dotenv import load_dotenv

from helper import (
    DEEPSEEK,
    DOLLY_EVAL,
    LLAMA2_7B,
    SELF_INSTRUCT_EVAL,
    create_combined_name,
)


BASE_DIR = Path(__file__).resolve().parent
ABLATION_DIR = BASE_DIR / "ablation"
PROMPT_FILE = BASE_DIR / "response_eval_prompt.txt"
VERIFY_DIR = ABLATION_DIR / "verify"

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = DEEPSEEK

BASE_LLM_MODELS = [LLAMA2_7B]
EVALUATION_DATASETS = [DOLLY_EVAL, SELF_INSTRUCT_EVAL]
EVALUATOR_MODELS = [DEEPSEEK]

BATCH_SIZE = 1
VERIFY_TIMES = 3
ATTEMPT_TIMES = 3
CHECKPOINT_EVERY = 10
RUN_VERIFICATION = True
RUN_CONSISTENCY_CHECK = True

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
    "Correctness": [
        "correctness",
        "accuracy",
        "factual",
        "correct",
        "truthfulness",
        "honesty",
    ],
    "Relevance": ["relevance", "relevant", "on_topic", "topicality"],
    "Completeness": [
        "completeness",
        "complete",
        "coverage",
        "thoroughness",
        "recall",
    ],
    "Clarity_Coherence": [
        "clarity",
        "coherence",
        "clarity_coherence",
        "readability",
        "structure",
        "clear",
    ],
    "Usefulness_Helpfulness": [
        "usefulness",
        "helpfulness",
        "useful",
        "helpful",
        "instruction_following",
        "practicality",
    ],
    "Style_Tone": ["style", "tone", "style_tone", "formality", "politeness"],
    "Conciseness": ["conciseness", "concise", "brevity", "verbosity"],
    "Safety_Compliance": [
        "safety",
        "safety_compliance",
        "compliance",
        "bias",
        "harmful",
        "safe",
    ],
}

# Response_A is always the original LLM response. Response_B is the method
# being compared against it.
VERIFY_KEYS_METHOD = {
    "ori_mepo": (
        "ori_prompt",
        "ori_response",
        "mepo_prompt",
        "mepo_response",
    ),
    "ori_rmepo": (
        "ori_prompt",
        "ori_response",
        "rmepo_prompt",
        "rmepo_response",
    ),
    "ori_rmepo_random": (
        "ori_prompt",
        "ori_response",
        "rmepo_random_prompt",
        "rmepo_random_response",
    ),
    "ori_rmepo_nearest": (
        "ori_prompt",
        "ori_response",
        "rmepo_nearest_prompt",
        "rmepo_nearest_response",
    ),
    "ori_rmepo_farthest": (
        "ori_prompt",
        "ori_response",
        "rmepo_farthest_prompt",
        "rmepo_farthest_response",
    ),
    "ori_rmepo_majority": (
        "ori_prompt",
        "ori_response",
        "rmepo_majority_prompt",
        "rmepo_majority_response",
    ),
    "ori_rmepo_without_clustering": (
        "ori_prompt",
        "ori_response",
        "rmepo_without_clustering_prompt",
        "rmepo_without_clustering_response",
    ),
    "ori_rmepo_without_consensus": (
        "ori_prompt",
        "ori_response",
        "rmepo_without_consensus_prompt",
        "rmepo_without_consensus_response",
    ),
}


def load_prompts():
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

    prompt_vars = {}
    with PROMPT_FILE.open("r", encoding="utf-8") as f:
        exec(f.read(), prompt_vars)
    return prompt_vars["SYSTEM_PROMPT"], prompt_vars["USER_PROMPT_TEMPLATE"]


SYSTEM_PROMPT, USER_PROMPT_TEMPLATE = load_prompts()


def has_text(value):
    return isinstance(value, str) and bool(value.strip())


def has_complete_pair(item, verify_methods):
    return all(has_text(item.get(key)) for key in verify_methods)


def format_user_prompt(prompt_a, response_a, prompt_b, response_b):
    return (
        USER_PROMPT_TEMPLATE
        .replace("{prompt_A}", prompt_a or "")
        .replace("{response_A}", response_a or "")
        .replace("{prompt_B}", prompt_b or "")
        .replace("{response_B}", response_b or "")
    )


def build_user_prompt(item, verify_methods):
    prompt_a, response_a, prompt_b, response_b = verify_methods
    context = item.get("context")
    context_prefix = f'Shared context:\n"""{context}"""\n\n' if has_text(context) else ""
    return format_user_prompt(
        context_prefix + item[prompt_a],
        item[response_a],
        context_prefix + item[prompt_b],
        item[response_b],
    )


def generate(system_prompt, user_prompt):
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is missing from the environment")

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
    except (TypeError, json.JSONDecodeError):
        pass

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    code_block = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
    if code_block:
        return code_block.group(1).strip()

    json_object = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    return json_object.group(0).strip() if json_object else raw.strip()


def normalize_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(min(1.0, max(0.0, float(value))), 1)


def map_to_schema(raw_scores):
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    mapped = {criterion: None for criterion in EXPECTED_CRITERIA}
    for model_key, model_value in raw_scores.items():
        score = normalize_score(model_value)
        if score is None:
            continue

        normalized_key = str(model_key).lower().strip().replace(" ", "_")
        for expected, keywords in CRITERIA_MAP.items():
            if mapped[expected] is None and (
                normalized_key == expected.lower() or normalized_key in keywords
            ):
                mapped[expected] = score
                break

    mapped_values = [value for value in mapped.values() if value is not None]
    fallback = round(sum(mapped_values) / len(mapped_values), 1) if mapped_values else 0.0
    return {
        criterion: value if value is not None else fallback
        for criterion, value in mapped.items()
    }


def normalize_evaluation(candidate):
    if not isinstance(candidate, dict):
        return None
    if "response_A" not in candidate or "response_B" not in candidate:
        return None

    return {
        "response_A": map_to_schema(candidate["response_A"]),
        "response_B": map_to_schema(candidate["response_B"]),
    }


def empty_schema():
    return {
        "response_A": {criterion: 0.0 for criterion in EXPECTED_CRITERIA},
        "response_B": {criterion: 0.0 for criterion in EXPECTED_CRITERIA},
    }


def decide_winner_from_scores(llm_eval, threshold=0.01):
    score_a = sum(llm_eval["response_A"].values()) / len(EXPECTED_CRITERIA)
    score_b = sum(llm_eval["response_B"].values()) / len(EXPECTED_CRITERIA)
    difference = score_a - score_b

    if abs(difference) < threshold:
        return 2
    return 0 if difference > 0 else 1


def verify_response(item_id, sample, verify_methods, attempt_times=ATTEMPT_TIMES):
    for attempt in range(1, attempt_times + 1):
        try:
            raw = generate(SYSTEM_PROMPT, build_user_prompt(sample, verify_methods))
            candidate = json.loads(extract_json(raw))
            parsed = normalize_evaluation(candidate)
            if parsed is not None:
                return parsed
            print(f"ID={item_id}: invalid schema on attempt {attempt}")
        except (KeyError, TypeError, ValueError, requests.RequestException) as error:
            print(f"ID={item_id}: attempt {attempt} failed: {error}")

    print(f"ID={item_id}: using zero-score fallback")
    return empty_schema()


def process_item(item, verify_key, verify_methods, fallback_id):
    item_id = item.get("id", fallback_id)
    parsed = verify_response(item_id, item, verify_methods)
    result = {
        "id": item_id,
        "context": item.get("context"),
    }
    for key in verify_methods:
        result[key] = item.get(key)

    result[f"{verify_key}_winner"] = decide_winner_from_scores(parsed)
    result[f"{verify_key}_llm_evaluation"] = parsed
    return result


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def verify_response_batch(verify_key, verify_methods, verify_times=VERIFY_TIMES):
    for base_llm in BASE_LLM_MODELS:
        for dataset in EVALUATION_DATASETS:
            for evaluator in EVALUATOR_MODELS:
                file_name = create_combined_name(base_llm, dataset, evaluator)
                input_path = ABLATION_DIR / f"{file_name}.json"
                if not input_path.exists():
                    print(f"Skipping missing input: {input_path}")
                    continue

                with input_path.open("r", encoding="utf-8") as f:
                    source_data = json.load(f)

                valid_items = [
                    item
                    for item in source_data
                    if has_complete_pair(item, verify_methods)
                ]
                skipped = len(source_data) - len(valid_items)
                if not valid_items:
                    print(
                        f"Skipping {verify_key} for {file_name}: "
                        "required prompt/response keys are missing"
                    )
                    continue

                print(
                    f"\n{verify_key} | {file_name}: "
                    f"{len(valid_items)} valid, {skipped} skipped"
                )
                for run_idx in range(1, verify_times + 1):
                    output_path = (
                        VERIFY_DIR
                        / verify_key
                        / f"{file_name}_eval_{run_idx}.json"
                    )
                    results = []
                    completed_ids = set()
                    if output_path.exists():
                        with output_path.open("r", encoding="utf-8") as f:
                            results = json.load(f)
                        completed_ids = {item.get("id") for item in results}

                    pending_items = [
                        item
                        for index, item in enumerate(valid_items, start=1)
                        if item.get("id", index) not in completed_ids
                    ]
                    print(
                        f"Run {run_idx}/{verify_times}: "
                        f"{len(pending_items)} pending"
                    )

                    for index, item in enumerate(pending_items, start=1):
                        results.append(
                            process_item(
                                item,
                                verify_key,
                                verify_methods,
                                fallback_id=index,
                            )
                        )
                        if index % CHECKPOINT_EVERY == 0:
                            save_json(output_path, results)
                            print(f"  checkpoint: {len(results)} results")

                    save_json(output_path, results)
                    print(f"Saved: {output_path}")


def load_consistency_runs(verify_key, file_name, check_runs):
    runs = []
    for run_idx in range(1, check_runs + 1):
        path = VERIFY_DIR / verify_key / f"{file_name}_eval_{run_idx}.json"
        if not path.exists():
            print(f"Missing consistency run: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def winner_distribution(items, exclude_ids):
    counts = {0: 0, 1: 0, 2: 0}
    excluded_count = 0

    for item in items:
        if item.get("id") in exclude_ids:
            excluded_count += 1
            continue
        winners = item.get("winners_per_run", [])
        if winners:
            counts[Counter(winners).most_common(1)[0][0]] += 1

    total = sum(counts.values())
    percentage = lambda count: f"{(count / total * 100):.2f}%" if total else "0.00%"
    return {
        "ori_response_win": counts[0],
        "ori_response_win_rate": percentage(counts[0]),
        "other_response_win": counts[1],
        "other_response_win_rate": percentage(counts[1]),
        "draw": counts[2],
        "draw_rate": percentage(counts[2]),
        "excluded_count": excluded_count,
        "counted_items": total,
    }


def check_verify_consistency(
    verify_key,
    verify_methods,
    check_runs=VERIFY_TIMES,
):
    for base_llm in BASE_LLM_MODELS:
        for dataset in EVALUATION_DATASETS:
            for evaluator in EVALUATOR_MODELS:
                file_name = create_combined_name(base_llm, dataset, evaluator)
                runs = load_consistency_runs(verify_key, file_name, check_runs)
                if len(runs) != check_runs:
                    print(
                        f"Skipping consistency for {verify_key}/{file_name}: "
                        f"found {len(runs)}/{check_runs} runs"
                    )
                    continue

                runs_by_id = [
                    {item.get("id"): item for item in run}
                    for run in runs
                ]
                common_ids = set(runs_by_id[0])
                for run_by_id in runs_by_id[1:]:
                    common_ids &= set(run_by_id)

                mismatches = []
                consistencies = []
                for item_id in sorted(common_ids, key=str):
                    first = runs_by_id[0][item_id]
                    winners = [
                        run[item_id][f"{verify_key}_winner"]
                        for run in runs_by_id
                    ]
                    item_data = {
                        "id": item_id,
                        **{
                            key: first.get(key)
                            for key in verify_methods
                        },
                        "winners_per_run": winners,
                        "llm_evaluations_per_run": [
                            run[item_id][f"{verify_key}_llm_evaluation"]
                            for run in runs_by_id
                        ],
                    }
                    target = mismatches if len(set(winners)) > 1 else consistencies
                    target.append(item_data)

                exclude_ids = set()
                if "vicuna" in dataset.lower():
                    exclude_ids.add(81)
                elif "dolly" in dataset.lower():
                    exclude_ids.add(201)

                total = len(common_ids)
                comparison_dir = VERIFY_DIR / verify_key
                mismatch_result = {
                    "total_items": total,
                    "total_mismatches": len(mismatches),
                    "mismatch_rate": (
                        f"{(len(mismatches) / total * 100):.2f}%"
                        if total
                        else "0.00%"
                    ),
                    "check_runs": check_runs,
                    "mismatches": mismatches,
                }
                consistency_result = {
                    "total_items": total,
                    "total_consistent": len(consistencies),
                    "consistency_rate": (
                        f"{(len(consistencies) / total * 100):.2f}%"
                        if total
                        else "0.00%"
                    ),
                    "check_runs": check_runs,
                    "winner_distribution": winner_distribution(
                        consistencies,
                        exclude_ids,
                    ),
                    "excluded_ids": sorted(exclude_ids),
                    "consistent_items": consistencies,
                }

                save_json(
                    comparison_dir / "mismatch" / f"{file_name}_mismatch.json",
                    mismatch_result,
                )
                save_json(
                    comparison_dir
                    / "consistency"
                    / f"{file_name}_consistency.json",
                    consistency_result,
                )
                print(
                    f"{verify_key}/{file_name}: "
                    f"{len(consistencies)} consistent, "
                    f"{len(mismatches)} mismatched"
                )


def main():
    for verify_key, verify_methods in VERIFY_KEYS_METHOD.items():
        print(f"\n=== VERIFY: {verify_key} ===")
        if RUN_VERIFICATION:
            verify_response_batch(verify_key, verify_methods)
        if RUN_CONSISTENCY_CHECK:
            check_verify_consistency(verify_key, verify_methods)


if __name__ == "__main__":
    main()
