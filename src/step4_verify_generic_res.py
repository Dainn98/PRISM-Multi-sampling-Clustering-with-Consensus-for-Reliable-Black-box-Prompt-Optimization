import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import requests
from dotenv import load_dotenv

from helper import (
    DEEPSEEK,
    clean_name,
    consistency_folder_name,
    create_combined_name,
    eval_folder_name,
    mismatch_folder_name,
    base_llm_models,
    embedding_models,
    evaluation_datasets,
)


BASE_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = BASE_DIR / eval_folder_name
PROMPT_FILE = BASE_DIR / "response_eval_prompt.txt"

MODEL_NAME = DEEPSEEK
VERIFY_FOLDER_NAME = f"verify_{clean_name(MODEL_NAME)}"
VERIFY_DIR_NAME = VERIFY_FOLDER_NAME
EVALUATOR_MODELS = [DEEPSEEK]

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 120
API_RETRY_TIMES = 5
API_RETRY_BASE_DELAY = 2
VERIFY_TIMES = 3
CHECKPOINT_EVERY = 10
RUN_VERIFICATION = True
RUN_CONSISTENCY_CHECK = True

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

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

VERIFY_KEYS_METHOD = {
    "ori_generic": (
        "ori_prompt",
        "ori_response",
        "generic_prompt",
        "generic_response",
    ),
    "ori_rgeneric": (
        "ori_prompt",
        "ori_response",
        "rgeneric_prompt",
        "rgeneric_response",
    ),
    "generic_rgeneric": (
        "generic_prompt",
        "generic_response",
        "rgeneric_prompt",
        "rgeneric_response",
    ),
}


def load_system_prompt():
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

    prompt_vars = {}
    with PROMPT_FILE.open("r", encoding="utf-8") as f:
        exec(f.read(), prompt_vars)
    return prompt_vars["SYSTEM_PROMPT"]


SYSTEM_PROMPT = load_system_prompt()


def has_text(value):
    return isinstance(value, str) and bool(value.strip())


def has_complete_pair(item, verify_methods):
    return all(has_text(item.get(key)) for key in verify_methods)


def build_user_prompt(item, verify_methods):
    prompt_a, response_a, prompt_b, response_b = verify_methods
    context = item.get("context")
    context_section = (
        f'Shared context for both prompts:\n"""{context}"""\n\n'
        if has_text(context)
        else ""
    )
    return f"""
{context_section}Prompt_A (used to generate Response_A):
\"\"\"{item[prompt_a]}\"\"\"

Response_A:
\"\"\"{item[response_a]}\"\"\"

Prompt_B (used to generate Response_B):
\"\"\"{item[prompt_b]}\"\"\"

Response_B:
\"\"\"{item[response_b]}\"\"\"

# IMPORTANT RULES:
- Judge Response_A ONLY based on Prompt_A and the shared context, if provided
- Judge Response_B ONLY based on Prompt_B and the shared context, if provided
- Do NOT compare Response_A and Response_B directly
- Use the SAME scoring scale for both
- Score each criterion independently
- If a response fails to answer, give low scores (0.0 to 0.2)
- Use only numbers between 0.0 and 1.0
- Use one decimal place
- Do NOT include explanations or extra text

# OUTPUT:
Return STRICTLY valid JSON using this exact structure:
{{
  "response_A": {{
    "Correctness": 0.0,
    "Relevance": 0.0,
    "Completeness": 0.0,
    "Clarity_Coherence": 0.0,
    "Usefulness_Helpfulness": 0.0,
    "Style_Tone": 0.0,
    "Conciseness": 0.0,
    "Safety_Compliance": 0.0
  }},
  "response_B": {{
    "Correctness": 0.0,
    "Relevance": 0.0,
    "Completeness": 0.0,
    "Clarity_Coherence": 0.0,
    "Usefulness_Helpfulness": 0.0,
    "Style_Tone": 0.0,
    "Conciseness": 0.0,
    "Safety_Compliance": 0.0
  }}
}}
"""


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
        timeout=REQUEST_TIMEOUT_SECONDS,
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


def verify_response(item_id, sample, verify_methods, attempt_times=API_RETRY_TIMES):
    last_error = None
    for attempt in range(1, attempt_times + 1):
        try:
            raw = generate(SYSTEM_PROMPT, build_user_prompt(sample, verify_methods))
            candidate = json.loads(extract_json(raw))
            parsed = normalize_evaluation(candidate)
            if parsed is not None:
                return parsed
            raise ValueError("DeepSeek response does not match the evaluation schema")
        except (KeyError, TypeError, ValueError, requests.RequestException) as error:
            last_error = error
            if attempt == attempt_times:
                break

            delay = API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"ID={item_id}: attempt {attempt}/{attempt_times} failed: "
                f"{error}. Retrying in {delay}s..."
            )
            time.sleep(delay)

    raise RuntimeError(
        f"ID={item_id}: failed after {attempt_times} attempts"
    ) from last_error


def process_item(item, verify_key, verify_methods, fallback_id):
    item_id = item.get("id", fallback_id)
    parsed = verify_response(item_id, item, verify_methods)
    result = {
        "id": item_id,
        "ori_prompt": item.get("ori_prompt"),
        "context": item.get("context"),
    }
    for key in verify_methods:
        result[key] = item.get(key)

    result[f"{verify_key}_winner"] = decide_winner_from_scores(parsed)
    result[f"{verify_key}_llm_evaluation"] = parsed
    return result


def save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_path, path)


def load_checkpoint(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    if not isinstance(checkpoint, list):
        raise ValueError(f"Invalid checkpoint format: {path}")
    return checkpoint


def get_input_path(embed_model, base_llm, dataset, evaluator):
    file_name = create_combined_name(base_llm, dataset, evaluator)
    return EVALUATION_DIR / clean_name(embed_model) / f"{file_name}.json"


def get_output_path(embed_model, verify_key, file_name, run_idx):
    return (
        EVALUATION_DIR
        / clean_name(embed_model)
        / VERIFY_DIR_NAME
        / verify_key
        / f"{file_name}_eval_{run_idx}.json"
    )


def verify_response_batch(verify_key, verify_methods, verify_times=VERIFY_TIMES):
    for embed_model in embedding_models:
        for base_llm in base_llm_models:
            for dataset in evaluation_datasets:
                for evaluator in EVALUATOR_MODELS:
                    file_name = create_combined_name(base_llm, dataset, evaluator)
                    input_path = get_input_path(
                        embed_model,
                        base_llm,
                        dataset,
                        evaluator,
                    )
                    if not input_path.exists():
                        print(f"Skipping missing input: {input_path}")
                        continue

                    with input_path.open("r", encoding="utf-8") as f:
                        source_data = json.load(f)

                    indexed_items = []
                    skipped = 0
                    for index, item in enumerate(source_data, start=1):
                        if has_complete_pair(item, verify_methods):
                            item_to_process = dict(item)
                            item_to_process.setdefault("id", index)
                            indexed_items.append(item_to_process)
                        else:
                            skipped += 1

                    if not indexed_items:
                        print(
                            f"Skipping {verify_key} for {file_name}: "
                            "required prompt/response keys are missing"
                        )
                        continue

                    print(
                        f"\n{verify_key} | {clean_name(embed_model)} | "
                        f"{file_name}: {len(indexed_items)} valid, "
                        f"{skipped} skipped"
                    )
                    for run_idx in range(1, verify_times + 1):
                        output_path = get_output_path(
                            embed_model,
                            verify_key,
                            file_name,
                            run_idx,
                        )
                        results = load_checkpoint(output_path)
                        completed_ids = {item.get("id") for item in results}
                        pending_items = [
                            item
                            for item in indexed_items
                            if item.get("id") not in completed_ids
                        ]

                        print(
                            f"Run {run_idx}/{verify_times}: "
                            f"{len(results)}/{len(indexed_items)} completed, "
                            f"{len(pending_items)} pending"
                        )

                        try:
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
                                    save_json_atomic(output_path, results)
                                    print(f"  checkpoint: {len(results)} results")

                            save_json_atomic(output_path, results)
                            print(f"Saved: {output_path}")
                        except KeyboardInterrupt:
                            print(f"\nInterrupted. Progress saved at: {output_path}")
                            raise


def load_consistency_runs(embed_model, verify_key, file_name, check_runs):
    runs = []
    for run_idx in range(1, check_runs + 1):
        path = get_output_path(embed_model, verify_key, file_name, run_idx)
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

    def percentage(count):
        return f"{(count / total * 100):.2f}%" if total else "0.00%"

    return {
        "response_A_win": counts[0],
        "response_A_win_rate": percentage(counts[0]),
        "response_B_win": counts[1],
        "response_B_win_rate": percentage(counts[1]),
        "draw": counts[2],
        "draw_rate": percentage(counts[2]),
        "excluded_count": excluded_count,
        "counted_items": total,
    }


def comparison_output_dir(embed_model, verify_key):
    return EVALUATION_DIR / clean_name(embed_model) / VERIFY_DIR_NAME / verify_key


def check_verify_consistency(
    verify_key,
    verify_methods,
    check_runs=VERIFY_TIMES,
):
    for embed_model in embedding_models:
        for base_llm in base_llm_models:
            for dataset in evaluation_datasets:
                for evaluator in EVALUATOR_MODELS:
                    file_name = create_combined_name(base_llm, dataset, evaluator)
                    runs = load_consistency_runs(
                        embed_model,
                        verify_key,
                        file_name,
                        check_runs,
                    )
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
                            "ori_prompt": first.get("ori_prompt"),
                            "context": first.get("context"),
                            **{key: first.get(key) for key in verify_methods},
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

                    output_dir = comparison_output_dir(embed_model, verify_key)
                    save_json_atomic(
                        output_dir
                        / mismatch_folder_name
                        / f"{file_name}_mismatch.json",
                        mismatch_result,
                    )
                    save_json_atomic(
                        output_dir
                        / consistency_folder_name
                        / f"{file_name}_consistency.json",
                        consistency_result,
                    )
                    print(
                        f"{verify_key}/{clean_name(embed_model)}/{file_name}: "
                        f"{len(consistencies)} consistent, "
                        f"{len(mismatches)} mismatched"
                    )


def main():
    print(f"===== Generic Response Verification | Judge: {MODEL_NAME} =====")
    for verify_key, verify_methods in VERIFY_KEYS_METHOD.items():
        print(f"\n=== VERIFY: {verify_key} ===")
        print(f"Keys: {verify_methods}")
        if RUN_VERIFICATION:
            verify_response_batch(verify_key, verify_methods)
        if RUN_CONSISTENCY_CHECK:
            check_verify_consistency(verify_key, verify_methods)


if __name__ == "__main__":
    main()
