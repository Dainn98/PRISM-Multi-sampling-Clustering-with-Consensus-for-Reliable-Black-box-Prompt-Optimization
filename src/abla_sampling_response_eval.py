import argparse
import csv
import gc
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import requests
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from config import MODEL_CACHE_PATH, SEED
from helper import DEEPSEEK, HF_TOKEN, LLAMA2_7B, device, load_model_and_tokenizer, set_global_seed
from utils import generate_batch


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "ablation_sampling_hparams" / "all-MiniLM-L12-v2" / "llama_dolly_deepseek"
RESPONSE_SUFFIX = "_responses"
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

SYSTEM_PROMPT = """You are a strict and consistent judge of language-model responses.
Score each response independently against its own prompt and the shared context.
Use the same scale for both responses. Return one valid JSON object only, with no markdown or explanation.
Each score must be a number from 0.0 to 1.0."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ori/rbpo responses and compute RBPO win, tie, and loss rates."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="One ablation JSON file or a directory containing ablation JSON files.",
    )
    parser.add_argument("--stage", choices=("all", "generate", "judge"), default="all")
    parser.add_argument("--base-model", default=LLAMA2_7B)
    parser.add_argument("--judge-model", default=DEEPSEEK)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-runs", type=int, default=3)
    parser.add_argument("--tie-threshold", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite-responses", action="store_true")
    parser.add_argument("--overwrite-judgments", action="store_true")
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def output_path_for(input_path):
    return input_path.with_name(f"{input_path.stem}{RESPONSE_SUFFIX}.json")


def discover_inputs(path):
    path = path.resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")

    paths = []
    for candidate in sorted(path.rglob("*.json")):
        if candidate.stem.endswith(RESPONSE_SUFFIX) or candidate.name.startswith("summary"):
            continue
        data = load_json(candidate)
        if isinstance(data, list) and any(
            isinstance(item, dict) and "ori_prompt" in item and "rbpo_prompt" in item
            for item in data
        ):
            paths.append(candidate)
    return paths


def has_text(value):
    return isinstance(value, str) and bool(value.strip())


def format_generation_prompt(tokenizer, prompt, context):
    content = prompt
    if has_text(context):
        content = f"Context:\n{context}\n\nQuestion:\n{prompt}"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and concise assistant. Reply in English only.",
        },
        {"role": "user", "content": content},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, ValueError, TypeError):
        return content


def generate_responses(data, model, tokenizer, args, output_path):
    jobs = []
    for item_index, item in enumerate(data):
        for prompt_key, response_key in (
            ("ori_prompt", "ori_response"),
            ("rbpo_prompt", "rbpo_response"),
        ):
            if not has_text(item.get(prompt_key)):
                continue
            if has_text(item.get(response_key)) and not args.overwrite_responses:
                continue
            jobs.append(
                (
                    item_index,
                    response_key,
                    format_generation_prompt(
                        tokenizer,
                        item[prompt_key],
                        item.get("context"),
                    ),
                )
            )

    position = 0
    current_batch_size = min(args.batch_size, len(jobs)) if jobs else args.batch_size
    progress = tqdm(total=len(jobs), desc="Generating responses")
    while position < len(jobs):
        batch = jobs[position : position + current_batch_size]
        try:
            with torch.inference_mode():
                responses = generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=[job[2] for job in batch],
                    batch_size=current_batch_size,
                    max_new_tokens=args.max_new_tokens,
                    apply_chat_template=False,
                    do_sample=False,
                    device=device,
                )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if current_batch_size == 1:
                progress.close()
                raise RuntimeError(
                    "CUDA OOM at batch_size=1. Reduce --max-new-tokens or "
                    "stop other GPU processes before retrying."
                ) from None
            current_batch_size = max(1, current_batch_size // 2)
            print(f"CUDA OOM: retrying with batch_size={current_batch_size}")
            continue

        for (item_index, response_key, _), response in zip(batch, responses):
            data[item_index][response_key] = response
        position += len(batch)
        progress.update(len(batch))
        save_json_atomic(output_path, data)
    progress.close()

    return len(jobs)


def build_judge_prompt(item):
    context = item.get("context") if has_text(item.get("context")) else "None"
    prompt_a = item["ori_prompt"]
    response_a = item["ori_response"]
    prompt_b = item["rbpo_prompt"]
    response_b = item["rbpo_response"]
    schema = {
        "response_A": {criterion: 0.0 for criterion in CRITERIA},
        "response_B": {criterion: 0.0 for criterion in CRITERIA},
    }
    return f"""Shared context:
\"\"\"{context}\"\"\"

Prompt_A:
\"\"\"{prompt_a}\"\"\"
Response_A:
\"\"\"{response_a}\"\"\"

Prompt_B:
\"\"\"{prompt_b}\"\"\"
Response_B:
\"\"\"{response_b}\"\"\"

Judge Response_A only against Prompt_A and Response_B only against Prompt_B. Score correctness,
relevance, completeness, clarity, usefulness, style, conciseness, and safety independently.
Return exactly this JSON structure with numeric scores:
{json.dumps(schema)}"""


def extract_json(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start : end + 1] if start >= 0 and end > start else raw


def normalize_evaluation(candidate):
    normalized = {}
    for response_key in ("response_A", "response_B"):
        scores = candidate.get(response_key)
        if not isinstance(scores, dict):
            raise ValueError(f"Missing {response_key} scores")
        normalized[response_key] = {}
        for criterion in CRITERIA:
            value = scores.get(criterion)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Missing numeric score: {response_key}.{criterion}")
            normalized[response_key][criterion] = min(1.0, max(0.0, float(value)))
    return normalized


def request_judgment(api_key, model_name, user_prompt, attempts=5):
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 2048,
                    "response_format": {"type": "json_object"},
                },
                timeout=120,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            return normalize_evaluation(json.loads(extract_json(raw)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, requests.RequestException) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Judge failed after {attempts} attempts: {last_error}") from last_error


def mean_score(scores):
    return sum(scores.values()) / len(CRITERIA)


def result_from_scores(ori_score, rbpo_score, tie_threshold):
    difference = rbpo_score - ori_score
    if abs(difference) < tie_threshold:
        return "tie"
    return "win" if difference > 0 else "loss"


def judge_item(item, api_key, args):
    runs = [] if args.overwrite_judgments else item.get("ori_rbpo_judge_runs", [])
    if not isinstance(runs, list):
        runs = []

    for run_index in range(len(runs), args.judge_runs):
        evaluation = request_judgment(
            api_key,
            args.judge_model,
            build_judge_prompt(item),
        )
        ori_score = mean_score(evaluation["response_A"])
        rbpo_score = mean_score(evaluation["response_B"])
        runs.append(
            {
                "run": run_index + 1,
                "ori_score": ori_score,
                "rbpo_score": rbpo_score,
                "result": result_from_scores(
                    ori_score,
                    rbpo_score,
                    args.tie_threshold,
                ),
                "evaluation": evaluation,
            }
        )

    item["ori_rbpo_judge_runs"] = runs[: args.judge_runs]
    refresh_result_from_runs(item, args)


def refresh_result_from_runs(item, args):
    runs = item.get("ori_rbpo_judge_runs", [])
    if not isinstance(runs, list) or len(runs) < args.judge_runs:
        return False
    selected_runs = runs[: args.judge_runs]
    votes = []
    for run in selected_runs:
        run_result = result_from_scores(
            run["ori_score"],
            run["rbpo_score"],
            args.tie_threshold,
        )
        run["result"] = run_result
        votes.append(run_result)

    vote_counts = Counter(votes)
    highest_count = max(vote_counts.values())
    leaders = [result for result, count in vote_counts.items() if count == highest_count]
    has_majority = highest_count > len(votes) / 2
    majority_result = leaders[0] if len(leaders) == 1 and has_majority else "tie"
    ori_score = sum(run["ori_score"] for run in selected_runs) / args.judge_runs
    rbpo_score = sum(run["rbpo_score"] for run in selected_runs) / args.judge_runs
    difference = rbpo_score - ori_score
    item["ori_rbpo_result"] = {
        "result": majority_result,
        "votes": votes,
        "vote_counts": {
            "win": vote_counts["win"],
            "tie": vote_counts["tie"],
            "loss": vote_counts["loss"],
        },
        "ori_score": ori_score,
        "rbpo_score": rbpo_score,
        "score_difference": difference,
        "tie_threshold": args.tie_threshold,
    }
    return True


def judge_responses(data, api_key, args, output_path):
    judged = 0
    for item in tqdm(data, desc="Judging ori vs RBPO"):
        required = ("ori_prompt", "ori_response", "rbpo_prompt", "rbpo_response")
        if not all(has_text(item.get(key)) for key in required):
            continue
        complete = len(item.get("ori_rbpo_judge_runs", [])) >= args.judge_runs
        if complete and not args.overwrite_judgments:
            refresh_result_from_runs(item, args)
            continue
        judge_item(item, api_key, args)
        judged += 1
        save_json_atomic(output_path, data)
    return judged


def summarize(input_path, output_path, data):
    results = [
        item["ori_rbpo_result"]["result"]
        for item in data
        if isinstance(item.get("ori_rbpo_result"), dict)
    ]
    counts = Counter(results)
    total = len(results)
    rate = lambda key: counts[key] / total if total else 0.0
    return {
        "config": input_path.parent.name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_items": len(data),
        "num_evaluated": total,
        "win": counts["win"],
        "tie": counts["tie"],
        "loss": counts["loss"],
        "win_rate": rate("win"),
        "tie_rate": rate("tie"),
        "loss_rate": rate("loss"),
    }


def save_summary(root, rows):
    root = root if root.is_dir() else root.parent
    save_json_atomic(root / "response_eval_summary.json", rows)
    fieldnames = list(rows[0].keys()) if rows else []
    if fieldnames:
        with (root / "response_eval_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.judge_runs < 1:
        raise ValueError("--judge-runs must be at least 1")
    if args.tie_threshold < 0:
        raise ValueError("--tie-threshold cannot be negative")

    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")
    load_dotenv()
    set_global_seed(SEED)
    input_paths = discover_inputs(args.input)
    if not input_paths:
        raise FileNotFoundError(f"No ablation JSON files found under {args.input}")

    model = tokenizer = None
    if args.stage in ("all", "generate"):
        model, tokenizer = load_model_and_tokenizer(
            args.base_model,
            cache_dir=MODEL_CACHE_PATH,
            token=HF_TOKEN,
        )

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if args.stage in ("all", "judge") and not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for the judge stage")

    summary_rows = []
    for input_path in input_paths:
        output_path = output_path_for(input_path)
        source_path = output_path if output_path.exists() else input_path
        data = load_json(source_path)
        if args.limit is not None:
            data = data[: args.limit]

        print(f"\nProcessing: {input_path}")
        if args.stage in ("all", "generate"):
            generated = generate_responses(data, model, tokenizer, args, output_path)
            print(f"Generated responses: {generated}")
        elif not output_path.exists():
            raise FileNotFoundError(f"Run the generate stage first: {output_path}")

        if args.stage in ("all", "judge"):
            judged = judge_responses(data, api_key, args, output_path)
            print(f"Judged items: {judged}")
        save_json_atomic(output_path, data)
        summary_rows.append(summarize(input_path, output_path, data))
        save_summary(args.input.resolve(), summary_rows)

    if model is not None:
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    print("\nRBPO vs original summary")
    for row in summary_rows:
        print(
            f"{row['config']}: win={row['win_rate']:.2%}, "
            f"tie={row['tie_rate']:.2%}, loss={row['loss_rate']:.2%} "
            f"(n={row['num_evaluated']})"
        )


if __name__ == "__main__":
    main()
