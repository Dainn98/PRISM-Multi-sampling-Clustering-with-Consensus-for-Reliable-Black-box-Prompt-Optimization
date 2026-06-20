"""Factorial ablation of Aspect Expansion and Instructional Framing on Dolly.

The public entry point launches every GPU-heavy phase in a fresh subprocess:

1. BPO candidate generation for AE, IF, and AE+IF.
2. PRISM clustering and representative selection with all-MiniLM-L12-v2.
3. Deterministic response generation with Llama-2-7b-chat-hf.
4. Position-balanced pairwise evaluation against the original prompt.

Intermediate JSON files are checkpoints, so an interrupted experiment can be
resumed without repeating completed samples.  Process isolation ensures model
memory is returned to the OS between phases and substantially reduces OOM risk.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence


STRATEGIES = ("ae", "if", "ae_if")
ALL_METHODS = ("ori",) + STRATEGIES
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

STRATEGY_INSTRUCTIONS = {
    "ae": """Rewrite the original prompt using ASPECT EXPANSION ONLY.
Add relevant dimensions, details, constraints, or considerations that make the
request more complete and useful. Preserve the original intent. Do not add a
role, step-by-step procedure, explicit output template, or meta-instructions.
Return only the rewritten prompt, with no analysis or label.""",
    "if": """Rewrite the original prompt using INSTRUCTIONAL FRAMING ONLY.
Improve how the task is instructed through an appropriate role, explicit
directions, reasoning guidance, or a clear output format. Preserve the original
intent and do not introduce new substantive topics, requirements, facts, or
aspects. Return only the rewritten prompt, with no analysis or label.""",
    "ae_if": """Rewrite the original prompt using BOTH ASPECT EXPANSION AND
INSTRUCTIONAL FRAMING. Add relevant dimensions, details, constraints, or
considerations, and also provide an appropriate role, explicit directions,
reasoning guidance, or clear output format. Preserve the original intent.
Return only the rewritten prompt, with no analysis or label.""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="src/testset/instruction-following/dolly_eval.json",
    )
    parser.add_argument(
        "--output-dir", default="src/extra_results/causal_strategies_dolly"
    )
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-model", default="THUDM/BPO")
    parser.add_argument("--base-model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L12-v2"
    )
    parser.add_argument("--distance-threshold", type=float, default=0.05)
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    parser.add_argument("--response-batch-size", type=int, default=1)
    parser.add_argument("--candidate-max-new-tokens", type=int, default=512)
    parser.add_argument("--response-max-new-tokens", type=int, default=1024)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--evaluator-model", default="deepseek-chat")
    parser.add_argument("--api-timeout", type=int, default=120)
    parser.add_argument("--api-retries", type=int, default=5)
    parser.add_argument(
        "--keep-model-cache",
        action="store_true",
        help="Keep MODEL_CACHE_PATH after each phase instead of deleting it",
    )
    parser.add_argument(
        "--phase",
        choices=("candidates", "select", "responses", "evaluate"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--config-json", help=argparse.SUPPRESS)
    return parser.parse_args()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cleanup_memory(*objects: object) -> None:
    del objects
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def remove_model_cache(config: dict, phase: str) -> None:
    """Delete downloaded model files after a model-backed phase."""
    if config.get("keep_model_cache", False):
        print(f"Keeping model cache after phase '{phase}'.")
        return

    from config import MODEL_CACHE_PATH

    cache_path = Path(MODEL_CACHE_PATH).resolve()
    workspace = Path.cwd().resolve()
    if cache_path == workspace or workspace not in cache_path.parents:
        raise RuntimeError(
            f"Refusing to delete unsafe MODEL_CACHE_PATH: {cache_path}"
        )
    if cache_path.exists():
        print(f"Removing model cache after phase '{phase}': {cache_path}")
        shutil.rmtree(cache_path, ignore_errors=True)


def artifact(config: dict, name: str) -> Path:
    return Path(config["output_dir"]) / name


def normalize_source(data: object, count: int) -> list[dict]:
    if not isinstance(data, list):
        raise ValueError("Dolly dataset must be a JSON list")
    rows = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        prompt = (
            item.get("ori_prompt")
            or item.get("instruction")
            or item.get("text")
            or item.get("prompt")
        )
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        rows.append(
            {
                "id": item.get(
                    "question_id", item.get("id", item.get("idx", index))
                ),
                "ori_prompt": prompt.strip(),
                "category": item.get("category"),
                "context": item.get("context"),
                "reference_response": item.get("response"),
            }
        )
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f"Found {len(rows)} usable prompts; requested {count}")
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_causal_lm(model_name: str):
    import torch
    from dotenv import load_dotenv
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from config import MODEL_CACHE_PATH

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_PATH,
        token=token,
        legacy=False,
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_PATH,
        token=token,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    # Some prompt-optimization checkpoints (including older BPO configs) store
    # ``return_dict=false``. Newer Transformers LlamaForCausalLM implementations
    # access ``outputs.last_hidden_state`` unconditionally, so tuple output from
    # the backbone crashes during generation. Keep the backbone/model contract
    # explicit and consistent across Transformers versions.
    model.config.return_dict = True
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.return_dict = True
    model.config.pad_token_id = tokenizer.pad_token_id
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tokenizer


def model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def generate_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    seed: int,
) -> list[str]:
    import torch

    results = []
    device = model_input_device(model)
    for start in range(0, len(prompts), batch_size):
        set_seed(seed + start)
        texts = prompts[start : start + batch_size]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        input_width = inputs["input_ids"].shape[1]
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs.update(temperature=0.9, top_p=0.9)
        with torch.inference_mode():
            outputs = model.generate(**inputs, **kwargs)
        for output in outputs:
            text = tokenizer.decode(
                output[input_width:], skip_special_tokens=True
            ).strip()
            results.append(text)
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def candidate_input(ori_prompt: str, strategy: str) -> str:
    return (
        "[INST] You are an expert prompt editor.\n\n"
        f"{STRATEGY_INSTRUCTIONS[strategy]}\n\n"
        f"Original prompt:\n{ori_prompt}\n\nRewritten prompt: [/INST]"
    )


def phase_candidates(config: dict) -> None:
    output = artifact(config, "01_candidates.json")
    if output.exists():
        rows = load_json(output)
        if not isinstance(rows, list):
            raise ValueError(f"Invalid checkpoint: {output}")
    else:
        rows = normalize_source(load_json(Path(config["dataset"])), config["num_samples"])

    model = tokenizer = None
    try:
        model, tokenizer = load_causal_lm(config["candidate_model"])
        for index, row in enumerate(rows):
            changed = False
            for strategy in STRATEGIES:
                key = f"{strategy}_candidates"
                existing = row.get(key)
                if isinstance(existing, list) and len(existing) == config["m"]:
                    continue
                prompts = [candidate_input(row["ori_prompt"], strategy)] * config["m"]
                row[key] = generate_batch(
                    model,
                    tokenizer,
                    prompts,
                    config["candidate_batch_size"],
                    config["candidate_max_new_tokens"],
                    True,
                    config["seed"] + index * 1000 + STRATEGIES.index(strategy) * 100,
                )
                changed = True
                atomic_write_json(output, rows)
                print(
                    f"Candidates {index + 1}/{len(rows)} {strategy}: "
                    f"{len(row[key])}/{config['m']}"
                )
            if changed:
                atomic_write_json(output, rows)
    finally:
        model = tokenizer = None
        cleanup_memory()
        remove_model_cache(config, "candidates")


def phase_select(config: dict) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    from config import MODEL_CACHE_PATH
    from step2_clustering_and_selecting import (
        compute_consensus_score,
        optimize_prompt_selection,
        prompt_clustering,
        representative_selection,
    )

    source = artifact(config, "01_candidates.json")
    output = artifact(config, "02_selected.json")
    rows = load_json(output if output.exists() else source)
    if not isinstance(rows, list):
        raise ValueError("Candidate checkpoint must be a JSON list")

    embed_model = None
    try:
        embed_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        embed_model = SentenceTransformer(
            config["embedding_model"],
            device=embed_device,
            cache_folder=MODEL_CACHE_PATH,
        )
        for index, row in enumerate(rows):
            for strategy in STRATEGIES:
                prompt_key = f"{strategy}_prompt"
                if isinstance(row.get(prompt_key), str) and row[prompt_key].strip():
                    continue
                candidate_key = f"{strategy}_candidates"
                clusters, embeddings = prompt_clustering(
                    candidate_key,
                    row,
                    embed_model,
                    config["m"],
                    config["distance_threshold"],
                )
                if not clusters or embeddings is None:
                    raise ValueError(f"No clusters for item {row.get('id')} {strategy}")
                representatives, single_cluster = representative_selection(
                    row, embed_model, clusters, embeddings
                )
                scores = compute_consensus_score(
                    candidate_key,
                    row,
                    embed_model,
                    clusters,
                    embeddings,
                    representatives,
                    single_cluster,
                )
                best = optimize_prompt_selection(
                    candidate_key, row, clusters, embeddings, representatives, scores
                )
                row[prompt_key] = row[candidate_key][best]
                row[f"{strategy}_selected_index"] = best
                row[f"{strategy}_clusters"] = clusters
                row[f"{strategy}_representative_indices"] = representatives
                row[f"{strategy}_consensus_scores"] = scores
                del embeddings
                atomic_write_json(output, rows)
                print(f"Selection {index + 1}/{len(rows)} {strategy}")
        atomic_write_json(output, rows)
    finally:
        embed_model = None
        cleanup_memory()
        remove_model_cache(config, "select")


def format_llama_prompt(tokenizer, prompt: str, context: str | None) -> str:
    content = prompt
    if isinstance(context, str) and context.strip():
        content = f"Context:\n{context.strip()}\n\nQuestion:\n{prompt}"
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"[INST] {content} [/INST]"


def phase_responses(config: dict) -> None:
    source = artifact(config, "02_selected.json")
    output = artifact(config, "03_responses.json")
    rows = load_json(output if output.exists() else source)
    if not isinstance(rows, list):
        raise ValueError("Selection checkpoint must be a JSON list")

    model = tokenizer = None
    try:
        model, tokenizer = load_causal_lm(config["base_model"])
        for index, row in enumerate(rows):
            missing = [
                method
                for method in ALL_METHODS
                if not isinstance(row.get(f"{method}_response"), str)
                or not row[f"{method}_response"].strip()
            ]
            if not missing:
                continue
            formatted = [
                format_llama_prompt(
                    tokenizer,
                    row["ori_prompt"] if method == "ori" else row[f"{method}_prompt"],
                    row.get("context"),
                )
                for method in missing
            ]
            responses = generate_batch(
                model,
                tokenizer,
                formatted,
                config["response_batch_size"],
                config["response_max_new_tokens"],
                False,
                config["seed"],
            )
            for method, response in zip(missing, responses):
                row[f"{method}_response"] = response
            atomic_write_json(output, rows)
            print(f"Responses {index + 1}/{len(rows)}: {', '.join(missing)}")
    finally:
        model = tokenizer = None
        cleanup_memory()
        remove_model_cache(config, "responses")


def evaluation_prompt(
    prompt_a: str, response_a: str, prompt_b: str, response_b: str
) -> str:
    criteria = ",\n".join(f'    "{criterion}": 0.0' for criterion in CRITERIA)
    return f"""Score two responses independently. Judge each response only against
its own prompt; do not reward or punish it merely for being longer. Use the same
scale: 0.0 is unusable, 0.5 is moderate, and 1.0 is excellent. Each score must
be a number from 0.0 to 1.0.

Prompt_A:\n{prompt_a}\n\nResponse_A:\n{response_a}

Prompt_B:\n{prompt_b}\n\nResponse_B:\n{response_b}

Return JSON only with exactly this structure:
{{
  "response_A": {{
{criteria}
  }},
  "response_B": {{
{criteria}
  }}
}}
"""


def extract_json(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Evaluator output is not an object")
    return value


def validate_scores(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("Score block is not an object")
    normalized = {}
    for criterion in CRITERIA:
        score = value.get(criterion)
        if not isinstance(score, (int, float)):
            raise ValueError(f"Missing numeric score: {criterion}")
        normalized[criterion] = min(1.0, max(0.0, float(score)))
    return normalized


def call_evaluator(config: dict, user_prompt: str) -> dict:
    import requests
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    payload = {
        "model": config["evaluator_model"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict, consistent response-quality judge. "
                    "Return one valid JSON object and no other text."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(config["api_retries"]):
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=config["api_timeout"],
            )
            response.raise_for_status()
            parsed = extract_json(response.json()["choices"][0]["message"]["content"])
            return {
                "response_A": validate_scores(parsed.get("response_A")),
                "response_B": validate_scores(parsed.get("response_B")),
            }
        except Exception as error:
            last_error = error
            if attempt + 1 < config["api_retries"]:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Evaluator failed after retries: {last_error}")


def mean_score(scores: dict[str, float]) -> float:
    return statistics.fmean(scores.values())


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_summaries(evaluations: Sequence[dict], output_dir: Path) -> None:
    comparison_rows = []
    criterion_rows = []
    sample_deltas: dict[str, dict[str, float]] = {strategy: {} for strategy in STRATEGIES}
    for strategy in STRATEGIES:
        group = [row for row in evaluations if row["strategy"] == strategy]
        by_item: dict[str, list[dict]] = {}
        for row in group:
            by_item.setdefault(str(row["id"]), []).append(row)
        strategy_values = [
            statistics.fmean(mean_score(row["strategy_scores"]) for row in item_rows)
            for item_rows in by_item.values()
        ]
        ori_values = [
            statistics.fmean(mean_score(row["ori_scores"]) for row in item_rows)
            for item_rows in by_item.values()
        ]
        deltas = [a - b for a, b in zip(strategy_values, ori_values)]
        sample_deltas[strategy] = {
            item_id: statistics.fmean(
                mean_score(row["strategy_scores"]) - mean_score(row["ori_scores"])
                for row in item_rows
            )
            for item_id, item_rows in by_item.items()
        }
        wins = sum(delta > 0.01 for delta in deltas)
        losses = sum(delta < -0.01 for delta in deltas)
        draws = len(deltas) - wins - losses
        comparison_rows.append(
            {
                "strategy": strategy,
                "num_judgments": len(group),
                "num_samples": len(by_item),
                "strategy_mean_score": statistics.fmean(strategy_values),
                "ori_mean_score": statistics.fmean(ori_values),
                "mean_paired_delta": statistics.fmean(deltas),
                "paired_delta_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate_excluding_draws": wins / (wins + losses) if wins + losses else 0.0,
            }
        )
        for criterion in CRITERIA:
            strategy_criterion = [
                statistics.fmean(row["strategy_scores"][criterion] for row in item_rows)
                for item_rows in by_item.values()
            ]
            ori_criterion = [
                statistics.fmean(row["ori_scores"][criterion] for row in item_rows)
                for item_rows in by_item.values()
            ]
            criterion_deltas = [
                a - b for a, b in zip(strategy_criterion, ori_criterion)
            ]
            criterion_rows.append(
                {
                    "strategy": strategy,
                    "criterion": criterion,
                    "strategy_mean": statistics.fmean(strategy_criterion),
                    "ori_mean": statistics.fmean(ori_criterion),
                    "mean_paired_delta": statistics.fmean(criterion_deltas),
                }
            )
    write_csv(output_dir / "comparison_summary.csv", comparison_rows)
    write_csv(output_dir / "criterion_summary.csv", criterion_rows)

    common_ids = set.intersection(
        *(set(sample_deltas[strategy]) for strategy in STRATEGIES)
    )
    ae_values = [sample_deltas["ae"][item_id] for item_id in common_ids]
    if_values = [sample_deltas["if"][item_id] for item_id in common_ids]
    ae_if_values = [sample_deltas["ae_if"][item_id] for item_id in common_ids]
    interactions = [
        ae_if - ae - instructional
        for ae_if, ae, instructional in zip(ae_if_values, ae_values, if_values)
    ]
    ae_main_effects = [
        0.5 * (ae + ae_if - instructional)
        for ae_if, ae, instructional in zip(ae_if_values, ae_values, if_values)
    ]
    if_main_effects = [
        0.5 * (instructional + ae_if - ae)
        for ae_if, ae, instructional in zip(ae_if_values, ae_values, if_values)
    ]
    effects = {
        "num_samples": len(common_ids),
        "AE_vs_ORI": statistics.fmean(ae_values),
        "IF_vs_ORI": statistics.fmean(if_values),
        "AE_IF_vs_ORI": statistics.fmean(ae_if_values),
        "AE_main_effect": statistics.fmean(ae_main_effects),
        "IF_main_effect": statistics.fmean(if_main_effects),
        "AE_x_IF_interaction": statistics.fmean(interactions),
        "AE_x_IF_interaction_std": (
            statistics.stdev(interactions) if len(interactions) > 1 else 0.0
        ),
        "interpretation": (
            "Interaction = (AE+IF - ORI) - (AE - ORI) - (IF - ORI). "
            "Positive values indicate synergy beyond additive individual effects."
        ),
    }
    atomic_write_json(output_dir / "factorial_effects.json", effects)


def phase_evaluate(config: dict) -> None:
    rows = load_json(artifact(config, "03_responses.json"))
    if not isinstance(rows, list):
        raise ValueError("Response checkpoint must be a JSON list")
    output = artifact(config, "04_evaluations.json")
    evaluations = load_json(output) if output.exists() else []
    if not isinstance(evaluations, list):
        raise ValueError("Evaluation checkpoint must be a JSON list")
    completed = {
        (str(row["id"]), row["strategy"], int(row["run"])) for row in evaluations
    }

    for item_index, row in enumerate(rows):
        for strategy in STRATEGIES:
            for run in range(1, config["eval_repeats"] + 1):
                key = (str(row["id"]), strategy, run)
                if key in completed:
                    continue
                strategy_first = (item_index + STRATEGIES.index(strategy) + run) % 2 == 0
                strategy_prompt = row[f"{strategy}_prompt"]
                if strategy_first:
                    prompt_a, response_a = strategy_prompt, row[f"{strategy}_response"]
                    prompt_b, response_b = row["ori_prompt"], row["ori_response"]
                else:
                    prompt_a, response_a = row["ori_prompt"], row["ori_response"]
                    prompt_b, response_b = strategy_prompt, row[f"{strategy}_response"]
                judged = call_evaluator(
                    config, evaluation_prompt(prompt_a, response_a, prompt_b, response_b)
                )
                strategy_scores = judged["response_A" if strategy_first else "response_B"]
                ori_scores = judged["response_B" if strategy_first else "response_A"]
                evaluations.append(
                    {
                        "id": row["id"],
                        "strategy": strategy,
                        "run": run,
                        "strategy_position": "A" if strategy_first else "B",
                        "strategy_scores": strategy_scores,
                        "ori_scores": ori_scores,
                        "strategy_mean": mean_score(strategy_scores),
                        "ori_mean": mean_score(ori_scores),
                        "paired_delta": mean_score(strategy_scores) - mean_score(ori_scores),
                    }
                )
                completed.add(key)
                atomic_write_json(output, evaluations)
                print(
                    f"Evaluation {item_index + 1}/{len(rows)} {strategy} run={run}"
                )
    build_summaries(evaluations, Path(config["output_dir"]))


def run_phase(phase: str, config: dict) -> None:
    set_seed(config["seed"])
    functions = {
        "candidates": phase_candidates,
        "select": phase_select,
        "responses": phase_responses,
        "evaluate": phase_evaluate,
    }
    functions[phase](config)


def config_from_args(args: argparse.Namespace) -> dict:
    config = vars(args).copy()
    config.pop("phase", None)
    config.pop("config_json", None)
    config["dataset"] = str(Path(config["dataset"]).resolve())
    config["output_dir"] = str(Path(config["output_dir"]).resolve())
    return config


def validate_config(config: dict) -> None:
    positive = (
        "num_samples",
        "m",
        "candidate_batch_size",
        "response_batch_size",
        "candidate_max_new_tokens",
        "response_max_new_tokens",
        "eval_repeats",
        "api_timeout",
        "api_retries",
    )
    for key in positive:
        if int(config[key]) < 1:
            raise SystemExit(f"--{key.replace('_', '-')} must be positive")
    if config["m"] < 2:
        raise SystemExit("--m must be at least 2 for clustering")


def main() -> int:
    args = parse_args()
    if args.phase:
        if not args.config_json:
            raise SystemExit("Internal --phase requires --config-json")
        config = load_json(Path(args.config_json))
        if not isinstance(config, dict):
            raise SystemExit("Invalid config JSON")
        run_phase(args.phase, config)
        return 0

    config = config_from_args(args)
    validate_config(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "experiment_config.json"
    atomic_write_json(config_path, config)

    script = str(Path(__file__).resolve())
    for phase in ("candidates", "select", "responses", "evaluate"):
        print(f"\n{'=' * 20} PHASE: {phase.upper()} {'=' * 20}", flush=True)
        subprocess.run(
            [
                sys.executable,
                script,
                "--phase",
                phase,
                "--config-json",
                str(config_path.resolve()),
            ],
            check=True,
            cwd=str(Path.cwd()),
        )
        cleanup_memory()
    print(f"\nCompleted. Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
