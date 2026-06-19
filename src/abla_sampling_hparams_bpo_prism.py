import argparse
import csv
import gc
import json
import os
import shutil
from itertools import product

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from config import MODEL_CACHE_PATH, SEED, prompt_template_optimize
from helper import (
    BPO_MODEL,
    DEEPSEEK,
    DOLLY_EVAL,
    HF_TOKEN,
    LLAMA2_7B,
    M,
    MINILM_EMBEDDING_MODEL,
    create_combined_name,
    device,
    distance_thresholds,
    load_model_and_tokenizer,
    set_global_seed,
)
from step2_clustering_and_selecting import (
    compute_consensus_score,
    optimize_prompt_selection,
    prompt_clustering,
    representative_selection,
)
from utils import generate_batch


OUTPUT_ROOT = "ablation_sampling_hparams"
TEMPERATURES = [0.0, 0.1, 0.5, 0.9, 1.0]
TOP_PS = [0.0, 0.1, 0.5, 0.9, 1.0]
EPS_TOP_P = 1e-8


def parse_float_list(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def value_tag(prefix, value):
    return f"{prefix}{str(value).replace('.', 'p')}"


def sampling_kwargs(temperature, top_p):
    if temperature <= 0:
        return {
            "do_sample": False,
            "effective_temperature": None,
            "effective_top_p": None,
            "sampling_note": "temperature<=0: greedy decoding; top_p is ignored.",
        }

    effective_top_p = max(top_p, EPS_TOP_P)
    note = None
    if top_p <= 0:
        note = f"top_p<=0: replaced by EPS_TOP_P={EPS_TOP_P} because sampling requires a positive nucleus mass."

    return {
        "do_sample": True,
        "effective_temperature": temperature,
        "effective_top_p": effective_top_p,
        "sampling_note": note,
    }


def load_instruction_dataset(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = []
    for idx, item in enumerate(raw_data, start=1):
        data.append(
            {
                "id": item.get("id") or item.get("question_id") or item.get("idx") or idx,
                "ori_prompt": item.get("instruction") or item.get("prompt") or item.get("text"),
                "context": item.get("context"),
                "category": item.get("category"),
                "expected_response": item.get("output") or item.get("good_res") or item.get("response"),
            }
        )

    if limit is not None:
        data = data[:limit]
    return data


def pairwise_diversity(samples, embedding_model):
    if len(samples) < 2:
        return {
            "num_candidates": len(samples),
            "unique_candidates": len(set(samples)),
            "unique_ratio": float(len(set(samples))) / len(samples) if samples else 0.0,
            "mean_pairwise_cosine": None,
            "mean_pairwise_distance": None,
            "min_pairwise_cosine": None,
            "max_pairwise_cosine": None,
        }

    embeddings = embedding_model.encode(samples, convert_to_tensor=True)
    embeddings = F.normalize(embeddings, dim=1)
    sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    upper = sim_matrix[torch.triu_indices(len(samples), len(samples), offset=1).unbind()]

    mean_cosine = float(upper.mean().item())
    return {
        "num_candidates": len(samples),
        "unique_candidates": len(set(samples)),
        "unique_ratio": float(len(set(samples))) / len(samples),
        "mean_pairwise_cosine": mean_cosine,
        "mean_pairwise_distance": 1.0 - mean_cosine,
        "min_pairwise_cosine": float(upper.min().item()),
        "max_pairwise_cosine": float(upper.max().item()),
    }


def select_prism_bpo_prompt(item, embedding_model, distance_threshold):
    clusters, embeddings = prompt_clustering(
        "rbpo_paraphrases",
        item,
        embedding_model,
        M,
        distance_threshold,
    )
    if clusters is None or len(clusters) == 0:
        return None

    cluster_representatives, single_cluster = representative_selection(
        item,
        embedding_model,
        clusters,
        embeddings,
    )
    consensus_scores = compute_consensus_score(
        "rbpo_paraphrases",
        item,
        embedding_model,
        clusters,
        embeddings,
        cluster_representatives,
        single_cluster,
    )
    best_rep_idx = optimize_prompt_selection(
        "rbpo_paraphrases",
        item,
        clusters,
        embeddings,
        cluster_representatives,
        consensus_scores,
    )

    item["rbpo_clusters"] = [[item["rbpo_paraphrases"][idx] for idx in cluster] for cluster in clusters]
    item["rbpo_prompt"] = item["rbpo_paraphrases"][best_rep_idx]
    item["rbpo_cluster_representatives"] = [
        item["rbpo_paraphrases"][idx] for idx in cluster_representatives
    ]
    item["rbpo_consensus_scores"] = consensus_scores
    item["rbpo_selected_idx"] = best_rep_idx

    return {
        "num_clusters": len(clusters),
        "selected_idx": best_rep_idx,
        "single_cluster": bool(single_cluster),
    }


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_summary_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "temperature",
        "top_p",
        "effective_temperature",
        "effective_top_p",
        "num_items",
        "avg_unique_ratio",
        "avg_mean_pairwise_cosine",
        "avg_mean_pairwise_distance",
        "avg_num_clusters",
        "single_cluster_ratio",
        "output_path",
        "sampling_note",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_ignore_none(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def run_config(
    model,
    tokenizer,
    embedding_model,
    source_data,
    output_path,
    temperature,
    top_p,
    batch_size,
    max_new_tokens,
):
    cfg = sampling_kwargs(temperature, top_p)
    generation_kwargs = {}
    if cfg["effective_temperature"] is not None:
        generation_kwargs["temperature"] = cfg["effective_temperature"]
    if cfg["effective_top_p"] is not None:
        generation_kwargs["top_p"] = cfg["effective_top_p"]

    data = []
    for item in source_data:
        data.append(dict(item))

    prompts = [prompt_template_optimize.format(item["ori_prompt"]) for item in data]
    bpo_prompts = generate_batch(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        apply_chat_template=False,
        do_sample=cfg["do_sample"],
        device=device,
        **generation_kwargs,
    )

    for item, bpo_prompt in zip(data, bpo_prompts):
        item["bpo_prompt"] = bpo_prompt

    candidate_prompts = []
    mapping = []
    for idx, item in enumerate(data):
        for _ in range(M):
            candidate_prompts.append(prompt_template_optimize.format(item["ori_prompt"]))
            mapping.append(idx)

    rbpo_outputs = []
    for start in tqdm(
        range(0, len(candidate_prompts), batch_size),
        desc=f"Candidates temp={temperature}, top_p={top_p}",
    ):
        batch = candidate_prompts[start : start + batch_size]
        rbpo_outputs.extend(
            generate_batch(
                model,
                tokenizer,
                batch,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                apply_chat_template=False,
                do_sample=cfg["do_sample"],
                device=device,
                **generation_kwargs,
            )
        )

    assert len(mapping) == len(rbpo_outputs), "Mismatch between generated candidates and mapping."
    for idx, output in zip(mapping, rbpo_outputs):
        data[idx].setdefault("rbpo_paraphrases", []).append(output)

    distance_threshold = distance_thresholds[MINILM_EMBEDDING_MODEL]
    diversity_rows = []
    for item in tqdm(data, desc=f"PRISM selection temp={temperature}, top_p={top_p}"):
        diversity = pairwise_diversity(item["rbpo_paraphrases"], embedding_model)
        selection = select_prism_bpo_prompt(item, embedding_model, distance_threshold)
        item["sampling_hparams"] = {
            "temperature": temperature,
            "top_p": top_p,
            "effective_temperature": cfg["effective_temperature"],
            "effective_top_p": cfg["effective_top_p"],
            "do_sample": cfg["do_sample"],
            "note": cfg["sampling_note"],
        }
        item["candidate_diversity"] = diversity
        if selection is not None:
            item["candidate_diversity"].update(selection)
        diversity_rows.append(item["candidate_diversity"])

    write_json(output_path, data)
    return {
        "temperature": temperature,
        "top_p": top_p,
        "effective_temperature": cfg["effective_temperature"],
        "effective_top_p": cfg["effective_top_p"],
        "num_items": len(data),
        "avg_unique_ratio": mean_ignore_none([x["unique_ratio"] for x in diversity_rows]),
        "avg_mean_pairwise_cosine": mean_ignore_none([x["mean_pairwise_cosine"] for x in diversity_rows]),
        "avg_mean_pairwise_distance": mean_ignore_none([x["mean_pairwise_distance"] for x in diversity_rows]),
        "avg_num_clusters": mean_ignore_none([x.get("num_clusters") for x in diversity_rows]),
        "single_cluster_ratio": mean_ignore_none(
            [1.0 if x.get("single_cluster") else 0.0 for x in diversity_rows if "single_cluster" in x]
        ),
        "output_path": output_path,
        "sampling_note": cfg["sampling_note"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ablate BPO/RBPO prompt sampling hyperparameters and measure candidate diversity."
    )
    parser.add_argument("--temperatures", default=",".join(map(str, TEMPERATURES)))
    parser.add_argument("--top-ps", default=",".join(map(str, TOP_PS)))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    set_global_seed(SEED)
    torch.cuda.empty_cache()
    gc.collect()

    source_path = os.path.join("testset", DOLLY_EVAL)
    source_data = load_instruction_dataset(source_path, limit=args.limit)
    run_name = create_combined_name(LLAMA2_7B, DOLLY_EVAL, DEEPSEEK)
    embedding_tag = "all-MiniLM-L12-v2"

    print("===== Sampling hyperparameter ablation: PRISM(BPO) vs BPO =====")
    print(f"Dataset: {DOLLY_EVAL}")
    print(f"Base LLM tag: {LLAMA2_7B}")
    print(f"Prompt optimizer: {BPO_MODEL}")
    print(f"Embedding: {MINILM_EMBEDDING_MODEL}")
    print(f"Items: {len(source_data)}")

    bpo_model, bpo_tokenizer = load_model_and_tokenizer(
        BPO_MODEL,
        device_map="auto",
        cache_dir=MODEL_CACHE_PATH,
        token=HF_TOKEN,
    )
    embedding_model = SentenceTransformer(
        MINILM_EMBEDDING_MODEL,
        device=device,
        cache_folder=MODEL_CACHE_PATH,
    )

    summary_rows = []
    for temperature, top_p in product(parse_float_list(args.temperatures), parse_float_list(args.top_ps)):
        config_name = f"{value_tag('temp_', temperature)}_{value_tag('top_p_', top_p)}"
        output_path = os.path.join(
            args.output_root,
            embedding_tag,
            run_name,
            config_name,
            f"{run_name}.json",
        )
        print(f"\nProcessing {config_name} -> {output_path}")
        summary_rows.append(
            run_config(
                bpo_model,
                bpo_tokenizer,
                embedding_model,
                source_data,
                output_path,
                temperature,
                top_p,
                args.batch_size,
                args.max_new_tokens,
            )
        )
        write_summary_csv(
            os.path.join(args.output_root, embedding_tag, run_name, "summary.csv"),
            summary_rows,
        )
        write_json(
            os.path.join(args.output_root, embedding_tag, run_name, "summary.json"),
            summary_rows,
        )

    del embedding_model
    del bpo_model
    del bpo_tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    if not args.keep_cache and os.path.exists(MODEL_CACHE_PATH):
        shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)


if __name__ == "__main__":
    main()
