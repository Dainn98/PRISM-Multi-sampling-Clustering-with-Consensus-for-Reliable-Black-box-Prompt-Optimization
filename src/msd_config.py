import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from config import SEED
from helper import (
    MINILM_EMBEDDING_MODEL,
    clean_name,
    create_combined_name,
    eval_folder_name,
    evaluation_datasets,
    evaluator_models,
    base_llm_models,
    experiment_file_name,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SEEDS = [SEED + offset for offset in range(5)]
MSD_ROOT_NAME = "evaluation_msd"
MSD_ROOT = BASE_DIR / MSD_ROOT_NAME
EXPERIMENT_FILE = BASE_DIR / experiment_file_name
MSD_EMBEDDING_MODELS = [MINILM_EMBEDDING_MODEL]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_dir(seed, output_root=MSD_ROOT):
    return Path(output_root) / f"seed_{seed}"


def experiment_names(limit=None):
    if EXPERIMENT_FILE.exists():
        with EXPERIMENT_FILE.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        names = [
            create_combined_name(base_model, dataset, evaluator)
            for base_model in base_llm_models
            for dataset in evaluation_datasets
            for evaluator in evaluator_models
        ]
    return names[:limit] if limit else names


def source_dataset_for_experiment(experiment_name):
    for base_model in base_llm_models:
        for dataset in evaluation_datasets:
            for evaluator in evaluator_models:
                if create_combined_name(base_model, dataset, evaluator) == experiment_name:
                    return base_model, dataset, evaluator
    raise ValueError(f"Unknown experiment name: {experiment_name}")


def source_json_path(experiment_name):
    legacy_path = BASE_DIR / eval_folder_name / f"{experiment_name}.json"
    if legacy_path.exists():
        return legacy_path

    _, dataset, _ = source_dataset_for_experiment(experiment_name)
    candidates = [
        BASE_DIR / dataset,
        BASE_DIR / "testset" / dataset,
        BASE_DIR.parent / dataset,
        BASE_DIR.parent / "testset" / dataset,
    ]
    for dataset_path in candidates:
        if dataset_path.exists():
            return dataset_path

    raise FileNotFoundError(
        f"Cannot find source JSON for {experiment_name}. "
        f"Checked: {legacy_path}, "
        + ", ".join(str(path) for path in candidates)
    )


def normalize_items(raw_data):
    normalized = []
    for idx, item in enumerate(raw_data, start=1):
        normalized.append(
            {
                "id": item.get("id") or item.get("question_id") or item.get("idx") or idx,
                "ori_prompt": item.get("ori_prompt")
                or item.get("instruction")
                or item.get("prompt")
                or item.get("text"),
                "context": item.get("context"),
                "category": item.get("category"),
                "expected_response": item.get("expected_response")
                or item.get("output")
                or item.get("good_res")
                or item.get("response"),
                "bpo_prompt": item.get("bpo_prompt") or item.get("optimized_prompt"),
            }
        )
    return normalized


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def ensure_seed_input(
    seed,
    experiment_name,
    output_root=MSD_ROOT,
    force=False,
    embedding_model_name=None,
):
    embedding_model_name = embedding_model_name or MSD_EMBEDDING_MODELS[0]
    output_path = embedded_json_path(
        seed,
        embedding_model_name,
        experiment_name,
        output_root,
    )
    if output_path.exists() and not force:
        return output_path, load_json(output_path)

    raw_data = load_json(source_json_path(experiment_name))
    data = normalize_items(raw_data)
    save_json(output_path, data)
    return output_path, data


def parse_seed_args(description, include_force=True):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--output-root", default=str(MSD_ROOT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--verify-times", type=int, default=3)
    parser.add_argument("--aggregate-only", action="store_true")
    if include_force:
        parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.experiments:
        args.experiment_names = args.experiments
    else:
        args.experiment_names = experiment_names(args.limit)

    if args.all_seeds:
        args.seed_values = DEFAULT_SEEDS
    elif args.seeds:
        args.seed_values = args.seeds
    elif args.seed is not None:
        args.seed_values = [args.seed]
    else:
        args.seed_values = [DEFAULT_SEEDS[0]]

    args.output_root = Path(args.output_root)
    return args


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_std_ci(values):
    values = [float(v) for v in values]
    if not values:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    half_ci = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, std, mean - half_ci, mean + half_ci


def embedded_json_path(seed, embedding_model_name, experiment_name, output_root=MSD_ROOT):
    return (
        seed_dir(seed, output_root)
        / clean_name(embedding_model_name)
        / f"{experiment_name}.json"
    )
