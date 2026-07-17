# python src/msd_merge_po_outputs.py --all-seeds --force
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

from helper import clean_name
from msd_config import (
    DEFAULT_SEEDS,
    MSD_EMBEDDING_MODELS,
    MSD_ROOT,
    experiment_names,
    load_json,
    save_json,
)


PO_MODEL_FIELD_PREFIXES = {
    "bpo": ("bpo_", "rbpo_"),
    "generic": ("generic_", "rgeneric_"),
    "mepo": ("mepo_", "rmepo_"),
}

COMMON_KEYS = [
    "id",
    "ori_prompt",
    "context",
    "category",
    "expected_response",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge independent MSD PO-model outputs into one file per seed, "
            "embedding model, and experiment."
        )
    )
    parser.add_argument("--input-root", default=str(MSD_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--embedding-models", nargs="+", default=None)
    parser.add_argument(
        "--po-models",
        nargs="+",
        choices=sorted(PO_MODEL_FIELD_PREFIXES),
        default=sorted(PO_MODEL_FIELD_PREFIXES),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.all_seeds:
        args.seed_values = DEFAULT_SEEDS
    elif args.seeds:
        args.seed_values = args.seeds
    elif args.seed is not None:
        args.seed_values = [args.seed]
    else:
        args.seed_values = [DEFAULT_SEEDS[0]]

    args.experiment_names = args.experiments or experiment_names(args.limit)
    args.embedding_model_names = args.embedding_models or MSD_EMBEDDING_MODELS
    args.input_root = Path(args.input_root)
    args.output_root = Path(args.output_root) if args.output_root else args.input_root / "merge"
    return args


def po_json_path(root, po_model, seed, embedding_model_name, experiment_name):
    return (
        Path(root)
        / po_model
        / f"seed_{seed}"
        / clean_name(embedding_model_name)
        / f"{experiment_name}.json"
    )


def merged_json_path(root, seed, embedding_model_name, experiment_name):
    return (
        Path(root)
        / f"seed_{seed}"
        / clean_name(embedding_model_name)
        / f"{experiment_name}.json"
    )


def item_id(item, fallback_index):
    return item.get("id") or fallback_index


def copy_common_fields(target, source):
    for key in COMMON_KEYS:
        if key in source and key not in target:
            target[key] = deepcopy(source[key])


def copy_po_fields(target, source, po_model):
    prefixes = PO_MODEL_FIELD_PREFIXES[po_model]
    for key, value in source.items():
        if key in COMMON_KEYS:
            continue
        if key.startswith(prefixes):
            target[key] = deepcopy(value)


def validate_common_fields(target, source, stats, path, sample_id):
    for key in ("ori_prompt", "expected_response"):
        if key not in target or key not in source:
            continue
        if target[key] != source[key]:
            stats[f"{key}_mismatch"] += 1
            print(
                f"Warning: {key} mismatch for id={sample_id} in {path}. "
                "Keeping the first value."
            )


def merge_experiment(seed, embedding_model_name, experiment_name, args):
    output_path = merged_json_path(
        args.output_root,
        seed,
        embedding_model_name,
        experiment_name,
    )
    if output_path.exists() and not args.force:
        print(f"Exists, skipping: {output_path}")
        return Counter({"skipped_existing": 1})

    stats = Counter()
    merged_by_id = {}
    id_order = []

    for po_model in args.po_models:
        path = po_json_path(
            args.input_root,
            po_model,
            seed,
            embedding_model_name,
            experiment_name,
        )
        if not path.exists():
            stats[f"missing_{po_model}_file"] += 1
            print(f"Missing {po_model} file, skipping: {path}")
            continue

        data = load_json(path)
        if not isinstance(data, list):
            stats[f"invalid_{po_model}_file"] += 1
            print(f"Invalid {po_model} file (expected list), skipping: {path}")
            continue

        for index, source_item in enumerate(data, start=1):
            sample_id = item_id(source_item, index)
            if sample_id not in merged_by_id:
                merged_by_id[sample_id] = {}
                id_order.append(sample_id)

            target_item = merged_by_id[sample_id]
            copy_common_fields(target_item, source_item)
            validate_common_fields(target_item, source_item, stats, path, sample_id)
            copy_po_fields(target_item, source_item, po_model)

        stats[f"loaded_{po_model}_files"] += 1
        stats[f"loaded_{po_model}_items"] += len(data)

    if not merged_by_id:
        stats["empty_merge"] += 1
        print(
            "No input data found for "
            f"seed={seed}, embed={clean_name(embedding_model_name)}, "
            f"experiment={experiment_name}"
        )
        return stats

    merged_data = [merged_by_id[sample_id] for sample_id in id_order]
    save_json(output_path, merged_data)
    stats["written_files"] += 1
    stats["written_items"] += len(merged_data)
    print(f"Saved {output_path} (items={len(merged_data)})")
    return stats


def main():
    args = parse_args()
    all_stats = Counter()
    for seed in args.seed_values:
        print(f"\n===== MSD MERGE | seed={seed} =====")
        for embedding_model_name in args.embedding_model_names:
            print(f"Embedding model: {clean_name(embedding_model_name)}")
            for experiment_name in args.experiment_names:
                all_stats.update(
                    merge_experiment(
                        seed,
                        embedding_model_name,
                        experiment_name,
                        args,
                    )
                )

    print("\nMSD merge summary:")
    for key in sorted(all_stats):
        print(f"  {key}: {all_stats[key]}")


if __name__ == "__main__":
    main()
