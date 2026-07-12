import gc
import json
import os
import shutil

import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from config import MODEL_CACHE_PATH
from helper import IMP_ENC, M, clean_name, device, distance_thresholds
from msd_config import (
    MSD_EMBEDDING_MODELS,
    embedded_json_path,
    ensure_seed_input,
    parse_seed_args,
    save_json,
    set_seed,
)


METHOD_KEYS = {
    "rbpo_paraphrases": "rbpo",
    "rmepo_paraphrases": "rmepo",
    "generic_paraphrases": "rgeneric",
}


def cluster_prompts(prompts, embedding_model, distance_threshold):
    embeddings = embedding_model.encode(prompts, convert_to_tensor=True)
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    ).fit_predict(embeddings.cpu().numpy())

    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)
    return list(clusters.values()), embeddings


def select_representatives(item, clusters, embeddings, embedding_model):
    original_embedding = embedding_model.encode(
        [item.get("ori_prompt", "")],
        convert_to_tensor=True,
    )[0]
    representatives = []
    for cluster in clusters:
        if len(cluster) == 1:
            representatives.append(cluster[0])
            continue
        cluster_embeddings = torch.stack([embeddings[idx] for idx in cluster])
        similarities = util.pytorch_cos_sim(original_embedding, cluster_embeddings)[0]
        sorted_indices = torch.argsort(similarities)
        representatives.append(cluster[sorted_indices[len(sorted_indices) // 2].item()])
    return representatives, original_embedding


def consensus_scores(representatives, embeddings, original_embedding):
    if len(representatives) == 1:
        return [0.0]
    scores = []
    for rep_idx in representatives:
        rep_embed = embeddings[rep_idx]
        score = 0.0
        for other_idx in representatives:
            if rep_idx != other_idx:
                score += util.pytorch_cos_sim(rep_embed, embeddings[other_idx]).item()
        score -= util.pytorch_cos_sim(rep_embed, original_embedding).item() * IMP_ENC
        scores.append(score)
    return scores


def apply_selection(item, source_key, output_key, embedding_model, distance_threshold):
    samples = item.get(source_key, [])[:M]
    if not samples:
        return
    if len(samples) == 1:
        item[f"{output_key}_clusters"] = [[samples[0]]]
        item[f"{output_key}_cluster_representatives"] = [samples[0]]
        item[f"{output_key}_consensus_scores"] = [0.0]
        item[f"{output_key}_prompt"] = samples[0]
        return

    clusters, embeddings = cluster_prompts(samples, embedding_model, distance_threshold)
    representatives, original_embedding = select_representatives(
        item,
        clusters,
        embeddings,
        embedding_model,
    )
    scores = consensus_scores(representatives, embeddings, original_embedding)
    best_position = max(range(len(scores)), key=scores.__getitem__)
    best_idx = representatives[best_position]

    item[f"{output_key}_clusters"] = [[samples[idx] for idx in cluster] for cluster in clusters]
    item[f"{output_key}_cluster_representative_indices"] = representatives
    item[f"{output_key}_cluster_representatives"] = [samples[idx] for idx in representatives]
    item[f"{output_key}_consensus_scores"] = scores
    item[f"{output_key}_selected_idx"] = best_idx
    item[f"{output_key}_prompt"] = samples[best_idx]


def run_seed(seed, args):
    set_seed(seed)
    print(f"\n===== MSD STEP 2 | seed={seed} =====")
    for embedding_model_name in MSD_EMBEDDING_MODELS:
        distance_threshold = distance_thresholds.get(embedding_model_name)
        if distance_threshold is None:
            raise ValueError(f"No distance threshold for {embedding_model_name}")

        embedding_model = SentenceTransformer(
            embedding_model_name,
            device=device,
            cache_folder=MODEL_CACHE_PATH,
        )
        print(
            f"Using embedding model {embedding_model_name} "
            f"(threshold={distance_threshold})"
        )

        for experiment_name in args.experiment_names:
            _, source_data = ensure_seed_input(
                seed,
                experiment_name,
                args.output_root,
                embedding_model_name=embedding_model_name,
            )
            data = json.loads(json.dumps(source_data, ensure_ascii=False))
            progress = tqdm(
                data,
                desc=f"Cluster seed={seed} {clean_name(embedding_model_name)} {experiment_name}",
                unit="item",
            )
            for item in progress:
                for source_key, output_key in METHOD_KEYS.items():
                    progress.set_postfix(method=output_key)
                    apply_selection(
                        item,
                        source_key,
                        output_key,
                        embedding_model,
                        distance_threshold,
                    )
                for key in ["bpo", "rbpo", "mepo", "rmepo", "generic", "rgeneric"]:
                    item.pop(f"{key}_response", None)

            output_path = embedded_json_path(
                seed,
                embedding_model_name,
                experiment_name,
                args.output_root,
            )
            save_json(output_path, data)
            print(f"Saved {output_path}")

        del embedding_model
        torch.cuda.empty_cache()
        gc.collect()
        if os.path.exists(MODEL_CACHE_PATH):
            shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)


def main():
    args = parse_seed_args("Cluster and select PRISM prompts for MSD seeds.")
    for seed in args.seed_values:
        run_seed(seed, args)


if __name__ == "__main__":
    main()
