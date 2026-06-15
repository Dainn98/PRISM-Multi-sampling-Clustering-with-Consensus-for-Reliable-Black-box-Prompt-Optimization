import gc
import json
import os
import re
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from config import MODEL_CACHE_PATH
from helper import (
    GENERIC_LLM_REWRITER,
    IMP_ENC,
    M,
    clean_name,
    device,
    distance_thresholds,
    embedding_models,
    eval_folder_name,
    experiment_file_name,
    load_model_and_tokenizer,
)


BASE_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = BASE_DIR / eval_folder_name
EXPERIMENT_FILE = BASE_DIR / experiment_file_name
CHECKPOINT_DIR = EVALUATION_DIR / "generic_rewriter"

GENERIC_REWRITE_COUNT = int(os.getenv("GENERIC_REWRITE_COUNT", str(M)))
GENERIC_REWRITER_MAX_NEW_TOKENS = int(
    os.getenv("GENERIC_REWRITER_MAX_NEW_TOKENS", "512")
)

REWRITER_SYSTEM_PROMPT = """
You are a general-purpose prompt rewriter.
Rewrite the user's prompt so it is clearer, more specific, and easier for a
language model to follow.

Rules:
- Preserve the original intent, meaning, requirements, and constraints.
- Do not answer the prompt.
- Do not add facts, assumptions, examples, or requirements not present in the
  original prompt.
- Preserve requested language, format, tone, audience, and safety constraints.
- Return only the rewritten prompt, with no explanation or quotation marks.
""".strip()

VARIATION_FOCUSES = [
    "Prioritize unambiguous wording and explicit task structure.",
    "Prioritize concise wording while retaining every requirement.",
    "Prioritize logical ordering of instructions and constraints.",
    "Prioritize clarity about the expected output.",
    "Prioritize preserving subtle intent while removing ambiguity.",
    "Prioritize readability for a general-purpose language model.",
    "Prioritize precise verbs and concrete references.",
    "Prioritize grouping related requirements together.",
    "Prioritize direct language without unnecessary verbosity.",
    "Prioritize faithful reformulation with a distinct sentence structure.",
]


def save_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")

    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary_path, path)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_experiment_names():
    if not EXPERIMENT_FILE.exists():
        raise FileNotFoundError(f"Experiment file not found: {EXPERIMENT_FILE}")

    with EXPERIMENT_FILE.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_source_path(experiment_name):
    direct_path = EVALUATION_DIR / f"{experiment_name}.json"
    if direct_path.exists():
        return direct_path

    for embedding_model_name in embedding_models:
        embedded_path = (
            EVALUATION_DIR
            / clean_name(embedding_model_name)
            / f"{experiment_name}.json"
        )
        if embedded_path.exists():
            return embedded_path

    raise FileNotFoundError(
        f"No evaluation JSON found for experiment: {experiment_name}"
    )


def normalize_rewrite(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.lower().startswith("text"):
            text = text[4:].strip()

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def apply_rewriter_chat_template(tokenizer, messages):
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **template_kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **template_kwargs)


def rewrite_prompt(model, tokenizer, original_prompt, variation_index=None):
    if variation_index is None:
        variation_instruction = (
            "Produce the single best faithful rewrite of the original prompt."
        )
    else:
        focus = VARIATION_FOCUSES[
            variation_index % len(VARIATION_FOCUSES)
        ]
        variation_instruction = (
            f"Produce rewrite candidate {variation_index + 1}. {focus} "
            "Use wording and structure that differ from other plausible "
            "rewrites, without changing the task."
        )

    user_prompt = (
        f"{variation_instruction}\n\n"
        "Original prompt:\n"
        f"<original_prompt>\n{original_prompt}\n</original_prompt>"
    )

    messages = [
        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    formatted_prompt = apply_rewriter_chat_template(tokenizer, messages)
    model_device = next(model.parameters()).device
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
    ).to(model_device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GENERIC_REWRITER_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
    rewritten_prompt = normalize_rewrite(
        tokenizer.decode(generated_ids, skip_special_tokens=True)
    )
    if not rewritten_prompt:
        raise ValueError("Generic rewriter returned an empty prompt")
    return rewritten_prompt


def prepare_checkpoint(experiment_name):
    checkpoint_path = CHECKPOINT_DIR / f"{experiment_name}.json"
    if checkpoint_path.exists():
        return checkpoint_path, load_json(checkpoint_path)

    source_path = get_source_path(experiment_name)
    data = load_json(source_path)
    save_json_atomic(checkpoint_path, data)
    return checkpoint_path, data


def generate_generic_prompts(model, tokenizer, experiment_name):
    checkpoint_path, data = prepare_checkpoint(experiment_name)
    print(
        f"\nGenerating generic rewrites for {experiment_name}: "
        f"{len(data)} items"
    )

    for item_index, item in enumerate(
        tqdm(data, desc=f"Rewrite {experiment_name}"),
        start=1,
    ):
        original_prompt = item.get("ori_prompt")
        if not isinstance(original_prompt, str) or not original_prompt.strip():
            print(f"Skipping item {item_index}: missing ori_prompt")
            continue

        if not isinstance(item.get("generic_prompt"), str) or not item[
            "generic_prompt"
        ].strip():
            item["generic_prompt"] = rewrite_prompt(
                model,
                tokenizer,
                original_prompt,
            )
            save_json_atomic(checkpoint_path, data)

        paraphrases = item.get("generic_paraphrases")
        if not isinstance(paraphrases, list):
            paraphrases = []

        paraphrases = [
            prompt.strip()
            for prompt in paraphrases
            if isinstance(prompt, str) and prompt.strip()
        ][:GENERIC_REWRITE_COUNT]
        item["generic_paraphrases"] = paraphrases

        while len(item["generic_paraphrases"]) < GENERIC_REWRITE_COUNT:
            variation_index = len(item["generic_paraphrases"])
            rewritten_prompt = rewrite_prompt(
                model,
                tokenizer,
                original_prompt,
                variation_index=variation_index,
            )
            item["generic_paraphrases"].append(rewritten_prompt)
            save_json_atomic(checkpoint_path, data)

    return checkpoint_path, data


def cluster_prompts(prompts, embedding_model, distance_threshold):
    embeddings = embedding_model.encode(
        prompts,
        convert_to_tensor=True,
    )
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    ).fit_predict(embeddings.cpu().numpy())

    clusters_by_label = {}
    for prompt_index, label in enumerate(labels):
        clusters_by_label.setdefault(int(label), []).append(prompt_index)
    return list(clusters_by_label.values()), embeddings


def select_cluster_representatives(
    original_prompt,
    clusters,
    embeddings,
    embedding_model,
):
    original_embedding = embedding_model.encode(
        [original_prompt],
        convert_to_tensor=True,
    )[0]
    representatives = []

    for cluster in clusters:
        if len(cluster) == 1:
            representatives.append(cluster[0])
            continue

        cluster_embeddings = torch.stack(
            [embeddings[index] for index in cluster]
        )
        similarities = util.pytorch_cos_sim(
            original_embedding,
            cluster_embeddings,
        )[0]
        sorted_indices = torch.argsort(similarities)
        median_position = sorted_indices[len(sorted_indices) // 2].item()
        representatives.append(cluster[median_position])

    return representatives, original_embedding


def compute_consensus_scores(
    representatives,
    embeddings,
    original_embedding,
):
    if len(representatives) == 1:
        return [0.0]

    scores = []
    for representative_index in representatives:
        representative_embedding = embeddings[representative_index]
        score = 0.0

        for other_index in representatives:
            if representative_index == other_index:
                continue
            score += util.pytorch_cos_sim(
                representative_embedding,
                embeddings[other_index],
            ).item()

        score -= (
            util.pytorch_cos_sim(
                representative_embedding,
                original_embedding,
            ).item()
            * IMP_ENC
        )
        scores.append(score)

    return scores


def apply_prism_selection(item, embedding_model, distance_threshold):
    original_prompt = item.get("ori_prompt", "")
    prompts = item.get("generic_paraphrases", [])[:GENERIC_REWRITE_COUNT]

    if len(prompts) == 0:
        item["rgeneric_prompt"] = None
        return
    if len(prompts) == 1:
        item["rgeneric_clusters"] = [[prompts[0]]]
        item["rgeneric_cluster_representatives"] = [prompts[0]]
        item["rgeneric_consensus_scores"] = [0.0]
        item["rgeneric_selected_idx"] = 0
        item["rgeneric_prompt"] = prompts[0]
        return

    clusters, embeddings = cluster_prompts(
        prompts,
        embedding_model,
        distance_threshold,
    )
    representatives, original_embedding = select_cluster_representatives(
        original_prompt,
        clusters,
        embeddings,
        embedding_model,
    )
    consensus_scores = compute_consensus_scores(
        representatives,
        embeddings,
        original_embedding,
    )
    best_score_position = max(
        range(len(consensus_scores)),
        key=consensus_scores.__getitem__,
    )
    selected_index = representatives[best_score_position]

    item["rgeneric_clusters"] = [
        [prompts[index] for index in cluster]
        for cluster in clusters
    ]
    item["rgeneric_cluster_representative_indices"] = representatives
    item["rgeneric_cluster_representatives"] = [
        prompts[index] for index in representatives
    ]
    item["rgeneric_consensus_scores"] = consensus_scores
    item["rgeneric_selected_idx"] = selected_index
    item["rgeneric_prompt"] = prompts[selected_index]


def run_prism_for_embedding_model(
    experiment_name,
    source_data,
    embedding_model_name,
):
    distance_threshold = distance_thresholds.get(embedding_model_name)
    if distance_threshold is None:
        raise ValueError(
            f"No distance threshold configured for {embedding_model_name}"
        )

    print(
        f"Applying PRISM with {embedding_model_name} "
        f"(threshold={distance_threshold})"
    )
    embedding_model = SentenceTransformer(
        embedding_model_name,
        device=device,
        cache_folder=MODEL_CACHE_PATH,
    )

    output_path = (
        EVALUATION_DIR
        / clean_name(embedding_model_name)
        / f"{experiment_name}.json"
    )
    if output_path.exists():
        data = load_json(output_path)
    else:
        data = json.loads(json.dumps(source_data, ensure_ascii=False))

    if len(data) != len(source_data):
        raise ValueError(
            f"Item count mismatch for {output_path}: "
            f"{len(data)} != {len(source_data)}"
        )

    for item_index, item in enumerate(
        tqdm(data, desc=f"PRISM {clean_name(embedding_model_name)}")
    ):
        generic_source = source_data[item_index]
        if item.get("ori_prompt") != generic_source.get("ori_prompt"):
            raise ValueError(
                f"ori_prompt mismatch at index {item_index} in {output_path}"
            )

        item["generic_prompt"] = generic_source.get("generic_prompt")
        item["generic_paraphrases"] = generic_source.get(
            "generic_paraphrases",
            [],
        )
        apply_prism_selection(
            item,
            embedding_model,
            distance_threshold,
        )
        item.pop("generic_response", None)
        item.pop("rgeneric_response", None)

    save_json_atomic(output_path, data)
    print(f"Saved: {output_path}")

    del embedding_model
    torch.cuda.empty_cache()
    gc.collect()


def main():
    experiment_names = load_experiment_names()
    print(f"Loading generic prompt rewriter: {GENERIC_LLM_REWRITER}")
    rewriter_model, rewriter_tokenizer = load_model_and_tokenizer(
        model_path=GENERIC_LLM_REWRITER,
        cache_dir=MODEL_CACHE_PATH,
    )

    for experiment_name in experiment_names:
        generate_generic_prompts(
            rewriter_model,
            rewriter_tokenizer,
            experiment_name,
        )

    del rewriter_model
    del rewriter_tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    for experiment_name in experiment_names:
        checkpoint_path = CHECKPOINT_DIR / f"{experiment_name}.json"
        generic_data = load_json(checkpoint_path)
        for embedding_model_name in embedding_models:
            run_prism_for_embedding_model(
                experiment_name,
                generic_data,
                embedding_model_name,
            )


if __name__ == "__main__":
    main()
