import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from helper import (CHATGPT, GEMMA_EMBEDDING_MODEL, LLAMA2_7B, MINILM_EMBEDDING_MODEL, VICUNA_7B, DOLLY_EVAL, VICUNA_EVAL, 
    DEEPSEEK, GEMMA3, clean_name, eval_folder_name, evaluator_models, base_llm_models, 
    evaluation_datasets, create_combined_name, mismatch_folder_name, 
    consistency_folder_name, embedding_models)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_NAME = CHATGPT
VERIFY_FOLDER_NAME = f"verify_{clean_name(MODEL_NAME)}"
PROMPT_FILE = os.path.join(BASE_DIR, "response_eval_prompt.txt")
API_RETRY_TIMES = 5
API_RETRY_BASE_DELAY = 2
REQUEST_TIMEOUT_SECONDS = 120

evaluator_models = [MODEL_NAME]

client = (
    OpenAI(
        api_key=OPENAI_API_KEY,
        max_retries=2,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if OPENAI_API_KEY
    else None
)

EXPECTED_CRITERIA = [
    "Correctness", "Relevance", "Completeness", "Clarity_Coherence",
    "Usefulness_Helpfulness", "Style_Tone", "Conciseness", "Safety_Compliance"
]


class ResponseScores(BaseModel):
    Correctness: float = Field(ge=0.0, le=1.0)
    Relevance: float = Field(ge=0.0, le=1.0)
    Completeness: float = Field(ge=0.0, le=1.0)
    Clarity_Coherence: float = Field(ge=0.0, le=1.0)
    Usefulness_Helpfulness: float = Field(ge=0.0, le=1.0)
    Style_Tone: float = Field(ge=0.0, le=1.0)
    Conciseness: float = Field(ge=0.0, le=1.0)
    Safety_Compliance: float = Field(ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    response_A: ResponseScores
    response_B: ResponseScores

# Keyword mapping: model tự sinh criteria → map về criteria đúng
CRITERIA_MAP = {
    "Correctness":            ["correctness", "accuracy", "factual", "correct", "truthfulness", "honesty"],
    "Relevance":              ["relevance", "relevant", "on_topic", "topicality"],
    "Completeness":           ["completeness", "complete", "coverage", "thoroughness", "recall"],
    "Clarity_Coherence":      ["clarity", "coherence", "clarity_coherence", "readability", "structure", "clear"],
    "Usefulness_Helpfulness": ["usefulness", "helpfulness", "useful", "helpful", "instruction_following", "practicality"],
    "Style_Tone":             ["style", "tone", "style_tone", "formality", "politeness"],
    "Conciseness":            ["conciseness", "concise", "brevity", "verbosity"],
    "Safety_Compliance":      ["safety", "safety_compliance", "compliance", "bias", "harmful", "safe"]
}

# LOAD PROMPT FROM FILE
if not os.path.exists(PROMPT_FILE):
    raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

_prompt_vars = {}
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    exec(f.read(), _prompt_vars)

SYSTEM_PROMPT = _prompt_vars["SYSTEM_PROMPT"]

print("Loaded SYSTEM_PROMPT from response_eval_prompt.txt")
print(SYSTEM_PROMPT[:200], "\n---")

# USER PROMPT
def build_user_prompt(item, verify_keys):
    return f"""
Prompt_A (used to generate Response_A):
\"\"\"{item.get(verify_keys[0], "")}\"\"\"

Response_A:
\"\"\"{item.get(verify_keys[1], "")}\"\"\"

Prompt_B (used to generate Response_B):
\"\"\"{item.get(verify_keys[2], "")}\"\"\"

Response_B:
\"\"\"{item.get(verify_keys[3], "")}\"\"\"

# IMPORTANT RULES:
- Judge Response_A ONLY based on Prompt_A
- Judge Response_B ONLY based on Prompt_B
- Do NOT compare Response_A and Response_B
- Do NOT use any other information
- Use the SAME scoring scale for both

# Score meaning:
0.0 = completely incorrect or useless
0.5 = partially correct / moderate quality
1.0 = fully correct / high quality

# Additional rules:
- Each criterion must be scored independently
- Scores should not all be identical unless fully justified
- If response fails to answer, give low scores (0.0–0.2)
- Use only numbers between 0.0 and 1.0
- Use one decimal place
- Do NOT include any explanation or extra text

# OUTPUT:
Return STRICTLY valid JSON.
No explanation, no markdown, no extra text.

# OUTPUT FORMAT:
Return JSON ONLY in the following format:
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

# GENERATION
def generate(system_prompt, user_prompt):
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is missing from the environment")

    response = client.responses.parse(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        text_format=EvaluationResult,
    )
    if response.output_parsed is None:
        raise ValueError("GPT did not return a parsed evaluation")

    return response.output_parsed.model_dump_json()

# EXTRACT JSON
def extract_json(raw):
    # check raw la json chua, neu da la json thi return string do luon
    import json
    import re
    try: 
        json.loads(raw)  # Verify it's valid JSON
        return raw  # Return the raw string, not the parsed dict
    except Exception:
        pass
        
    """Strip thinking tags, markdown code blocks, lấy chỉ phần JSON"""
    # Bỏ <think>...</think>
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    # Bỏ ```json ... ``` hoặc ``` ... ```
    match = re.search(r'```(?:json)?\s*(.*?)```', raw, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    # Nếu không có code block, tìm JSON object đầu tiên
    match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
    if match:
        return match.group(0).strip()
    return raw.strip()

def validate_schema(parsed):
    """Kiểm tra parsed JSON có đúng format không"""
    if not isinstance(parsed, dict):
        return False
    for key in ["response_A", "response_B"]:
        if key not in parsed:
            return False
        if not isinstance(parsed[key], dict):
            return False
        for criteria in EXPECTED_CRITERIA:
            if criteria not in parsed[key]:
                return False
            if not isinstance(parsed[key][criteria], (int, float)):
                return False
    return True

def map_to_schema(raw_scores):
    """Map criteria của model về schema đúng"""
    mapped = {c: None for c in EXPECTED_CRITERIA}

    for model_key, model_val in raw_scores.items():
        if not isinstance(model_val, (int, float)):
            continue
        model_key_lower = model_key.lower()

        # Tìm criteria phù hợp nhất
        for expected, keywords in CRITERIA_MAP.items():
            if mapped[expected] is not None:
                continue  # đã assign rồi, skip
            if model_key_lower in keywords or model_key_lower == expected.lower():
                mapped[expected] = float(model_val)
                break

    # Các criteria chưa map được → dùng average của các scores đã map
    mapped_values = [v for v in mapped.values() if v is not None]
    avg = sum(mapped_values) / len(mapped_values) if mapped_values else 0.0

    for c in EXPECTED_CRITERIA:
        if mapped[c] is None:
            mapped[c] = round(avg, 1)

    return mapped

# FALLBACK
def empty_schema():
    return {
        "response_A": {c: 0.0 for c in EXPECTED_CRITERIA},
        "response_B": {c: 0.0 for c in EXPECTED_CRITERIA}
    }

# check đầy đủ schema không
def is_complete(candidate):
    if not isinstance(candidate, dict):
        return False
    for key in ["response_A", "response_B"]:
        if key not in candidate:
            return False
        for c in EXPECTED_CRITERIA:
            if c not in candidate[key]:
                return False
            if not isinstance(candidate[key][c], (int, float)):
                return False
    return True

# DECIDE WINNER
def decide_winner_from_scores(llm_eval, threshold=0.01):
    """
    So sánh scores → return winner
    0 = draw
    1 = response_A win (res_0)
    2 = response_B win (res_1)
    """
    score_A = sum(llm_eval["response_A"].values()) / len(llm_eval["response_A"])
    score_B = sum(llm_eval["response_B"].values()) / len(llm_eval["response_B"])

    diff = score_A - score_B
    if abs(diff) < threshold:
        return 2  # draw
    elif diff > 0:
        return 0  # response_A win
    else:
        return 1  # response_B win
    
def verify_response_with_zero_fallback_legacy(item_id, sample, verify_methods, attempt_times=3):
    parsed = None
    candidate = None
    for attempt in range(attempt_times):  # retry tối đa attempt_times-1 lần
        try:
            raw = generate(SYSTEM_PROMPT, build_user_prompt(sample, verify_methods))
            # print(f"  raw (attempt {attempt+1}): {raw[:200]}")
            # print(f"[ID={item_id}] attempt {attempt+1}")
            candidate = json.loads(extract_json(raw))

            if is_complete(candidate):
                parsed = candidate
                return parsed

            # nếu có đủ A và B nhưng thiếu / sai criteria thì map
            if isinstance(candidate, dict) and "response_A" in candidate and "response_B" in candidate:
                mapped = {
                    "response_A": map_to_schema(candidate["response_A"]),
                    "response_B": map_to_schema(candidate["response_B"])
                }
                if is_complete(mapped):
                    parsed = mapped
                    print("Mapped schema")
                    return parsed
        except Exception as e:
            print(f"Parse fail (attempt {attempt+1}): {e}")
            continue
    if parsed is None:
        print(f"ID={item_id}: dùng fallback")
        parsed = empty_schema()
    return parsed


def verify_response(item_id, sample, verify_methods, attempt_times=API_RETRY_TIMES):
    """Retry transient failures and never mark a failed API call as completed."""
    last_error = None
    for attempt in range(1, attempt_times + 1):
        try:
            raw = generate(SYSTEM_PROMPT, build_user_prompt(sample, verify_methods))
            candidate = json.loads(extract_json(raw))

            if is_complete(candidate):
                return candidate

            if isinstance(candidate, dict) and "response_A" in candidate and "response_B" in candidate:
                mapped = {
                    "response_A": map_to_schema(candidate["response_A"]),
                    "response_B": map_to_schema(candidate["response_B"])
                }
                if is_complete(mapped):
                    print("Mapped schema")
                    return mapped

            raise ValueError("GPT response does not match the evaluation schema")
        except Exception as error:
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


def batch_iterator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def process_batch(batch, verify_key, verify_methods, attempt_times=3):
    res = []
    for idx, item in enumerate(batch):
        item_id = item.get("id", idx + 1)
        assert len(verify_methods) == 4, "verify_methods phải có đúng 4 keys tương ứng với Prompt_A, Response_A, Prompt_B, Response_B"
        parsed = verify_response(item_id, item, verify_methods, attempt_times)
        
        obj = {
            'id': item_id,
            'ori_prompt': item.get('ori_prompt', None),
            'context': item.get('context', None),
            verify_methods[0]: item.get(verify_methods[0], None),
            verify_methods[1]: item.get(verify_methods[1], None),
            verify_methods[2]: item.get(verify_methods[2], None),
            verify_methods[3]: item.get(verify_methods[3], None),
            f'{verify_key}_winner': decide_winner_from_scores(parsed),
            f'{verify_key}_llm_evaluation': parsed
        }
        
        res.append(obj)
    return res


def save_json_atomic(output_path, data):
    """Save a checkpoint without exposing a partially written JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = f"{output_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_path, output_path)


def load_checkpoint(output_path):
    if not os.path.exists(output_path):
        return []

    with open(output_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    if not isinstance(checkpoint, list):
        raise ValueError(f"Invalid checkpoint format: {output_path}")
    return checkpoint


# MAIN
def verify_response_batch(verify_key, verify_methods, verify_times = 5, BATCH_SIZE = 1):
    print(f"Starting verification for key: {verify_key} with methods: {verify_methods}")
    for embed_model in embedding_models:
        for base_llm in base_llm_models:
            for dataset in evaluation_datasets:
                for evaluator in evaluator_models:
                    print(f"\n=== VERIFYING: {base_llm} | {dataset} | {evaluator} ===")
                    input_path = create_combined_name(base_llm, dataset, evaluator)
                    print(f"Processing: {input_path}")
                    verify_path = f"{eval_folder_name}/{clean_name(embed_model)}/{input_path}.json"
                    print(f"Verify path: {verify_path}")
                    
                    # continue
                    with open(verify_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for run_idx in range(verify_times):
                        output_path = f"{eval_folder_name}/{clean_name(embed_model)}/{VERIFY_FOLDER_NAME}/{verify_key}/{input_path}_eval_{run_idx+1}.json"
                        results = load_checkpoint(output_path)
                        completed_ids = {item.get("id") for item in results}
                        pending_items = [
                            (index, item)
                            for index, item in enumerate(data, start=1)
                            if item.get("id", index) not in completed_ids
                        ]

                        print(
                            f"Run {run_idx + 1}/{verify_times}: "
                            f"{len(results)}/{len(data)} completed, "
                            f"{len(pending_items)} pending"
                        )

                        try:
                            for source_index, item in pending_items:
                                item_id = item.get("id", source_index)
                                item_to_process = dict(item)
                                item_to_process.setdefault("id", item_id)
                                batch_results = process_batch(
                                    [item_to_process],
                                    verify_key,
                                    verify_methods,
                                    attempt_times=API_RETRY_TIMES,
                                )
                                results.extend(batch_results)
                                save_json_atomic(output_path, results)
                                print(
                                    f"Saved ID={item_id} "
                                    f"({len(results)}/{len(data)})"
                                )
                        except KeyboardInterrupt:
                            print(
                                f"\nInterrupted. Progress is saved at: {output_path}"
                            )
                            raise
                        except Exception:
                            print(
                                "Stopped at the failed item. Run the script again "
                                f"to resume from: {output_path}"
                            )
                            raise

def load_data_for_consistency_check(checked_path, check_runs):
    data = []
    for i in range(1, check_runs + 1):
        run_path = checked_path.replace("_eval_1.json", f"_eval_{i}.json")
        if os.path.exists(run_path):
            with open(run_path, "r", encoding="utf-8") as f:
                data.append(json.load(f))
        else:
            print(f"Missing file for consistency check: {run_path}")
    return data

def check_verify_consistency(verify_key,verify_methods, check_runs=5):
    """
    So sánh kết quả winner giữa nhiều lần chạy verify:
    - Lưu các kết quả LỆCH nhau vào file _mismatch.json
    - Lưu các kết quả ỔN ĐỊNH (giống nhau) vào file _consistency.json đồng thời đếm tỷ lệ winner 0, 1, 2.
    - Lưu ý file eval có id 81 (vicuna) và 201 (dolly) là id tổng hợp kết quả nên winner = 2 (bỏ đi không tính cái này)
    """
    for embed_model in embedding_models:
        for base_llm in base_llm_models:
            for dataset in evaluation_datasets:
                for evaluator in evaluator_models:
                    file_name = create_combined_name(base_llm, dataset, evaluator)
                    print(f"\nChecking consistency for: {file_name}")
                    saved_path = f"{eval_folder_name}/{clean_name(embed_model)}/{VERIFY_FOLDER_NAME}/{verify_key}/{file_name}_eval_1.json"
                    if not os.path.exists(saved_path):
                        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
                    
                    
                    data = load_data_for_consistency_check(saved_path, check_runs)
                    if not data:
                        print(f"No data loaded for {file_name}")
                        continue
                    
                    # Xác định ID cần bỏ dựa trên dataset
                    exclude_ids = set()
                    if "vicuna" in dataset.lower():
                        exclude_ids.add(81)
                    elif "dolly" in dataset.lower():
                        exclude_ids.add(201)
                    
                    mismatches = []
                    consistencies = []
                    
                    num_items = len(data[0])
                    print(f"Checking {num_items} items across {len(data)} runs...")
                    
                    for idx in range(num_items):
                        winners = [run[idx][f"{verify_key}_winner"] for run in data if idx < len(run)]
                        item_id = data[0][idx].get("id")
                        item_data = {
                            "id": item_id,
                            "ori_prompt": data[0][idx].get("ori_prompt"),
                            f"{verify_methods[0]}": data[0][idx].get(verify_methods[0]),
                            f"{verify_methods[1]}": data[0][idx].get(verify_methods[1]),
                            f"{verify_methods[2]}": data[0][idx].get(verify_methods[2]),
                            f"{verify_methods[3]}": data[0][idx].get(verify_methods[3]),
                            "winners_per_run": winners,
                            "llm_evaluations_per_run": [run[idx][f"{verify_key}_llm_evaluation"] for run in data if idx < len(run)]
                        }
                        
                        if len(set(winners)) > 1:  # Có sự khác biệt
                            mismatches.append(item_data)
                        else:  # Ổn định (tất cả winner giống nhau)
                            consistencies.append(item_data)
                    
                    def count_winner_distribution(items, exclude_ids):
                        winner_counts = {2: 0, 0: 0, 1: 0}  # 2=draw, 0=A_win, 1=B_win
                        excluded_count = 0
                        
                        for item in items:
                            item_id = item.get("id")
                            
                            if item_id in exclude_ids:
                                excluded_count += 1
                                continue
                            
                            winners = item.get("winners_per_run", [])
                            if winners:
                                if len(set(winners)) == 1:
                                    winner = winners[0]
                                else:
                                    from collections import Counter
                                    winner = Counter(winners).most_common(1)[0][0]
                                
                                if winner in winner_counts:
                                    winner_counts[winner] += 1
                        
                        total = sum(winner_counts.values())
                        if total == 0:
                            return {
                                "response_A_win": 0, "response_A_win_rate": "0.00%",
                                "response_B_win": 0, "response_B_win_rate": "0.00%",
                                "draw": 0, "draw_rate": "0.00%",
                                "excluded_count": excluded_count
                            }
                        
                        return {
                            "response_A_win": winner_counts[0],
                            "response_A_win_rate": f"{(winner_counts[0] / total * 100):.2f}%",
                            "response_B_win": winner_counts[1],
                            "response_B_win_rate": f"{(winner_counts[1] / total * 100):.2f}%",
                            "draw": winner_counts[2],
                            "draw_rate": f"{(winner_counts[2] / total * 100):.2f}%",
                            "excluded_count": excluded_count,
                            "counted_items": total
                        }
                    
                    # Lưu file _mismatch.json
                    if mismatches:
                        mismatch_path = f"{eval_folder_name}/{clean_name(embed_model)}/{VERIFY_FOLDER_NAME}/{verify_key}/{mismatch_folder_name}/{file_name}_mismatch.json"
                        os.makedirs(os.path.dirname(mismatch_path), exist_ok=True)
                        
                        mismatch_result = {
                            "total_items": num_items,
                            "total_mismatches": len(mismatches),
                            "mismatch_rate": f"{(len(mismatches) / num_items * 100):.2f}%",
                            "check_runs": len(data),
                            "mismatches": mismatches
                        }
                        
                        with open(mismatch_path, "w", encoding="utf-8") as f:
                            json.dump(mismatch_result, f, ensure_ascii=False, indent=2)
                        
                        print(f"Found {len(mismatches)}/{num_items} mismatches ({mismatch_result['mismatch_rate']})")
                        print(f"Saved: {mismatch_path}")
                    
                    # Lưu file _consistency.json
                    if consistencies:
                        consistency_path = f"{eval_folder_name}/{clean_name(embed_model)}/{VERIFY_FOLDER_NAME}/{verify_key}/{consistency_folder_name}/{file_name}_consistency.json"
                        os.makedirs(os.path.dirname(consistency_path), exist_ok=True)
                        
                        consistency_winner_dist = count_winner_distribution(consistencies, exclude_ids)
                        
                        consistency_result = {
                            "total_items": num_items,
                            "total_consistent": len(consistencies),
                            "consistency_rate": f"{(len(consistencies) / num_items * 100):.2f}%",
                            "check_runs": len(data),
                            "winner_distribution": consistency_winner_dist,
                            "note": f"Excluded ID {', '.join(map(str, sorted(exclude_ids)))} (aggregated results)" if exclude_ids else "No exclusions",
                            "consistent_items": consistencies
                        }
                        
                        with open(consistency_path, "w", encoding="utf-8") as f:
                            json.dump(consistency_result, f, ensure_ascii=False, indent=2)
                        
                        print(f"Found {len(consistencies)}/{num_items} consistent items ({consistency_result['consistency_rate']})")
                        print(f"Winner (excluding IDs {', '.join(map(str, sorted(exclude_ids)))}): A={consistency_winner_dist['response_A_win']} ({consistency_winner_dist['response_A_win_rate']}) | B={consistency_winner_dist['response_B_win']} ({consistency_winner_dist['response_B_win_rate']}) | Draw={consistency_winner_dist['draw']} ({consistency_winner_dist['draw_rate']}) [Excluded: {consistency_winner_dist.get('excluded_count', 0)}]")
                        print(f"Saved: {consistency_path}")
                    
                    # Summary
                    if not mismatches and not consistencies:
                        print(f"No items to process")
                    else:
                        print(f"Summary: {len(mismatches)} mismatches + {len(consistencies)} consistent = {num_items} total")
                
BATCH_SIZE = 1
VERIFY_TIMES = 3
embedding_models = [GEMMA_EMBEDDING_MODEL, MINILM_EMBEDDING_MODEL]

verify_keys_method = {
    "rbpo_bpo": ['rbpo_prompt', 'rbpo_response', 'bpo_prompt', 'bpo_response'], # đánh giá RBPO vs BPO
    "rbpo_mepo": ['rbpo_prompt', 'rbpo_response', 'mepo_prompt', 'mepo_response'], # đánh giá RBPO vs MEPO
    "rmepo_mepo": ['rmepo_prompt', 'rmepo_response', 'mepo_prompt', 'mepo_response'] # đánh giá RMEPO vs MEPO
}
for key, keys in verify_keys_method.items():
    print(f"\n=== VERIFY KEY: {key} ===")
    print(f"Verify keys: {keys}")
    verify_response_batch(key, keys, verify_times=VERIFY_TIMES, BATCH_SIZE=BATCH_SIZE)
    check_verify_consistency(key,keys,check_runs=VERIFY_TIMES)
