"""
baseline_evaluation.py
Pretrained Baseline Evaluation on Multi-Schema Spider Benchmark

Evaluates the un-tuned base model (Qwen/Qwen2.5-Coder-1.5B-Instruct) on unseen
cross-database test queries from the Spider dataset. Measures exact-match SQL
accuracy, syntax validity, latency, and GPU memory usage before fine-tuning.

The evaluation engine in this file (prompt, decoding, metrics) is shared with
evaluate_finetuned.py, so the baseline and the fine-tuned model are scored identically.

Usage:
  # Baseline on 100 randomly sampled (seed 42) test questions:
  python baseline_evaluation.py --sample-size 100

  # Baseline on the full held-out test split:
  python baseline_evaluation.py --all

  # Fine-tuned model (after train_qlora.py): see evaluate_finetuned.py

Note: "exact match" here is a normalized STRING match against the reference SQL.
It is stricter than Spider's official metrics (a correct query written differently
counts as a miss). Execution accuracy cannot be computed because the dataset only
ships schema text, not the SQLite database files.
"""

import argparse
import json
import math
import random
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = BASE_DIR / "results"
TEST_FILE = SPLITS_DIR / "test.json"
VALIDATION_FILE = SPLITS_DIR / "validation.json"
STATS_FILE = SPLITS_DIR / "split_stats.json"

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SAMPLE_SEED = 42


def print_banner(title: str):
    """Prints a structured section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def split_path(split: str) -> Path:
    return VALIDATION_FILE if split == "validation" else TEST_FILE


def results_stem(tag: str, split: str) -> str:
    """File-name stem: baseline_eval / finetuned_eval for the test split, baseline_validation_eval ... otherwise."""
    return f"{tag}_eval" if split == "test" else f"{tag}_{split}_eval"


DEFAULT_ADAPTER_NAME = "qlora_adapter"


def adapter_tag(adapter_name: str) -> str:
    """Results-file tag: 'finetuned' for the default adapter, 'finetuned_<name>' for any other one,
    so evaluating a second adapter never overwrites the first one's results."""
    return "finetuned" if adapter_name == DEFAULT_ADAPTER_NAME else f"finetuned_{adapter_name}"


def verify_dataset_integrity(split: str = "test"):
    """Confirms split existence, unseen databases > 0, and non-empty schema fields."""
    print_banner(f"DATASET SPLIT & MULTI-SCHEMA INTEGRITY VERIFICATION ({split.upper()} SPLIT)")
    data_file = split_path(split)

    if not data_file.exists() or not STATS_FILE.exists():
        print(f"[!] ERROR: Missing split files in: {SPLITS_DIR}")
        print("    Please run: python split_dataset.py first!")
        sys.exit(1)

    with open(STATS_FILE, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # 1. Check that the evaluated split uses databases the model never trained on
    db_dist = stats.get("audit", {}).get("database_distribution", {})
    prefix = "val" if split == "validation" else "test"
    unseen_dbs = db_dist.get(f"{prefix}_unseen_in_train", 0)
    total_split_dbs = db_dist.get(f"{prefix}_unique_databases", 0)
    label = split.capitalize()

    print(f"[*] Audit Verification from {STATS_FILE.name}:")
    print(f"    - Total Databases in Dataset : {db_dist.get('total_unique_databases', 'N/A')}")
    print(f"    - Unique Databases in Train  : {db_dist.get('train_unique_databases', 'N/A')}")
    print(f"    - Unique Databases in {label:<5}: {total_split_dbs}")
    print(f"    - {label} Databases Unseen in Train ({prefix}_unseen_in_train): {unseen_dbs}")

    if unseen_dbs > 0:
        print(f"    [OK] CONFIRMED: {prefix}_unseen_in_train = {unseen_dbs} (> 0).")
        print(f"         This split evaluates true cross-database zero-shot generalization!")
    else:
        print(f"    [!] WARNING: {prefix}_unseen_in_train is 0. This split does NOT evaluate unseen databases.")

    # 2. Check required fields
    with open(data_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    required_keys = ["db_id", "db_schema", "question", "query"]
    missing_counts = {k: 0 for k in required_keys}

    for item in samples:
        for k in required_keys:
            if not item.get(k):
                missing_counts[k] += 1

    print(f"\n[*] Sample Schema Field Completeness (Total: {len(samples):,} records):")
    for k, v in missing_counts.items():
        status = "[OK] Present" if v == 0 else f"[!] {v} Missing"
        print(f"    - Field '{k:<10}': {status}")

    if any(v > 0 for v in missing_counts.values()):
        print(f"[!] ERROR: Incomplete records found in {data_file.name}.")
        sys.exit(1)

    print(f"[OK] All {len(samples):,} {split} records contain valid db_id, schema, question, and query.")
    return samples, stats


def clean_sql(raw_output: str) -> str:
    """Extracts and normalizes clean SQL from model response."""
    text = raw_output.strip()
    if not text:
        return ""

    # Match markdown code fences ```sql ... ```
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        m_inner = re.search(r"((?:\bWITH\s+\w+\s+AS\b|\bSELECT\b)[\s\S]*?)(?:;|$)", content, re.IGNORECASE)
        text = m_inner.group(1).strip() if m_inner else content
    else:
        m_query = re.search(r"((?:\bWITH\s+\w+\s+AS\b|\bSELECT\b)[\s\S]*?)(?:;|$)", text, re.IGNORECASE)
        if m_query:
            text = m_query.group(1).strip()
        else:
            return ""

    text = " ".join(text.split())
    upper_start = text.upper()
    if not (upper_start.startswith("SELECT") or upper_start.startswith("WITH")):
        return ""
    if text and not text.endswith(";"):
        text += ";"
    return text


def normalize_for_em(sql: str) -> str:
    """Normalizes SQL for string-level Exact-Match comparison.

    Keywords, identifiers and function names are compared case-insensitively
    (COUNT == count), spacing around punctuation/operators, quote style and the
    trailing semicolon are ignored (`count(*)` == `COUNT ( * )`, "x" == 'x').
    String literals keep their case, because SQLite compares text case-sensitively:
    WHERE country = 'France' is NOT the same query as WHERE country = 'france'.
    """
    if not sql:
        return ""
    s = sql.strip().rstrip(";").strip()

    # Park string literals behind placeholders so they are excluded from lower-casing/spacing rules.
    # "double-quoted" literals are rewritten as 'single-quoted' ones first (Spider uses both styles).
    literals: list[str] = []

    def park(match):
        text = match.group(0)
        if text[0] == '"':
            text = "'" + text[1:-1].replace('""', '"').replace("'", "''") + "'"
        literals.append(text)
        return f"\x00{len(literals) - 1}\x00"

    s = re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", park, s)
    s = " ".join(s.split()).lower()
    s = s.replace("<>", "!=")
    s = re.sub(r"\s*([(),])\s*", r"\1", s)
    s = re.sub(r"\s*(!=|<=|>=|=|<|>)\s*", r"\1", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: literals[int(m.group(1))], s)


def relaxed_tokens(sql: str) -> Counter:
    """Bag of SQL tokens that ignores everything that does not change the meaning of the query:
    table aliases (T1/t1/x), alias-qualified columns (t1.name -> name), join/ON order, spacing, case
    of keywords and identifiers (string literals keep their case)."""
    s = normalize_for_em(sql)
    tokens = re.findall(
        r"'(?:[^']|'')*'|[a-z_]\w*(?:\.(?:[a-z_]\w*|\*))?|\d+(?:\.\d+)?|!=|<=|>=|[=<>(),*+\-/%]", s)
    aliases = {tokens[i + 1] for i in range(len(tokens) - 1)
               if tokens[i] == "as" and re.fullmatch(r"[a-z_]\w*", tokens[i + 1])}
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "as" and i + 1 < len(tokens) and tokens[i + 1] in aliases:
            i += 2                      # drop "AS alias"
            continue
        if "." in tok and not tok.startswith("'") and not tok[0].isdigit():
            tok = tok.split(".", 1)[1]  # t1.name -> name
        if tok not in aliases:
            out.append(tok)
        i += 1
    return Counter(out)


def relaxed_match(generated: str, reference: str) -> bool:
    """True when both queries use exactly the same tokens, ignoring aliases / join order."""
    return bool(generated) and relaxed_tokens(generated) == relaxed_tokens(reference)


def check_syntax(sql: str) -> tuple[bool, str]:
    """Validates SQL query syntax using the SQLite parser."""
    if not sql:
        return False, "Empty SQL output"
    conn = sqlite3.connect(":memory:")
    try:
        # EXPLAIN validates syntax without needing the actual underlying tables
        conn.execute(f"EXPLAIN {sql}")
        return True, ""
    except sqlite3.OperationalError as e:
        err = str(e)
        # "no such table/column" means the statement itself parsed fine
        if "no such table" in err or "no such column" in err:
            return True, ""
        return False, err
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


# Prompt: must stay IDENTICAL to build_system_prompt / format_user_prompt in
# train_qlora.py, otherwise the fine-tuned model is evaluated on a format it was
# not trained on and the baseline-vs-finetuned comparison is not apples-to-apples.
def build_system_prompt() -> str:
    return (
        "You are a professional, schema-aware Text-to-SQL assistant. "
        "Given the target database identifier and its schema, generate a single "
        "syntactically valid SQLite query to answer the question. "
        "Return only the executable SQL query without explanations or markdown."
    )


def format_user_prompt(db_id: str, db_schema: str, question: str) -> str:
    return f"""Database ID:
{db_id.strip()}

Database Schema:
{db_schema.strip()}

Question:
{question.strip()}
"""


def build_prompt(tokenizer, db_id: str, db_schema: str, question: str) -> str:
    """Constructs the prompt using the model's chat template."""
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": format_user_prompt(db_id, db_schema, question)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_causal_lm(model_id, quantization_config, device_map, dtype):
    """from_pretrained with the dtype kwarg renamed across transformers versions."""
    from transformers import AutoModelForCausalLM
    kwargs = dict(quantization_config=quantization_config, device_map=device_map, trust_remote_code=True)
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


def resolve_adapter_path(adapter: str) -> Path:
    path = Path(adapter)
    if not path.exists() and not path.is_absolute():
        path = BASE_DIR / adapter
    if not (path / "adapter_config.json").exists():
        print(f"[!] ERROR: No LoRA adapter found at: {path}")
        print("    Run: python train_qlora.py first!")
        sys.exit(1)
    return path


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval (in %) for a rate measured on `total` questions."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def sign_test_p(improved: int, regressed: int) -> float:
    """Two-sided exact sign test: how likely is a split this lopsided if both models were equally good?"""
    n = improved + regressed
    if n == 0:
        return 1.0
    k = min(improved, regressed)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def print_paired_comparison(base_predictions: list, new_predictions: list,
                            old_label: str = "Baseline", new_label: str = "Fine-tuned"):
    """Question-by-question comparison of the baseline and the fine-tuned model."""
    base_by_id = {p["id"]: p for p in base_predictions}
    pairs = [(base_by_id[p["id"]], p) for p in new_predictions if p["id"] in base_by_id]
    if not pairs:
        return
    n = len(pairs)
    print_banner(f"DID {new_label.upper()} IMPROVE OVER {old_label.upper()}? ({n} identical questions, compared one by one)")
    metrics = [("Exact match", "exact_match")]
    if all("relaxed_match" in b and "relaxed_match" in f for b, f in pairs):
        metrics.append(("Relaxed match (ignores aliases / join order)", "relaxed_match"))
    metrics.append(("Syntax valid", "syntax_valid"))
    for metric, key in metrics:
        both = sum(1 for b, f in pairs if b[key] and f[key])
        fixed = sum(1 for b, f in pairs if not b[key] and f[key])
        broke = sum(1 for b, f in pairs if b[key] and not f[key])
        neither = n - both - fixed - broke
        b_ok, f_ok = both + broke, both + fixed
        b_lo, b_hi = wilson_interval(b_ok, n)
        f_lo, f_hi = wilson_interval(f_ok, n)
        p_value = sign_test_p(fixed, broke)
        print(f"  {metric}")
        print(f"    - {old_label:<12}: {100 * b_ok / n:5.1f}%   (95% CI {b_lo:.1f}-{b_hi:.1f}%)")
        print(f"    - {new_label:<12}: {100 * f_ok / n:5.1f}%   (95% CI {f_lo:.1f}-{f_hi:.1f}%)")
        print(f"    - Fixed by {new_label}: {fixed}  |  Broken by {new_label}: {broke}  |  right in both: {both}  |  wrong in both: {neither}")
        if fixed == broke:
            verdict = "no difference"
        elif p_value < 0.05:
            verdict = f"{new_label} is {'BETTER' if fixed > broke else 'WORSE'} (sign test p = {p_value:.4f})"
        else:
            verdict = f"{new_label} looks {'better' if fixed > broke else 'worse'}, but this could be chance (p = {p_value:.3f}); evaluate more questions (--all)"
        print(f"    - Verdict    : {verdict}")
    print("  (Exact match is a strict string comparison; equivalent SQL written differently counts as a miss.)")
    print("=" * 80)


def evaluate_model(model_id: str, sample_size: int | None = None, adapter: str | None = None, split: str = "test",
                   compare_with: str | None = None):
    """Loads the base model (optionally + LoRA adapter), runs inference, saves benchmarks."""
    test_samples, stats = verify_dataset_integrity(split)
    adapter_path = resolve_adapter_path(adapter) if adapter else None
    tag = adapter_tag(adapter_path.name) if adapter_path else "baseline"
    # What to compare against: the pretrained baseline (default) or another adapter (--compare-with NAME)
    if compare_with and compare_with != "baseline":
        compare_tag, compare_label = adapter_tag(compare_with), compare_with
    else:
        compare_tag, compare_label = "baseline", "Baseline"
    label = "FINE-TUNED (QLoRA)" if adapter_path else "PRETRAINED BASELINE"

    if sample_size and sample_size < len(test_samples):
        # Seeded random sample: test.json is ordered by database, so taking the
        # first N would only cover a handful of databases.
        eval_samples = random.Random(SAMPLE_SEED).sample(test_samples, sample_size)
        print(f"\n[*] Evaluating on a random sample (seed {SAMPLE_SEED}) of {len(eval_samples)} {split} questions")
    else:
        eval_samples = test_samples
        print(f"\n[*] Evaluating on FULL held-out {split} set: {len(eval_samples)} questions")

    # Load Model & Tokenizer
    print_banner(f"LOADING MODEL: {model_id}" + (f" + ADAPTER {adapter_path.name}" if adapter_path else ""))
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig

    use_cuda = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    ) if use_cuda else None

    print(f"[*] CUDA Available: {use_cuda}")
    if use_cuda:
        print(f"[*] GPU Device:    {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
    else:
        print("[!] No GPU found: running on CPU in fp32 (slow).")

    start_load = time.time()
    tok_source = str(adapter_path) if adapter_path and (adapter_path / "tokenizer_config.json").exists() else model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_causal_lm(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0} if use_cuda else "cpu",
        dtype=compute_dtype if use_cuda else torch.float32,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    print(f"[OK] Model loaded in {time.time() - start_load:.2f} seconds.")

    # Inference Loop
    print_banner(f"RUNNING {label} INFERENCE ON {len(eval_samples)} SAMPLES")
    predictions = []
    syntax_valid_count = 0
    em_count = 0
    total_latency = 0.0
    stem = results_stem(tag, split)
    ckpt_file = RESULTS_DIR / f"{stem}_checkpoint.json"
    relaxed_count = 0

    for idx, item in enumerate(eval_samples, 1):
        prompt_text = build_prompt(tokenizer, item["db_id"], item["db_schema"], item["question"])
        inputs = tokenizer([prompt_text], return_tensors="pt", add_special_tokens=False).to(model.device)

        if use_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,          # greedy decoding (temperature is irrelevant)
                temperature=None,         # silence "sampling flag" warnings
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,   # Qwen's default 1.1 would penalize repeated SQL tokens (t1.id ...)
                pad_token_id=tokenizer.pad_token_id,
            )
        if use_cuda:
            torch.cuda.synchronize()
        latency = time.perf_counter() - start_time
        total_latency += latency

        input_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[0][input_len:]
        raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        gen_sql = clean_sql(raw_output)

        ref_sql = item["query"]
        is_syntax_valid, err_msg = check_syntax(gen_sql)
        is_em = (normalize_for_em(gen_sql) == normalize_for_em(ref_sql))
        is_relaxed = is_em or relaxed_match(gen_sql, ref_sql)

        if is_syntax_valid:
            syntax_valid_count += 1
        if is_em:
            em_count += 1
        if is_relaxed:
            relaxed_count += 1

        predictions.append({
            "id": item.get("id", f"sample_{idx:05d}"),
            "db_id": item["db_id"],
            "question": item["question"],
            "reference_sql": ref_sql,
            "raw_output": raw_output,
            "generated_sql": gen_sql,
            "syntax_valid": is_syntax_valid,
            "exact_match": is_em,
            "relaxed_match": is_relaxed,
            "error_message": err_msg,
            "latency_seconds": round(latency, 3)
        })

        if idx % 10 == 0 or idx == len(eval_samples):
            current_em = (em_count / idx) * 100
            current_valid = (syntax_valid_count / idx) * 100
            print(f"  [{idx:>4}/{len(eval_samples)}] db: {item['db_id']:<15} | Latency: {latency:.2f}s | "
                  f"EM: {current_em:>5.1f}% | Relaxed: {(relaxed_count / idx) * 100:>5.1f}% | Valid: {current_valid:>5.1f}%")

        if idx % 50 == 0:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(ckpt_file, "w", encoding="utf-8") as f:
                json.dump({"evaluated_count": idx, "predictions": predictions}, f, indent=2)

    # Metrics Summary
    total_evaluated = len(eval_samples)
    syntax_rate = (syntax_valid_count / total_evaluated) * 100
    em_rate = (em_count / total_evaluated) * 100
    relaxed_rate = (relaxed_count / total_evaluated) * 100
    avg_latency = total_latency / total_evaluated

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if use_cuda else 0.0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_report = {
        "model_id": model_id,
        "adapter": str(adapter_path) if adapter_path else None,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": split,
        "total_test_samples": total_evaluated,
        "full_test_set": total_evaluated == len(test_samples),
        "sample_seed": None if total_evaluated == len(test_samples) else SAMPLE_SEED,
        "exact_match_count": em_count,
        "exact_match_accuracy": round(em_rate, 2),
        "relaxed_match_count": relaxed_count,
        "relaxed_match_accuracy": round(relaxed_rate, 2),
        "syntax_valid_count": syntax_valid_count,
        "syntax_valid_rate": round(syntax_rate, 2),
        "invalid_sql_count": total_evaluated - syntax_valid_count,
        "average_latency_seconds": round(avg_latency, 3),
        "peak_vram_allocated_mb": round(peak_vram_mb, 2),
        "unseen_databases_tested": stats.get("audit", {}).get("database_distribution", {}).get("val_unseen_in_train" if split == "validation" else "test_unseen_in_train", 0)
    }

    out_file = RESULTS_DIR / f"{stem}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": summary_report, "predictions": predictions}, f, indent=2)
    if ckpt_file.exists():
        ckpt_file.unlink()  # the full results file supersedes the partial checkpoint
    print(f"\n[OK] Detailed results saved to: {out_file.name}")

    # Report table
    print_banner(f"{label} MODEL EVALUATION REPORT (VIDEO EVIDENCE)")
    print(f"  • Model Tested        : {model_id} ({'4-bit NF4' if use_cuda else 'fp32, CPU'})" + (f" + LoRA adapter '{adapter_path.name}'" if adapter_path else ""))
    print(f"  • Split Evaluated     : {split} - {total_evaluated:,} questions")
    print(f"  • Unseen Databases    : {summary_report['unseen_databases_tested']} (Cross-Database Generalization)")
    print(f"  • Exact-Match SQL     : {em_rate:.2f}% ({em_count}/{total_evaluated})  [normalized string match]")
    print(f"  • Relaxed Match       : {relaxed_rate:.2f}% ({relaxed_count}/{total_evaluated})  [ignores aliases / join order]")
    print(f"  • Syntax Validity Rate: {syntax_rate:.2f}% ({syntax_valid_count}/{total_evaluated})")
    print(f"  • Invalid SQL Cases   : {total_evaluated - syntax_valid_count} failures")
    print(f"  • Average Latency     : {avg_latency:.2f} seconds / query")
    print(f"  • Peak GPU VRAM       : {peak_vram_mb:.1f} MB (~{peak_vram_mb/1024:.2f} GB)")

    # Per-database breakdown: shows whether one odd database drags the whole score down
    per_db: dict = {}
    for pred in predictions:
        row = per_db.setdefault(pred["db_id"], [0, 0, 0, 0])
        row[0] += 1
        row[1] += pred["exact_match"]
        row[2] += pred["relaxed_match"]
        row[3] += pred["syntax_valid"]
    print("  " + "-" * 60)
    print(f"  {'database':<16}{'n':>5}{'exact':>9}{'relaxed':>10}{'valid':>9}")
    for db, (n_db, em_db, rx_db, ok_db) in sorted(per_db.items(), key=lambda kv: -kv[1][0]):
        print(f"  {db:<16}{n_db:>5}{100 * em_db / n_db:>8.1f}%{100 * rx_db / n_db:>9.1f}%{100 * ok_db / n_db:>8.1f}%")

    # Before/after comparison when a matching baseline run exists
    base_file = RESULTS_DIR / f"{results_stem(compare_tag, split)}_results.json"
    if adapter_path and base_file.exists():
        with open(base_file, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        base = base_data["metrics"]
        if base.get("total_test_samples") == total_evaluated:
            print("  " + "-" * 60)
            print(f"  • Compared with : {compare_label}")
            print(f"  • Exact-Match   : {base['exact_match_accuracy']:.2f}% -> {em_rate:.2f}%  ({em_rate - base['exact_match_accuracy']:+.2f} pts)")
            if "relaxed_match_accuracy" in base:
                print(f"  • Relaxed-Match : {base['relaxed_match_accuracy']:.2f}% -> {relaxed_rate:.2f}%  ({relaxed_rate - base['relaxed_match_accuracy']:+.2f} pts)")
            print(f"  • Syntax Valid  : {base['syntax_valid_rate']:.2f}% -> {syntax_rate:.2f}%  ({syntax_rate - base['syntax_valid_rate']:+.2f} pts)")
        else:
            print(f"  [i] {compare_label} was evaluated on {base.get('total_test_samples')} samples; rerun it with the same "
                  f"--sample-size/--all for a like-for-like comparison.")
            base_data = None
    else:
        base_data = None
    print("=" * 80 + "\n")
    if base_data:
        print_paired_comparison(base_data["predictions"], predictions,
                                old_label=compare_label, new_label=adapter_path.name if compare_tag != "baseline" else "Fine-tuned")
    elif adapter_path:
        size_flag = "--all" if total_evaluated == len(test_samples) else f"--sample-size {total_evaluated}"
        how = (f"python baseline_evaluation.py --split {split} {size_flag}" if compare_tag == "baseline"
               else f"python evaluate_finetuned.py --adapter {compare_with} --split {split} {size_flag}")
        print(f"[i] No {compare_label} results to compare with. Run: {how}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the pretrained Qwen baseline on Spider.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("-n", "--sample-size", type=int, default=100, help="Number of test samples to evaluate (default: 100)")
    parser.add_argument("--split", choices=["test", "validation"], default="test",
                        help="Which held-out split to evaluate (default: test)")
    parser.add_argument("--all", action="store_true", help="Evaluate across the entire selected split")
    args = parser.parse_args()

    size = None if args.all else args.sample_size
    evaluate_model(model_id=args.model_id, sample_size=size, split=args.split)


if __name__ == "__main__":
    main()