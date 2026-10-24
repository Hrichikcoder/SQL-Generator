"""
evaluate_finetuned.py
Fine-Tuned Model Evaluation on Multi-Schema Spider Benchmark

Evaluates the QLoRA-fine-tuned Text-to-SQL model (base model + LoRA adapter produced by
train_qlora.py) on the held-out, cross-database test split, and compares it with the
pretrained baseline measured by baseline_evaluation.py.

It reuses the evaluation engine from baseline_evaluation.py (same prompt, same greedy
decoding, same SQL cleaning, same normalized exact-match and syntax checks, same seeded
test sample), so the before/after numbers are directly comparable.

Usage:
  # 1) Baseline first (writes results/baseline_eval_results.json):
  python baseline_evaluation.py --sample-size 100

  # 2) Fine-tuned model on the SAME 100 randomly sampled test questions:
  python evaluate_finetuned.py --sample-size 100

  # Full held-out test split (run the baseline with --all too):
  python evaluate_finetuned.py --all

  # A newer adapter vs the previous one (evaluate the previous adapter first, same options):
  python evaluate_finetuned.py --adapter qlora_adapter    --split validation --sample-size 200
  python evaluate_finetuned.py --adapter qlora_adapter_v2 --split validation --sample-size 200 --compare-with qlora_adapter

  # Held-out VALIDATION databases (same SQL style as the training data):
  python baseline_evaluation.py --split validation --sample-size 200
  python evaluate_finetuned.py  --split validation --sample-size 200

Output: results/finetuned_eval_results.json  (results/finetuned_validation_eval_results.json for --split validation;
        adapters other than 'qlora_adapter' get their own files: finetuned_<adapter-name>_...)
Note: "exact match" is a normalized STRING match against the reference SQL, which is stricter
than Spider's official metrics. Execution accuracy is not possible because the dataset ships
schema text only, not the SQLite database files.
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

try:
    from baseline_evaluation import (
        DEFAULT_MODEL_ID,
        adapter_tag,
        RESULTS_DIR,
        evaluate_model,
        print_banner,
        resolve_adapter_path,
        results_stem,
    )
except ImportError as err:
    print(f"[!] ERROR: could not import baseline_evaluation.py ({err}).")
    print("    Keep evaluate_finetuned.py in the same folder as baseline_evaluation.py.")
    sys.exit(1)

DEFAULT_ADAPTER_DIR = "qlora_adapter"


def read_base_model_id(adapter_path: Path, fallback: str) -> str:
    """The adapter remembers which base model it was trained on; use that unless overridden."""
    try:
        with open(adapter_path / "adapter_config.json", "r", encoding="utf-8") as f:
            return json.load(f).get("base_model_name_or_path") or fallback
    except (OSError, json.JSONDecodeError):
        return fallback


def print_training_summary(adapter_path: Path):
    """Shows what this adapter was trained with (written by train_qlora.py)."""
    summary_file = adapter_path / "training_summary.json"
    if not summary_file.exists():
        return
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
    hp = summary.get("hyperparameters", {})
    res = summary.get("results", {})
    print_banner("ADAPTER TRAINING SUMMARY (from train_qlora.py)")
    print(f"  • Trained on      : {summary.get('model_id')} ({summary.get('completion_timestamp')})")
    print(f"  • Training time   : {summary.get('training_time_formatted')}  |  steps: {res.get('total_training_steps')}")
    print(f"  • Epochs / LR     : {hp.get('epochs')} / {hp.get('learning_rate')}  |  LoRA r={hp.get('lora_r')}, alpha={hp.get('lora_alpha')}")
    print(f"  • Validation loss : {res.get('final_val_loss')}  |  mean train loss: {res.get('mean_train_loss', res.get('final_train_loss'))}")


def print_failure_examples(split: str = "test", tag: str = "finetuned", max_examples: int = 3):
    """Prints a few wrong predictions so you can see HOW the model fails, not just how often."""
    results_file = RESULTS_DIR / f"{results_stem(tag, split)}_results.json"
    if not results_file.exists():
        return
    with open(results_file, "r", encoding="utf-8") as f:
        predictions = json.load(f).get("predictions", [])
    failures = [p for p in predictions if not p["exact_match"]]
    if not failures:
        return
    print_banner(f"SAMPLE MISSES ({min(max_examples, len(failures))} of {len(failures)} non-exact-match predictions)")
    for p in failures[:max_examples]:
        print(f"  db: {p['db_id']}  |  syntax valid: {p['syntax_valid']}  |  relaxed match: {p.get('relaxed_match')}")
        print(f"  Q  : {p['question']}")
        print(f"  REF: {p['reference_sql']}")
        print(f"  GEN: {p['generated_sql'] or '(no SQL extracted) ' + p.get('raw_output', '')[:120]}")
        print("  " + "-" * 76)
    print("  Many misses are equivalent SQL written differently (e.g. aliases, join order);")
    print("  exact match under-counts correct queries. Read the misses before judging the model.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the QLoRA fine-tuned Text-to-SQL adapter on the held-out Spider test split.")
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER_DIR,
                        help=f"Path to the trained LoRA adapter folder (default: {DEFAULT_ADAPTER_DIR})")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Base model ID. Default: read from the adapter's adapter_config.json")
    parser.add_argument("-n", "--sample-size", type=int, default=100,
                        help="Number of test samples to evaluate (default: 100; use the same value as the baseline run)")
    parser.add_argument("--compare-with", type=str, default=None,
                        help="Compare against another adapter's results (folder name, e.g. qlora_adapter) instead of the "
                             "pretrained baseline. Evaluate that adapter first, with the same --split and --sample-size.")
    parser.add_argument("--split", choices=["test", "validation"], default="test",
                        help="Which held-out split to evaluate (default: test)")
    parser.add_argument("--all", action="store_true", help="Evaluate across the entire selected split")
    args = parser.parse_args()

    adapter_path = resolve_adapter_path(args.adapter)  # exits with a clear message if training has not been run
    model_id = args.model_id or read_base_model_id(adapter_path, DEFAULT_MODEL_ID)

    print_training_summary(adapter_path)
    size = None if args.all else args.sample_size
    evaluate_model(model_id=model_id, sample_size=size, adapter=str(adapter_path), split=args.split,
                   compare_with=args.compare_with)
    print_failure_examples(args.split, adapter_tag(adapter_path.name))


if __name__ == "__main__":
    main()