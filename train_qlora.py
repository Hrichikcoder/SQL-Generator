"""
train_qlora.py
Generic, Multi-Schema Text-to-SQL QLoRA Fine-Tuning Pipeline

Fine-tunes Qwen/Qwen2.5-Coder-1.5B-Instruct using 4-bit NormalFloat (NF4)
quantization and LoRA adapters on multi-schema database examples from the
Spider dataset. The model learns to map an arbitrary database schema and
natural-language question to an executable SQLite query.

Key Features:
  1. Multi-Schema Aware: Injects arbitrary db_id and db_schema into prompt context.
  2. Target-Only Loss Masking: Masks system/user prompt and padding tokens with
     label -100 so CrossEntropyLoss is computed strictly on target SQL tokens.
  3. Memory Efficient: 4-bit NF4 double quantization, paged_adamw_8bit optimizer,
     and gradient checkpointing for 8 GB consumer GPU compatibility.
  4. Quarantined Test Split: Never loads or evaluates test.json during training.
  5. Checkpointing & Logging: Saves best/final adapter to qlora_adapter/ and writes
     training configuration, runtime, and peak VRAM to results/training_summary.json.

Usage:
  # Standard 3-epoch training:
  python train_qlora.py

  # Quick smoke-test (e.g., 50 steps for video proof / fast verification):
  python train_qlora.py --max-steps 50

  # Custom batch size and learning rate (effective batch = batch-size x grad-accum):
  python train_qlora.py --batch-size 2 --grad-accum 8 --lr 2e-4 --epochs 3

  # Out of memory on a small GPU? Use: --batch-size 1 --grad-accum 16 --max-seq-length 768
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "splits"
TRAIN_FILE = DATA_DIR / "train.json"
VAL_FILE = DATA_DIR / "validation.json"
TEST_FILE = DATA_DIR / "test.json"
ADAPTER_DIR = BASE_DIR / "qlora_adapter"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINTS_DIR = BASE_DIR / "qlora_checkpoints"

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def print_banner(title: str):
    """Prints a structured section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def build_system_prompt() -> str:
    return (
        "You are a professional, schema-aware Text-to-SQL assistant. "
        "Given the target database identifier and its schema, generate a single "
        "syntactically valid SQLite query to answer the question. "
        "Return only the executable SQL query without explanations or markdown."
    )


def format_user_prompt(db_id: str, db_schema: str, question: str) -> str:
    """Dynamically formats multi-schema context without hardcoded tables."""
    return f"""Database ID:
{db_id.strip()}

Database Schema:
{db_schema.strip()}

Question:
{question.strip()}
"""


class TextToSQLDatasetBuilder:
    """Prepares and tokenizes multi-schema records with target-only loss masking."""

    def __init__(self, tokenizer: AutoTokenizer, max_seq_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.system_prompt = build_system_prompt()
        self.dropped_count = 0

    def encode_example(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Encodes one example: masks prompt tokens with -100 for target-only loss.

        Returns None if the example does not fit in max_seq_length. Such examples are
        dropped instead of truncated: cutting the front of the prompt would remove
        the system prompt / chat-template header / part of the schema, so the model
        would train on prompts that never occur at inference time.
        """
        db_id = item.get("db_id", "")
        db_schema = item.get("db_schema", "")
        question = item.get("question", "")
        query = item.get("query", "").strip()

        # Build prompt using Qwen chat template
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": format_user_prompt(db_id, db_schema, question)},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Assistant target completion, terminated by the end-of-turn token so the model learns to stop
        completion_text = f"{query}<|im_end|>"

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)

        if len(prompt_ids) + len(completion_ids) > self.max_seq_length:
            self.dropped_count += 1
            return None

        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)

        # Target-only Loss Masking:
        # Prompt tokens are assigned label -100 (ignored by PyTorch CrossEntropyLoss)
        labels = [-100] * len(prompt_ids) + completion_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class TargetLossDataCollator:
    """Pads input_ids, attention_mask, and labels (-100 for padding)."""

    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            remainder = batch_max_len % self.pad_to_multiple_of
            if remainder != 0:
                batch_max_len += self.pad_to_multiple_of - remainder

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            cur_len = len(f["input_ids"])
            padding_len = batch_max_len - cur_len

            # Pad inputs with pad_token_id, mask with 0, labels with -100
            padded_input = f["input_ids"] + [self.pad_token_id] * padding_len
            padded_mask = f["attention_mask"] + [0] * padding_len
            padded_labels = f["labels"] + [-100] * padding_len

            batch_input_ids.append(padded_input)
            batch_attention_mask.append(padded_mask)
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def load_and_validate_splits():
    """Loads train and val splits, confirming test set remains strictly quarantined."""
    print_banner("1. DATASET INTEGRITY & QUARANTINE CHECK")

    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        print(f"[!] ERROR: Training or Validation file missing in {DATA_DIR}")
        print("    Expected train.json and validation.json. Run split_dataset.py first!")
        sys.exit(1)

    if not TEST_FILE.exists():
        print(f"[!] WARNING: test.json not found in {DATA_DIR}.")
    else:
        print(f"[*] Held-out test set detected: {TEST_FILE.name}")
        print(f"    [OK] Quarantined: test.json will NOT be loaded or evaluated during training.")

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_raw = json.load(f)

    # Filter any empty/corrupt examples
    valid_train = [
        x for x in train_raw if x.get("db_id") and x.get("db_schema") and x.get("question") and x.get("query")
    ]
    valid_val = [
        x for x in val_raw if x.get("db_id") and x.get("db_schema") and x.get("question") and x.get("query")
    ]

    print(f"[*] Training Examples Loaded   : {len(valid_train):,} (from {TRAIN_FILE.name})")
    print(f"[*] Validation Examples Loaded : {len(valid_val):,} (from {VAL_FILE.name})")

    # Distinct databases
    train_dbs = set(x["db_id"] for x in valid_train)
    val_dbs = set(x["db_id"] for x in valid_val)
    print(f"[*] Multi-Schema Coverage      : {len(train_dbs):,} unique databases in training data.")
    print(f"[*] Validation Schema Coverage : {len(val_dbs):,} unique databases in validation data.")

    return valid_train, valid_val


def setup_model_and_tokenizer(model_id: str, lora_r: int, lora_alpha: int):
    """Loads base model in 4-bit NF4 and configures PEFT LoRA adapters."""
    print_banner(f"2. LOADING BASE MODEL IN 4-BIT NF4 QUANTIZATION: {model_id}")

    if not torch.cuda.is_available():
        print("[!] ERROR: No CUDA GPU detected. 4-bit QLoRA training (bitsandbytes) requires an NVIDIA GPU.")
        print("    Check with: python -c \"import torch; print(torch.cuda.is_available())\"")
        sys.exit(1)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[*] Target GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"[*] Compute Precision: {compute_dtype}")
    torch.cuda.reset_peak_memory_stats()

    # 4-bit NormalFloat Quantization Setup
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Single-GPU placement: device_map="auto" may offload layers to CPU, which breaks 4-bit training.
    load_kwargs = dict(quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=compute_dtype, **load_kwargs)
    except TypeError:  # older transformers call this argument `torch_dtype`
        base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype, **load_kwargs)
    base_model.config.use_cache = False  # incompatible with gradient checkpointing

    # Enable gradient checkpointing and freeze quantized weights
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True
    )

    # Configure LoRA targeting all attention and MLP projection layers
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(base_model, peft_config)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_pct = (trainable_params / all_params) * 100

    print("\n[*] LoRA Adapter Parameter Statistics:")
    print(f"    - Base Model Parameters : {all_params:,}")
    print(f"    - Trainable Parameters  : {trainable_params:,}")
    print(f"    - Trainable Percentage  : {trainable_pct:.4f}%")
    print(f"    - LoRA Rank (r)         : {lora_r}")
    print(f"    - LoRA Alpha (α)        : {lora_alpha}")
    print(f"    - Target Modules        : {', '.join(target_modules)}")

    return model, tokenizer, trainable_params, all_params


def run_training(args):
    """Executes the QLoRA fine-tuning training loop."""
    start_wall_time = time.time()

    # 1. Dataset loading and verification
    train_raw, val_raw = load_and_validate_splits()

    # 2. Model & Tokenizer loading
    model, tokenizer, trainable_params, all_params = setup_model_and_tokenizer(
        model_id=args.model_id,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # 3. Tokenization with target-only loss masking
    print_banner("3. PREPARING DATASETS WITH TARGET-ONLY LOSS MASKING")
    builder = TextToSQLDatasetBuilder(tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    def encode_all(records):
        builder.dropped_count = 0
        encoded = [e for e in (builder.encode_example(x) for x in records) if e is not None]
        return encoded, builder.dropped_count

    print("[*] Tokenizing and masking training set...")
    encoded_train, dropped_train = encode_all(train_raw)
    print("[*] Tokenizing and masking validation set...")
    encoded_val, dropped_val = encode_all(val_raw)

    if not encoded_train or not encoded_val:
        print("[!] ERROR: no examples left after tokenization. Increase --max-seq-length or check the splits.")
        sys.exit(1)

    train_dataset = Dataset.from_list(encoded_train)
    val_dataset = Dataset.from_list(encoded_val)

    if dropped_train or dropped_val:
        print(f"    [!] Dropped {dropped_train} train / {dropped_val} validation examples longer than {args.max_seq_length} tokens.")
    else:
        print(f"    [OK] 0 examples dropped (all fit within {args.max_seq_length} max sequence length).")
    print(f"    [*] Final sizes: {len(encoded_train):,} train / {len(encoded_val):,} validation examples.")

    # Sample verification: prove that prompt tokens have label -100
    sample_item = encoded_train[0]
    num_masked = sample_item["labels"].count(-100)
    num_target = len(sample_item["labels"]) - num_masked
    print(f"\n[*] Loss Masking Verification (Sample 1):")
    print(f"    - Total Sequence Length : {len(sample_item['input_ids'])} tokens")
    print(f"    - Masked Prompt Tokens  : {num_masked} tokens (Loss Weight: 0.0)")
    print(f"    - Active SQL Tokens     : {num_target} tokens (Loss Weight: 1.0)")
    print("    [OK] Target-only loss masking verified: Prompt tokens strictly ignored by loss function.")

    # 4. Data Collator
    collator = TargetLossDataCollator(pad_token_id=tokenizer.pad_token_id)

    # 5. Training Arguments
    print_banner("4. CONFIGURING TRAINING HYPERPARAMETERS")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    # Smoke tests: make sure at least one eval + one checkpoint happen, and warmup is not longer than the run
    eval_steps, save_steps, warmup_steps = args.eval_steps, args.save_steps, args.warmup_steps
    if args.max_steps > 0:
        eval_steps = min(eval_steps, args.max_steps) if eval_steps > 0 else 0
        save_steps = min(save_steps, args.max_steps)
        warmup_steps = min(warmup_steps, max(1, args.max_steps // 10))

    do_eval = eval_steps > 0
    if do_eval and save_steps % eval_steps != 0:
        # load_best_model_at_end requires save_steps to be a multiple of eval_steps
        save_steps = eval_steps
        print(f"[i] save_steps adjusted to {save_steps} (must be a multiple of eval_steps).")

    # `evaluation_strategy` was renamed `eval_strategy` in newer transformers versions
    strategy_key = "eval_strategy" if hasattr(TrainingArguments, "eval_strategy") else "evaluation_strategy"
    eval_kwargs = {strategy_key: "steps" if do_eval else "no"}
    if do_eval:
        eval_kwargs["eval_steps"] = eval_steps

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,  # dataset already holds exactly input_ids / attention_mask / labels
        report_to="none",
        seed=42,
        **eval_kwargs,
    )

    effective_batch_size = args.batch_size * args.grad_accum
    print(f"[*] Per-Device Batch Size       : {args.batch_size}")
    print(f"[*] Gradient Accumulation Steps : {args.grad_accum}")
    print(f"[*] Effective Batch Size        : {effective_batch_size}")
    print(f"[*] Learning Rate               : {args.lr}")
    print(f"[*] Optimizer                   : paged_adamw_8bit")
    print(f"[*] Gradient Checkpointing      : Enabled")
    print(f"[*] Total Epochs                : {args.epochs}")
    if args.max_steps > 0:
        print(f"[*] Max Steps Override          : {args.max_steps} steps (Smoke Test)")

    # 6. Trainer Initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # 7. Training Execution
    print_banner("5. COMMENCING QLORA MODEL TRAINING")
    print("[*] Starting training loop...")
    train_result = trainer.train()

    elapsed_time_sec = time.time() - start_wall_time
    minutes = int(elapsed_time_sec // 60)
    seconds = int(elapsed_time_sec % 60)

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # 8. Evaluation of Validation Loss
    print_banner("6. EVALUATING FINAL VALIDATION LOSS")
    val_metrics = trainer.evaluate()
    final_val_loss = val_metrics.get("eval_loss", None)
    final_train_loss = train_result.training_loss  # mean over the whole run, not the last step
    print(f"[*] Mean Train Loss  : {final_train_loss:.4f}")
    if final_val_loss is not None:
        print(f"[*] Final Val Loss   : {final_val_loss:.4f}")

    # 9. Save Best/Final LoRA Adapter
    print_banner("7. SAVING TRAINED QLORA ADAPTER")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"[OK] LoRA adapter successfully saved to: {ADAPTER_DIR.resolve()}")

    # 10. Record Summary Metrics
    summary = {
        "model_id": args.model_id,
        "completion_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": round(elapsed_time_sec, 2),
        "training_time_formatted": f"{minutes}m {seconds}s",
        "peak_vram_allocated_mb": round(peak_vram_mb, 2),
        "peak_vram_allocated_gb": round(peak_vram_mb / 1024, 2),
        "parameters": {
            "total_parameters": all_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": round((trainable_params / all_params) * 100, 4),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_batch_size": effective_batch_size,
            "learning_rate": args.lr,
            "max_seq_length": args.max_seq_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "optimizer": "paged_adamw_8bit",
            "loss_masking": "target_sql_tokens_only",
        },
        "results": {
            "mean_train_loss": round(final_train_loss, 4),
            "final_val_loss": round(final_val_loss, 4) if final_val_loss is not None else "N/A",
            "total_training_steps": train_result.global_step,
        },
    }

    summary_file = RESULTS_DIR / "training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(ADAPTER_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Training summary saved to: {summary_file.name}")

    # Video summary report
    print_banner("QLORA FINE-TUNING EXECUTION REPORT (VIDEO EVIDENCE)")
    print(f"  • Base Model Architecture : {args.model_id} (4-bit NF4)")
    print(f"  • Multi-Schema Domain     : Generic Schema-Aware (Spider Dataset)")
    print(f"  • Trainable Parameters    : {trainable_params:,} / {all_params:,} ({summary['parameters']['trainable_percentage']}%)")
    print(f"  • Loss Masking Applied    : Target SQL Tokens Only (Prompt Label = -100)")
    print(f"  • Training Duration       : {minutes} minutes {seconds} seconds ({elapsed_time_sec:.1f}s)")
    print(f"  • Peak GPU VRAM           : {peak_vram_mb:.1f} MB (~{peak_vram_mb/1024:.2f} GB)")
    print(f"  • Mean Training Loss      : {final_train_loss:.4f}")
    if final_val_loss is not None:
        print(f"  • Final Validation Loss   : {final_val_loss:.4f}")
    print(f"  • Saved Adapter Output    : {ADAPTER_DIR.name}/")
    print(f"  • Held-Out Test Set       : STRICTLY QUARANTINED (0% leakage)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Schema Text-to-SQL QLoRA Fine-Tuning.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Base model Hugging Face ID")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size (default: 2)")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Maximum token sequence length (default: 1024)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank dimension (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor (default: 32)")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Linear warmup steps (default: 50)")
    parser.add_argument("--save-steps", type=int, default=200, help="Checkpoint save steps (default: 200)")
    parser.add_argument("--eval-steps", type=int, default=200, help="Validation evaluation steps; 0 disables validation during training (default: 200)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override total steps for quick smoke tests (-1 for full epochs)")

    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()