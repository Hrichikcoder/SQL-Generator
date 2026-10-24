"""
load_dataset.py
Multi-Schema Text-to-SQL Dataset Loader

Loads the multi-schema Spider Text-to-SQL benchmark dataset from Hugging Face
(SuperMax991/spider-text2sql), inspects its structure, metadata, features, and
existing splits, and exports the raw records locally for offline preprocessing.

Dataset Source: https://huggingface.co/datasets/SuperMax991/spider-text2sql
"""

import json
import sys
import time
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_NAME = "SuperMax991/spider-text2sql"


def print_banner(title: str):
    """Prints a structured section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def load_and_inspect_dataset():
    """Loads the dataset from Hugging Face and logs full metadata."""
    print_banner(f"STEP 1: LOADING DATASET FROM HUGGING FACE\n Source: {DATASET_NAME}")

    try:
        from datasets import load_dataset
    except ImportError:
        print("[!] ERROR: 'datasets' package is not installed.")
        print("    Please run: pip install datasets")
        sys.exit(1)

    print(f"[*] Connecting to Hugging Face Hub...")
    print(f"[*] Downloading / loading repository: '{DATASET_NAME}'...")
    start_time = time.time()
    
    try:
        dataset_dict = load_dataset(DATASET_NAME)
    except Exception as e:
        print(f"[!] Failed to load dataset from Hugging Face: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"[OK] Dataset successfully fetched in {elapsed:.2f} seconds.")

    # Create destination directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Inspect splits
    splits = list(dataset_dict.keys())
    print_banner(f"DATASET SPLIT & FEATURE OVERVIEW")
    print(f"[*] Available Splits in Repository: {splits}")

    total_records = 0
    all_dbs = set()
    split_summaries = {}

    for split_name in splits:
        split_data = dataset_dict[split_name]
        split_count = len(split_data)
        total_records += split_count

        # Extract features/columns
        columns = list(split_data.column_names)
        
        # Count unique databases and questions
        unique_dbs = set(split_data["db_id"]) if "db_id" in split_data.column_names else set()
        unique_questions = set(split_data["question"]) if "question" in split_data.column_names else set()
        all_dbs.update(unique_dbs)

        split_summaries[split_name] = {
            "record_count": split_count,
            "unique_databases": len(unique_dbs),
            "unique_questions": len(unique_questions),
            "columns": columns,
            "databases": sorted(list(unique_dbs))
        }

        print(f"\n  Split: '{split_name.upper()}'")
        print(f"    - Total Rows / Examples:  {split_count:,}")
        print(f"    - Unique Databases:       {len(unique_dbs):,}")
        print(f"    - Unique Questions:       {len(unique_questions):,}")
        print(f"    - Columns / Features:     {columns}")

        # Warn early if the columns later steps depend on are missing
        has_sql = "query" in columns or "sql" in columns
        has_schema = "db_schema" in columns or "schema" in columns
        if not ({"db_id", "question"} <= set(columns) and has_sql and has_schema):
            print(f"    [!] WARNING: split '{split_name}' lacks one of db_id / question / query / db_schema.")
            print("        Later steps need all four; check the dataset card for the real column names.")

    # Display full feature schema from first available split
    first_split = splits[0]
    sample_item = dataset_dict[first_split][0]

    print_banner("SAMPLE RECORD STRUCTURE (RAW)")
    print(f"[*] First record from '{first_split}' split:")
    sample_preview = {k: v for k, v in sample_item.items() if k != "question_toks"}
    if "question_toks" in sample_item:
        sample_preview["question_toks"] = sample_item["question_toks"][:8] + ["..."]
    print(json.dumps(sample_preview, indent=2))

    # Save raw records locally
    print_banner("SAVING RAW DATA LOCALLY")
    for split_name in splits:
        records = [dict(row) for row in dataset_dict[split_name]]
        out_path = RAW_DATA_DIR / f"raw_{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved {len(records):,} raw records to: {out_path.name}")

    # Save summary metadata
    meta = {
        "dataset_source": f"https://huggingface.co/datasets/{DATASET_NAME}",
        "dataset_identifier": DATASET_NAME,
        "loaded_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_records": total_records,
        "total_unique_databases": len(all_dbs),
        "splits": split_summaries
    }
    meta_path = RAW_DATA_DIR / "dataset_info.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved dataset metadata log to: {meta_path.name}")

    # Summary table for presentation / video
    splits_str = ", ".join(f"{k} ({v['record_count']:,})" for k, v in split_summaries.items())
    print_banner("DATASET LOADING SUMMARY (VIDEO EVIDENCE)")
    print(f"  • Source Repository   : {DATASET_NAME}")
    print(f"  • Total Examples      : {total_records:,}")
    print(f"  • Unique Databases    : {len(all_dbs):,} (Multi-Schema Cross-Domain)")
    print(f"  • Splits Discovered   : {splits_str}")
    print(f"  • Core Schema Columns : db_id, question, query, db_schema")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    load_and_inspect_dataset()