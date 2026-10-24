"""
split_dataset.py
Multi-Schema Text-to-SQL Dataset Split & Data-Leakage Audit Pipeline

Inspects existing splits, checks for data leakage (question overlap and database
distribution), preserves existing splits if already complete, and creates a
proper 3-way split (Train / Validation / Test) if a validation split is missing,
while also preserving the original 2-way benchmark split.

Input:  data/processed/processed_*.json
Output: data/splits/train.json
        data/splits/validation.json
        data/splits/test.json
        data/splits/original_train.json
        data/splits/original_test.json
        data/splits/dataset_full.json
        data/splits/split_stats.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Deterministic Seed for Reproducibility
RANDOM_SEED = 42


def print_banner(title: str):
    """Prints a structured section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def group_split_by_db(records: list[dict], fraction: float, seed: int = RANDOM_SEED) -> tuple[list[dict], list[dict]]:
    """Splits records by WHOLE databases (not by row).

    Returns (held_out, remaining). The held-out part holds ~`fraction` of the
    records and its databases never appear in `remaining`. This keeps the
    validation set cross-database, like the test set, so validation loss is an
    honest signal for 'works on a schema it has never seen'.
    """
    by_db: dict[str, list[dict]] = {}
    for r in records:
        by_db.setdefault(r["db_id"], []).append(r)

    db_ids = sorted(by_db)
    random.Random(seed).shuffle(db_ids)

    target = int(len(records) * fraction)
    held_out, remaining = [], []
    for i, db in enumerate(db_ids):
        # Fill held-out until the target is reached; the last DB always stays in `remaining`
        if len(held_out) < target and i < len(db_ids) - 1:
            held_out.extend(by_db[db])
        else:
            remaining.extend(by_db[db])
    return held_out, remaining


def check_data_leakage(train_data: list[dict], val_data: list[dict], test_data: list[dict]) -> dict:
    """Verifies that no example leaks between splits and analyzes database distributions.

    Two notions of overlap are reported:
      * identical (db_id, question) pairs  -> real leakage (the same task twice)
      * identical question text only       -> informational; Spider reuses generic
        questions ('How many singers are there?') across different databases,
        which is not leakage because the schema differs.
    """
    def qs(data):
        return set(d["question"].strip().lower() for d in data)

    def pairs(data):
        return set((d["db_id"].strip().lower(), d["question"].strip().lower()) for d in data)

    train_qs, val_qs, test_qs = qs(train_data), qs(val_data), qs(test_data)
    train_p, val_p, test_p = pairs(train_data), pairs(val_data), pairs(test_data)

    leak_train_val = len(train_p & val_p)
    leak_train_test = len(train_p & test_p)
    leak_val_test = len(val_p & test_p)

    # Database distribution
    train_dbs = set(d["db_id"] for d in train_data)
    val_dbs = set(d["db_id"] for d in val_data)
    test_dbs = set(d["db_id"] for d in test_data)
    all_dbs = train_dbs.union(val_dbs).union(test_dbs)

    return {
        "question_leakage": {
            "train_val_overlap": leak_train_val,
            "train_test_overlap": leak_train_test,
            "val_test_overlap": leak_val_test,
            "zero_leakage_verified": (leak_train_val == 0 and leak_train_test == 0 and leak_val_test == 0),
            "generic_question_text_overlap": {
                "train_val": len(train_qs & val_qs),
                "train_test": len(train_qs & test_qs),
                "val_test": len(val_qs & test_qs),
            },
        },
        "database_distribution": {
            "total_unique_databases": len(all_dbs),
            "train_unique_databases": len(train_dbs),
            "val_unique_databases": len(val_dbs),
            "test_unique_databases": len(test_dbs),
            "val_unseen_in_train": len(val_dbs - train_dbs),
            "test_unseen_in_train": len(test_dbs - train_dbs),
            "test_shared_with_train": len(test_dbs.intersection(train_dbs))
        }
    }


def run_split_pipeline(force: bool = False):
    """Inspects, validates, and partitions dataset splits with zero leakage."""
    print_banner("STEP 3: DATASET SPLIT INSPECTION & DATA LEAKAGE AUDIT")

    if not PROCESSED_DATA_DIR.exists():
        print(f"[!] ERROR: Processed data directory not found at: {PROCESSED_DATA_DIR}")
        print("    Please run: python preprocess_dataset.py first!")
        sys.exit(1)

    processed_files = sorted(list(PROCESSED_DATA_DIR.glob("processed_*.json")))
    if not processed_files:
        print(f"[!] ERROR: No processed files found in: {PROCESSED_DATA_DIR}")
        print("    Please run: python preprocess_dataset.py first!")
        sys.exit(1)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all processed splits
    processed_splits = {}
    for p in processed_files:
        split_name = p.stem.replace("processed_", "")
        with open(p, "r", encoding="utf-8") as f:
            processed_splits[split_name] = json.load(f)
        print(f"[*] Found preprocessed split: '{split_name.upper()}' with {len(processed_splits[split_name]):,} records.")

    # 1. Check if complete, valid 3-way splits ALREADY exist in SPLITS_DIR
    target_3way_files = [SPLITS_DIR / "train.json", SPLITS_DIR / "validation.json", SPLITS_DIR / "test.json"]
    all_3way_exist = all(f.exists() and f.stat().st_size > 0 for f in target_3way_files)
    if all_3way_exist and force:
        print("\n[i] --force given: ignoring existing split files and re-splitting from processed data.")
        all_3way_exist = False

    if all_3way_exist:
        print("\n[i] Detected existing 3-way splits in data/splits/ (train.json, validation.json, test.json).")
        print("[*] Verifying existing splits without re-splitting...")
        with open(SPLITS_DIR / "train.json", "r", encoding="utf-8") as f:
            train_set = json.load(f)
        with open(SPLITS_DIR / "validation.json", "r", encoding="utf-8") as f:
            val_set = json.load(f)
        with open(SPLITS_DIR / "test.json", "r", encoding="utf-8") as f:
            test_set = json.load(f)
    else:
        print("\n[*] Inspecting split structure...")
        # Check available processed splits
        has_train = "train" in processed_splits
        has_test = "test" in processed_splits
        has_val = "validation" in processed_splits or "dev" in processed_splits

        # Save both splitting perspectives:
        # A) Original Benchmark Splits (Preserve exactly what was loaded)
        for s_name, s_data in processed_splits.items():
            orig_path = SPLITS_DIR / f"original_{s_name}.json"
            with open(orig_path, "w", encoding="utf-8") as f:
                json.dump(s_data, f, indent=2, ensure_ascii=False)
            print(f"[OK] Preserved original benchmark split to: {orig_path.name} ({len(s_data):,} records)")

        # B) Standard 3-Way Fine-Tuning Split (Train / Validation / Test)
        if has_train and has_test and not has_val:
            print("\n[*] Raw dataset contains 2 splits (Train and Test), missing dedicated Validation split.")
            print("[*] Creating 3-way split: holding out ~10% of Train as Validation (Seed: 42).")
            print("    Split is done by WHOLE DATABASE so validation is cross-database like the test set.")

            test_set = processed_splits["test"]
            val_set, train_set = group_split_by_db(processed_splits["train"], 0.10, RANDOM_SEED)

        elif has_train and has_val and has_test:
            print("\n[*] Valid 3-way splits (Train, Validation, Test) already present in processed data.")
            train_set = processed_splits["train"]
            val_key = "validation" if "validation" in processed_splits else "dev"
            val_set = processed_splits[val_key]
            test_set = processed_splits["test"]

        else:
            # Fallback if unpartitioned: combine all and partition ~80/10/10 by whole database
            print("\n[*] Partitioning combined dataset into ~80/10/10 Train/Val/Test (by database)...")
            all_records = []
            for s_data in processed_splits.values():
                all_records.extend(s_data)

            test_set, rest = group_split_by_db(all_records, 0.10, RANDOM_SEED)
            val_set, train_set = group_split_by_db(rest, 0.10 / 0.90, RANDOM_SEED + 1)

        # Save target 3-way split files
        print("\n[*] Writing standardized 3-way split files...")
        with open(SPLITS_DIR / "train.json", "w", encoding="utf-8") as f:
            json.dump(train_set, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Saved Train Set:      {SPLITS_DIR / 'train.json'} ({len(train_set):,} records)")

        with open(SPLITS_DIR / "validation.json", "w", encoding="utf-8") as f:
            json.dump(val_set, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Saved Validation Set: {SPLITS_DIR / 'validation.json'} ({len(val_set):,} records)")

        with open(SPLITS_DIR / "test.json", "w", encoding="utf-8") as f:
            json.dump(test_set, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Saved Test Set:       {SPLITS_DIR / 'test.json'} ({len(test_set):,} records)")

    # Unified Master Dataset
    full_dataset = train_set + val_set + test_set
    full_path = SPLITS_DIR / "dataset_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved Full Dataset:   {full_path.name} ({len(full_dataset):,} total records)")

    # 2. Comprehensive Data Leakage & Schema Distribution Audit
    print_banner("RUNNING DATA LEAKAGE AUDIT")
    audit_results = check_data_leakage(train_set, val_set, test_set)
    q_leak = audit_results["question_leakage"]
    db_dist = audit_results["database_distribution"]

    total_samples = len(full_dataset)
    train_pct = len(train_set) / total_samples * 100
    val_pct = len(val_set) / total_samples * 100
    test_pct = len(test_set) / total_samples * 100

    print(f"[*] Overlap of identical (db_id, question) pairs (true leakage check):")
    print(f"    - Train <-> Validation Overlap : {q_leak['train_val_overlap']} questions")
    print(f"    - Train <-> Test Overlap       : {q_leak['train_test_overlap']} questions")
    print(f"    - Validation <-> Test Overlap  : {q_leak['val_test_overlap']} questions")
    
    if q_leak["zero_leakage_verified"]:
        print(f"    [OK] STRICT SEPARATION VERIFIED: ZERO (0%) DUPLICATE (db_id, question) EXAMPLES ACROSS SPLITS!")
        g = q_leak["generic_question_text_overlap"]
        print(f"    [i] Generic question text shared across different DBs (not leakage): "
              f"train/val={g['train_val']}, train/test={g['train_test']}, val/test={g['val_test']}")
    else:
        print(f"    [!] WARNING: identical (db_id, question) examples detected between splits!")

    print(f"\n[*] Multi-Database Distribution:")
    print(f"    - Total Unique Databases       : {db_dist['total_unique_databases']} databases")
    print(f"    - Train Databases              : {db_dist['train_unique_databases']} databases")
    print(f"    - Validation Databases         : {db_dist['val_unique_databases']} databases")
    print(f"    - Test Databases               : {db_dist['test_unique_databases']} databases")
    print(f"    - Validation DBs Unseen in Train: {db_dist['val_unseen_in_train']}")
    print(f"    - Test Databases Unseen in Train: {db_dist['test_unseen_in_train']} (Cross-Database Evaluation)")
    print(f"    - Test Databases Shared w/ Train: {db_dist['test_shared_with_train']} (In-Domain Evaluation)")

    # Save complete statistical log
    split_meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_records": total_samples,
        "splits": {
            "train": {"count": len(train_set), "percentage": round(train_pct, 2)},
            "validation": {"count": len(val_set), "percentage": round(val_pct, 2)},
            "test": {"count": len(test_set), "percentage": round(test_pct, 2)}
        },
        "audit": audit_results
    }
    stats_out = SPLITS_DIR / "split_stats.json"
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
    print(f"\n[OK] Split statistics and leakage audit logged to: {stats_out.name}")

    # Video summary table
    print_banner("DATASET SPLIT SUMMARY TABLE (VIDEO EVIDENCE)")
    print(f"  +----------------+---------------+------------+--------------------+")
    print(f"  | Split Name     | Sample Count  | Percentage | Unique Databases   |")
    print(f"  +----------------+---------------+------------+--------------------+")
    print(f"  | Train Set      | {len(train_set):>13,} | {train_pct:>9.2f}% | {db_dist['train_unique_databases']:>18} |")
    print(f"  | Validation Set | {len(val_set):>13,} | {val_pct:>9.2f}% | {db_dist['val_unique_databases']:>18} |")
    print(f"  | Test Set       | {len(test_set):>13,} | {test_pct:>9.2f}% | {db_dist['test_unique_databases']:>18} |")
    print(f"  +----------------+---------------+------------+--------------------+")
    print(f"  | TOTAL          | {total_samples:>13,} |   100.00%  | {db_dist['total_unique_databases']:>18} |")
    print(f"  +----------------+---------------+------------+--------------------+")
    print(f"  • Data Leakage Status: {'ZERO DUPLICATE EXAMPLES ACROSS SPLITS' if q_leak['zero_leakage_verified'] else 'OVERLAP DETECTED - SEE ABOVE'}")
    print(f"  • Cross-Database Generalization: {db_dist['test_unseen_in_train']} unseen test databases tested.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/validation/test splits and audit leakage.")
    parser.add_argument("--force", action="store_true",
                        help="Re-split from data/processed even if data/splits/{train,validation,test}.json already exist "
                             "(use this after re-running preprocess_dataset.py)")
    run_split_pipeline(force=parser.parse_args().force)