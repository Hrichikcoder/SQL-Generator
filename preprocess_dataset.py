"""
preprocess_dataset.py
Multi-Schema Text-to-SQL Dataset Preprocessing Pipeline

Cleans, validates, and standardizes raw Spider Text-to-SQL records across all
splits. Enforces required fields, removes duplicate and invalid records,
normalizes whitespace and SQL statements, and preserves cross-database schema
definitions (db_id, db_schema).

Input:  data/raw/raw_<split>.json
Output: data/processed/processed_<split>.json
        data/processed/preprocessing_stats.json
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def print_banner(title: str):
    """Prints a structured section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def clean_text(text: str) -> str:
    """Removes non-printable characters and normalizes whitespace."""
    if not isinstance(text, str):
        return ""
    # Replace non-breaking spaces and tabs
    text = text.replace("\u00a0", " ").replace("\t", " ")
    # Collapse multiple spaces into one
    text = " ".join(text.split())
    return text.strip()


def clean_sql_query(query: str) -> str:
    """Normalizes whitespace and the trailing semicolon of a SQL query.

    NOTE: parentheses/keywords are intentionally NOT rewritten. The old version
    turned `count(*)` into `count (*)` (and also edited text inside string
    literals), which taught the model an unnatural style and caused spurious
    exact-match failures.
    """
    if not isinstance(query, str):
        return ""
    q = clean_text(query)
    q = q.rstrip(";").strip()
    if q:
        q += ";"
    return q


def is_valid_sql(query: str) -> bool:
    """Basic validation to ensure query is a valid read-only SQL statement."""
    if not query or len(query) < 8:
        return False
    q_upper = query.strip().upper()
    valid_starts = ("SELECT", "WITH")
    return any(q_upper.startswith(prefix) for prefix in valid_starts)


def preprocess_records(raw_records: list[dict], split_name: str) -> tuple[list[dict], dict]:
    """Processes, cleans, validates, and deduplicates a list of raw records."""
    cleaned_records = []
    seen_signatures = set()

    stats = {
        "split": split_name,
        "raw_count": len(raw_records),
        "valid_count": 0,
        "dropped_missing_fields": 0,
        "dropped_invalid_sql": 0,
        "dropped_duplicates": 0,
        "unique_databases_retained": 0
    }

    preserved_dbs = set()

    for idx, item in enumerate(raw_records):
        # 1. Field extraction (handling possible key variations)
        db_id = clean_text(item.get("db_id", ""))
        question = clean_text(item.get("question", ""))
        query = clean_sql_query(item.get("query") or item.get("sql") or "")
        db_schema = clean_text(item.get("db_schema") or item.get("schema") or "")
        question_toks = item.get("question_toks", [])

        # 2. Check for missing required fields
        # db_schema is required downstream (training prompt + evaluation prompt)
        if not db_id or not question or not query or not db_schema:
            stats["dropped_missing_fields"] += 1
            continue

        # 3. Check for valid SQL structure
        if not is_valid_sql(query):
            stats["dropped_invalid_sql"] += 1
            continue

        # 4. Check for duplicates: unique signature of (db_id, normalized_question, normalized_query)
        signature = (db_id.lower(), question.lower(), query.lower())
        if signature in seen_signatures:
            stats["dropped_duplicates"] += 1
            continue
        seen_signatures.add(signature)

        # 5. Build standardized record
        standardized_item = {
            "id": f"{split_name}_{len(cleaned_records):05d}",
            "db_id": db_id,
            "question": question,
            "query": query,
            "db_schema": db_schema,
            "question_toks": question_toks
        }

        cleaned_records.append(standardized_item)
        preserved_dbs.add(db_id)

    stats["valid_count"] = len(cleaned_records)
    stats["unique_databases_retained"] = len(preserved_dbs)

    return cleaned_records, stats


def run_preprocessing():
    """Main pipeline execution for dataset preprocessing."""
    print_banner("STEP 2: PREPROCESSING, CLEANING & VALIDATING SPIDER DATASET")

    if not RAW_DATA_DIR.exists():
        print(f"[!] ERROR: Raw data directory not found at: {RAW_DATA_DIR}")
        print("    Please run: python load_dataset.py first!")
        sys.exit(1)

    raw_files = sorted(list(RAW_DATA_DIR.glob("raw_*.json")))
    if not raw_files:
        print(f"[!] ERROR: No raw JSON files found in: {RAW_DATA_DIR}")
        print("    Please run: python load_dataset.py first!")
        sys.exit(1)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_raw = 0
    total_cleaned = 0
    total_duplicates = 0
    total_invalid = 0
    all_retained_dbs = set()
    split_stats_list = []

    for raw_path in raw_files:
        split_name = raw_path.stem.replace("raw_", "")
        print(f"\n[*] Processing split: '{split_name.upper()}' from {raw_path.name}...")

        with open(raw_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        cleaned_data, stats = preprocess_records(raw_data, split_name)
        split_stats_list.append(stats)

        # Write processed file
        out_path = PROCESSED_DATA_DIR / f"processed_{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        # Aggregate metrics
        total_raw += stats["raw_count"]
        total_cleaned += stats["valid_count"]
        total_duplicates += stats["dropped_duplicates"]
        total_invalid += (stats["dropped_missing_fields"] + stats["dropped_invalid_sql"])
        
        # Collect databases
        split_dbs = set(item["db_id"] for item in cleaned_data)
        all_retained_dbs.update(split_dbs)

        print(f"    - Raw Examples Evaluated : {stats['raw_count']:,}")
        print(f"    - Duplicates Dropped     : {stats['dropped_duplicates']:,}")
        print(f"    - Invalid Records Dropped: {stats['dropped_missing_fields'] + stats['dropped_invalid_sql']:,}")
        print(f"    - Clean Examples Retained: {stats['valid_count']:,}")
        print(f"    - Unique Databases       : {len(split_dbs):,}")
        print(f"    [OK] Saved to: {out_path.name}")

    # Save overall preprocessing statistics
    stats_meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_raw_records": total_raw,
        "total_cleaned_records": total_cleaned,
        "total_duplicates_dropped": total_duplicates,
        "total_invalid_dropped": total_invalid,
        "total_unique_databases": len(all_retained_dbs),
        "split_breakdown": split_stats_list
    }
    stats_path = PROCESSED_DATA_DIR / "preprocessing_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_meta, f, indent=2)
    print(f"\n[OK] Preprocessing statistics saved to: {stats_path.name}")

    # Video summary table
    print_banner("PREPROCESSING AUDIT REPORT (VIDEO EVIDENCE)")
    print(f"  • Total Input Examples     : {total_raw:,}")
    print(f"  • Clean Standardized Output: {total_cleaned:,} (Retention: {total_cleaned/total_raw*100:.2f}%)")
    print(f"  • Dropped Duplicates       : {total_duplicates:,}")
    print(f"  • Dropped Invalid/Incomplete: {total_invalid:,}")
    print(f"  • Preserved Unique Schemas : {len(all_retained_dbs):,} databases")
    print(f"  • Multi-Schema Integrity   : [VERIFIED] All db_id and db_schema fields preserved.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_preprocessing()