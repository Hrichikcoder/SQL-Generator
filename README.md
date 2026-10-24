# Multi-Schema Text-to-SQL Dataset Pipeline

This directory contains the dataset preparation module for the Multi-Schema Text-to-SQL project using the **Spider benchmark** hosted on Hugging Face: [`SuperMax991/spider-text2sql`](https://huggingface.co/datasets/SuperMax991/spider-text2sql).

---

## 1. Pipeline Overview

The pipeline consists of three separate, modular Python scripts:

| Script | Purpose | Key Outputs |
| :--- | :--- | :--- |
| **`load_dataset.py`** | Fetches the Spider dataset from Hugging Face, inspects its raw structure, columns, and splits, and saves raw JSON files locally. | `data/raw/raw_*.json`<br>`data/raw/dataset_info.json` |
| **`preprocess_dataset.py`** | Validates required fields, standardizes text/SQL formatting, removes invalid queries and duplicates, and preserves multi-database schemas. | `data/processed/processed_*.json`<br>`data/processed/preprocessing_stats.json` |
| **`split_dataset.py`** | Audits data leakage (question overlap), preserves the original 2-way benchmark split, generates a complete 3-way split (Train / Validation / Test) for fine-tuning, and analyzes cross-database distributions. | `data/splits/train.json`<br>`data/splits/validation.json`<br>`data/splits/test.json`<br>`data/splits/split_stats.json` |

---

## 2. Setup Instructions

Before running the scripts, make sure the required packages are installed in your Python environment:

```powershell
cd D:\GenericSQLGenerator
pip install -r requirements.txt
```

*(Install PyTorch first with the build for your CUDA version; see the comments in `requirements.txt`.)*

---

## 3. How to Execute the Pipeline

Run the scripts in sequential order:

### Step 1: Load and Inspect Raw Dataset
```powershell
python load_dataset.py
```
* **What it does**: Connects to Hugging Face, downloads `SuperMax991/spider-text2sql`, prints dataset dimensions, schema columns (`db_id`, `question`, `query`, `db_schema`), and writes raw files to `data/raw/`.
* **Video Evidence Provided**: Dataset source, overall size, column names, raw JSON record structure.

### Step 2: Clean and Preprocess Examples
```powershell
python preprocess_dataset.py
```
* **What it does**: Cleans whitespace, standardizes whitespace and trailing semicolons (SQL text itself is left untouched), drops malformed or empty queries, removes duplicate `(db_id, question, query)` triples, and guarantees multi-schema integrity.
* **Video Evidence Provided**: Preprocessing steps, number of duplicates removed, invalid queries dropped, and retention rate.

### Step 3: Split Inspection & Zero-Leakage Audit
```powershell
python split_dataset.py
```
* **What it does**: Verifies that zero question leakage exists across splits (`Train ∩ Test == 0`), preserves original benchmark splits, creates the dedicated 3-way partition (`train.json`, `validation.json`, `test.json`), and logs cross-database generalization stats.
* **Video Evidence Provided**: Split distribution table (Train, Validation, Test counts and percentages), proof of zero data leakage, and unique database counts.

---

## 4. Preserved Schema Format

Each processed record maintains a standardized schema structure for fine-tuning LLMs:

```json
{
  "id": "train_00001",
  "db_id": "concert_singer",
  "question": "How many singers do we have?",
  "query": "SELECT count(*) FROM singer;",
  "db_schema": "stadium : stadium_id (number) , location (text) ... | singer : singer_id (number) , name (text) ...",
  "question_toks": ["How", "many", "singers", "do", "we", "have", "?"]
}
```

---

## 5. Fine-Tuning and Evaluation

`prompts.py` is the single prompt definition shared by training and both evaluators. `eval_utils.py` holds the shared sampling, metrics and model loading.

```powershell
python split_dataset.py --force                 # re-split (db-disjoint validation); use --force after re-preprocessing
python baseline_evaluation.py -n 100            # seeded random sample, base model
python train_qlora.py --max-steps 50            # smoke test, then run without --max-steps
python evaluate_finetuned.py -n 100             # same samples, adapter attached, prints comparison
```

Re-run order after these fixes: `preprocess_dataset.py` -> `split_dataset.py --force` -> baseline -> train -> evaluate_finetuned.

Notes:
* Using your own database: convert its schema to the same compact format as `db_schema` (`table : col (type) , col (type) | table2 : ...`).

---

## 6. Interactive Web UI (Streamlit)

Launch the Generic Text-to-SQL web interface:

```powershell
pip install streamlit
python -m streamlit run streamlit_app.py
```
* Allows pasting arbitrary schemas (`CREATE TABLE` DDL or compact linear format)
* Accepts natural-language questions and generates schema-grounded SQL
* Validates SQLite syntax and highlights query results with `.sql` download option
* Displays detected GPU hardware and QLoRA training telemetry

* `INCLUDE_DB_ID` in `prompts.py` controls whether the db_id label is shown to the model (default off). Changing it requires retraining.