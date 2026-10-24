"""
streamlit_app.py
Generic, Schema-Aware Text-to-SQL Web Application

Interactive Streamlit interface for the fine-tuned Qwen2.5-Coder-1.5B QLoRA model.
Generates executable SQLite queries grounded in a database schema that can be entered in
three ways:
  1. Visual Builder  - add tables, edit columns in a grid (type, PK, NOT NULL, foreign key)
  2. SQL / Text      - paste CREATE TABLE DDL or the compact Spider notation
  3. Upload          - a .sql file, a SQLite .db file, or CSV files (one table per CSV)
All three are converted to one schema model and rendered in the selected prompt format.

Launch with:
    python -m streamlit run streamlit_app.py
"""

import html
import io
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = BASE_DIR / "qlora_adapter"
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Generic Text-to-SQL Generator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, modern typography and clean badge styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.3rem;
    }
    .table-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 6px;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .stCodeBlock { border-radius: 8px; }
    .table-pill-cols { font-weight: 400; opacity: 0.75; }
    .schema-info-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #334155;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Predefined Example Schemas (Standard SQL DDL format)
# ---------------------------------------------------------------------------
EXAMPLE_SCHEMAS = {
    "E-Commerce & Orders": {
        "db_id": "ecommerce_store",
        "schema": """-- Table 1: Registered Customers
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    country TEXT,
    signup_date DATE
);

-- Table 2: Catalog Products
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL,
    stock_quantity INTEGER DEFAULT 0
);

-- Table 3: Customer Orders
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date DATE NOT NULL,
    order_status TEXT CHECK(order_status IN ('Pending', 'Shipped', 'Delivered', 'Cancelled')),
    total_amount REAL NOT NULL
);

-- Table 4: Line Items per Order
CREATE TABLE order_items (
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    PRIMARY KEY (order_id, product_id)
);""",
        "sample_question": "Find the top 5 customers who have spent the most money on delivered orders.",
    },
    "University Course Enrollment": {
        "db_id": "university_mgmt",
        "schema": """-- Table 1: Academic Departments
CREATE TABLE departments (
    department_id INTEGER PRIMARY KEY,
    department_name TEXT UNIQUE NOT NULL,
    building TEXT
);

-- Table 2: Faculty Instructors
CREATE TABLE instructors (
    instructor_id INTEGER PRIMARY KEY,
    instructor_name TEXT NOT NULL,
    department_id INTEGER REFERENCES departments(department_id),
    salary REAL NOT NULL
);

-- Table 3: Department Courses
CREATE TABLE courses (
    course_id INTEGER PRIMARY KEY,
    course_title TEXT NOT NULL,
    department_id INTEGER REFERENCES departments(department_id),
    credits INTEGER DEFAULT 3
);

-- Table 4: Student Course Enrollments
CREATE TABLE student_enrollments (
    student_id INTEGER NOT NULL,
    course_id INTEGER REFERENCES courses(course_id),
    grade TEXT,
    semester TEXT NOT NULL,
    PRIMARY KEY (student_id, course_id, semester)
);""",
        "sample_question": "List each department name and the total number of courses offered by that department, ordered by course count descending.",
    },
    "Employee & Project Operations": {
        "db_id": "company_ops",
        "schema": """-- Table 1: Company Departments
CREATE TABLE departments (
    department_id INTEGER PRIMARY KEY,
    department_name TEXT UNIQUE NOT NULL,
    location TEXT NOT NULL
);

-- Table 2: Staff Employees
CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY,
    employee_name TEXT NOT NULL,
    department_id INTEGER REFERENCES departments(department_id),
    manager_id INTEGER REFERENCES employees(employee_id),
    salary REAL NOT NULL,
    hire_date DATE NOT NULL
);

-- Table 3: Active & Planned Projects
CREATE TABLE projects (
    project_id INTEGER PRIMARY KEY,
    project_name TEXT NOT NULL,
    department_id INTEGER REFERENCES departments(department_id),
    budget REAL NOT NULL,
    status TEXT CHECK(status IN ('Active', 'Completed', 'On Hold', 'Planning'))
);

-- Table 4: Project Assignments & Hours
CREATE TABLE employee_projects (
    employee_id INTEGER REFERENCES employees(employee_id),
    project_id INTEGER REFERENCES projects(project_id),
    hours_worked REAL DEFAULT 0.0,
    role TEXT,
    PRIMARY KEY (employee_id, project_id)
);""",
        "sample_question": "Which employees earn more than $120,000 and work on active projects?",
    },
    "Spider Benchmark (Linear Shorthand)": {
        "db_id": "concert_singer",
        "schema": """stadium : stadium_id (number) , location (text) , name (text) , capacity (number) , highest (number) , lowest (number) , average (number) | singer : singer_id (number) , name (text) , country (text) , song_name (text) , song_release_year (text) , age (number) , is_male (others) | concert : concert_id (number) , concert_name (text) , theme (text) , stadium_id (text) , year (text) | singer_in_concert : concert_id (number) , singer_id (others)""",
        "sample_question": "What is the average and maximum age of all singers from France?",
    },
    "Custom / Blank Template": {
        "db_id": "my_database",
        "schema": """-- Write or paste your CREATE TABLE statements here
CREATE TABLE my_table (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    created_at DATE
);""",
        "sample_question": "",
    },
}


# ---------------------------------------------------------------------------
# Schema Model: a schema is a list of tables
#   {"name": str, "columns": [{"name", "type", "pk", "not_null", "ref", "extra"}]}
# The visual builder, the pasted text, uploaded files and presets are all converted
# to / from this one structure, so every input method ends in the same prompt text.
# ---------------------------------------------------------------------------
SQL_TYPES = ["INTEGER", "REAL", "TEXT", "DATE", "BOOLEAN", "OTHER"]

# Spider-style compact type names <-> builder types
TO_COMPACT_TYPE = {"INTEGER": "number", "REAL": "number", "TEXT": "text",
                   "DATE": "time", "BOOLEAN": "boolean", "OTHER": "others"}
FROM_COMPACT_TYPE = {"number": "INTEGER", "text": "TEXT", "time": "DATE",
                     "boolean": "BOOLEAN", "others": "OTHER"}
DDL_TYPE = {"INTEGER": "INTEGER", "REAL": "REAL", "TEXT": "TEXT",
            "DATE": "DATE", "BOOLEAN": "BOOLEAN", "OTHER": "BLOB"}

SCHEMA_FORMATS = {
    "compact": "Compact  (table : col (type) , ... | ...)",
    "ddl": "SQL DDL  (CREATE TABLE ...)",
}
# Format the model should receive. The Spider preset below uses the compact style, which is
# what the training data's `db_schema` field looks like - confirm with:
#   python -c "import json; print(json.load(open('data/splits/train.json'))[0]['db_schema'])"
# and switch this constant to "ddl" if your training data used CREATE TABLE statements.
DEFAULT_SCHEMA_FORMAT = "compact"

COLUMN_COLS = ["Column", "Type", "PK", "Not Null", "References", "Extra"]

_IDENT = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*)'
_QUALIFIED = _IDENT + r"(?:\s*\.\s*" + _IDENT + r")?"
_CREATE_RE = re.compile(
    r"CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(" + _QUALIFIED + r")\s*\(",
    re.IGNORECASE,
)
_CONSTRAINT_KW = {
    "PRIMARY", "NOT", "NULL", "DEFAULT", "REFERENCES", "UNIQUE", "CHECK", "CONSTRAINT",
    "COLLATE", "GENERATED", "AUTOINCREMENT", "AUTO_INCREMENT", "COMMENT", "ON", "AS",
}


def unquote(name: str) -> str:
    name = name.strip()
    if len(name) >= 2 and name[0] in "\"`[" and name[-1] in "\"`]":
        return name[1:-1]
    return name


def clean_ident(value: Any) -> str:
    """Turns arbitrary user text into a safe SQL identifier ('Order Date' -> 'Order_Date')."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\W+", "_", value.strip()).strip("_")


def as_bool(value: Any) -> bool:
    return bool(value) if pd.notna(value) else False


def as_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize_type(raw: str) -> str:
    """Maps any SQL type name (VARCHAR(50), BIGINT, DATETIME, ...) to a builder type."""
    t = (raw or "").upper()
    if not t:
        return "TEXT"
    if re.match(r"^(?:UNSIGNED\s+)?(?:TINY|SMALL|MEDIUM|BIG)?INT|SERIAL", t):
        return "INTEGER"
    if any(k in t for k in ("REAL", "FLOA", "DOUB", "DEC", "NUM", "MONEY")):
        return "REAL"
    if any(k in t for k in ("CHAR", "TEXT", "CLOB", "STRING")):
        return "TEXT"
    if "BOOL" in t:
        return "BOOLEAN"
    if "DATE" in t or "TIME" in t:
        return "DATE"
    return "OTHER"


def sql_ident(name: str) -> str:
    """Quotes an identifier only when needed (spaces, symbols, leading digit)."""
    return name if re.fullmatch(r"[A-Za-z_]\w*", name) else '"' + name.replace('"', '""') + '"'


def format_ref(table: str, column: str = "") -> str:
    return f"{table}.{column}" if column else table


def parse_ref(ref: str) -> Tuple[Optional[str], Optional[str]]:
    """'customers.customer_id' | 'customers(customer_id)' | 'customers' -> (table, column)."""
    m = re.match(r"^\s*(\w+)\s*(?:[.(]\s*(\w+)\s*\)?)?\s*$", ref or "")
    return (m.group(1), m.group(2) or "") if m else (None, None)


# ---------------------------- parsing --------------------------------------
def strip_sql_comments(text: str) -> str:
    text = re.sub(r"/\*[\s\S]*?\*/", " ", text)
    return re.sub(r"--[^\n]*", " ", text)


def split_top_level(body: str, sep: str = ",") -> List[str]:
    """Splits on `sep` only outside parentheses and quotes (so CHECK(x IN ('a', 'b')) stays whole)."""
    parts, buf, depth, quote = [], [], 0, None
    for ch in body:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"`":
            quote = ch
            buf.append(ch)
        elif ch == "[":
            quote = "]"
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def extract_create_tables(text: str) -> List[Tuple[str, str]]:
    """Returns [(table_name, body_between_outer_parentheses)] for every CREATE TABLE."""
    text = strip_sql_comments(text)
    found = []
    for m in _CREATE_RE.finditer(text):
        start, depth, quote, i = m.end(), 1, None, m.end()
        while i < len(text) and depth > 0:
            ch = text[i]
            if quote:
                if ch == quote:
                    quote = None
            elif ch in "'\"`":
                quote = ch
            elif ch == "[":
                quote = "]"
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        if depth == 0:
            found.append((unquote(m.group(1).split(".")[-1]), text[start:i - 1]))
    return found


def parse_ddl_tables(text: str) -> List[Dict[str, Any]]:
    tables = []
    for tbl_name, body in extract_create_tables(text):
        columns, pk_cols, fks = [], set(), {}
        for item in split_top_level(body):
            stripped = re.sub(r"^CONSTRAINT\s+" + _IDENT + r"\s+", "", item, flags=re.IGNORECASE)

            # ---- table-level constraints
            m = re.match(r"PRIMARY\s+KEY\s*\(([^)]*)\)", stripped, re.IGNORECASE)
            if m:
                pk_cols |= {unquote(c).lower() for c in m.group(1).split(",")}
                continue
            m = re.match(
                r"FOREIGN\s+KEY\s*\(([^)]*)\)\s*REFERENCES\s+(" + _QUALIFIED + r")\s*(?:\(([^)]*)\))?",
                stripped, re.IGNORECASE,
            )
            if m:
                from_cols = [unquote(c) for c in m.group(1).split(",")]
                to_cols = [unquote(c) for c in m.group(3).split(",")] if m.group(3) else []
                ref_table = unquote(m.group(2).split(".")[-1])
                for idx, c in enumerate(from_cols):
                    fks[c.lower()] = format_ref(ref_table, to_cols[idx] if idx < len(to_cols) else "")
                continue
            if re.match(r"(?:UNIQUE|CHECK|CONSTRAINT)\b", stripped, re.IGNORECASE) or re.match(
                r"(?:FULLTEXT\s+|UNIQUE\s+)?(?:KEY|INDEX)\s+\S+\s*\(", stripped, re.IGNORECASE
            ):
                continue

            # ---- column definition
            m = re.match(r"\s*(" + _IDENT + r")\s*(.*)$", item, re.DOTALL)
            if not m:
                continue
            col_name, rest = unquote(m.group(1)), m.group(2).strip()

            type_words, remainder = [], rest
            while remainder:
                mt = re.match(r"([A-Za-z_]\w*)\s*(\([^)]*\))?\s*", remainder)
                if not mt or mt.group(1).upper() in _CONSTRAINT_KW:
                    break
                type_words.append(mt.group(1))
                remainder = remainder[mt.end():]

            is_pk = bool(re.search(r"\bPRIMARY\s+KEY\b", remainder, re.IGNORECASE))
            not_null = bool(re.search(r"\bNOT\s+NULL\b", remainder, re.IGNORECASE))
            mref = re.search(
                r"\bREFERENCES\s+(" + _QUALIFIED + r")\s*(?:\(\s*(" + _IDENT + r")\s*\))?", remainder, re.IGNORECASE
            )
            ref = format_ref(unquote(mref.group(1).split(".")[-1]), unquote(mref.group(2) or "")) if mref else ""

            # whatever is left (DEFAULT, UNIQUE, CHECK, AUTOINCREMENT, ...) is kept verbatim as "extra"
            extra = remainder
            extra = re.sub(r"\bPRIMARY\s+KEY\b", " ", extra, flags=re.IGNORECASE)
            extra = re.sub(r"\bNOT\s+NULL\b", " ", extra, flags=re.IGNORECASE)
            extra = re.sub(
                r"\bREFERENCES\s+" + _QUALIFIED + r"\s*(?:\([^)]*\))?(?:\s+ON\s+(?:DELETE|UPDATE)\s+(?:SET\s+\w+|CASCADE|RESTRICT|NO\s+ACTION))*",
                " ", extra, flags=re.IGNORECASE,
            )
            extra = " ".join(extra.split())
            extra = re.sub(r"\bAUTO_INCREMENT\b", "AUTOINCREMENT", extra, flags=re.IGNORECASE)  # SQLite spelling

            columns.append({
                "name": col_name, "type": normalize_type(" ".join(type_words)),
                "pk": is_pk, "not_null": not_null, "ref": ref, "extra": extra,
            })

        for c in columns:
            if c["name"].lower() in pk_cols:
                c["pk"] = True
            if not c["ref"] and c["name"].lower() in fks:
                c["ref"] = fks[c["name"].lower()]
        if columns:
            tables.append({"name": tbl_name, "columns": columns})
    return tables


def parse_compact_tables(text: str) -> List[Dict[str, Any]]:
    """Parses the Spider shorthand:  table : col (type) , col (type) | table2 : ..."""
    tables = []
    for part in text.split("|"):
        if ":" not in part:
            continue
        name, cols_part = part.split(":", 1)
        name = unquote(name.strip())
        columns = []
        for raw in cols_part.split(","):
            raw = raw.strip()
            if not raw:
                continue
            m = re.match(r"^(.*?)\s*\(\s*([^()]*?)\s*\)\s*$", raw)
            col_name, col_type = (m.group(1).strip(), m.group(2).strip().lower()) if m else (raw, "text")
            if col_name:
                columns.append({
                    "name": unquote(col_name), "type": FROM_COMPACT_TYPE.get(col_type, normalize_type(col_type)),
                    "pk": False, "not_null": False, "ref": "", "extra": "",
                })
        if name and columns:
            tables.append({"name": name, "columns": columns})
    return tables


def parse_schema_text(text: str) -> List[Dict[str, Any]]:
    """Auto-detects CREATE TABLE DDL or the compact Spider format."""
    if not text or not text.strip():
        return []
    return parse_ddl_tables(text) or parse_compact_tables(text)


def parse_schema_metadata(schema_text: str) -> List[Dict[str, Any]]:
    """Summary used by the table pills: [{'name', 'columns', 'count'}]."""
    return [
        {"name": t["name"], "columns": [c["name"] for c in t["columns"]], "count": len(t["columns"])}
        for t in parse_schema_text(schema_text)
    ]


# ---------------------------- generation -----------------------------------
def tables_to_ddl(tables: List[Dict[str, Any]]) -> str:
    blocks = []
    for t in tables:
        pk_cols = [sql_ident(c["name"]) for c in t["columns"] if c["pk"]]
        lines = []
        for c in t["columns"]:
            parts = [sql_ident(c["name"]), DDL_TYPE.get(c["type"], "TEXT")]
            if c["pk"] and len(pk_cols) == 1:
                parts.append("PRIMARY KEY")
            if c["not_null"] and not (c["pk"] and len(pk_cols) == 1):
                parts.append("NOT NULL")
            if c["ref"]:
                rt, rc = parse_ref(c["ref"])
                if rt:
                    parts.append(f"REFERENCES {sql_ident(rt)}({sql_ident(rc)})" if rc else f"REFERENCES {sql_ident(rt)}")
            if c.get("extra"):
                parts.append(c["extra"])
            lines.append("    " + " ".join(parts))
        if len(pk_cols) > 1:
            lines.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")
        blocks.append(f"CREATE TABLE {sql_ident(t['name'])} (\n" + ",\n".join(lines) + "\n);")
    return "\n\n".join(blocks)


def tables_to_compact(tables: List[Dict[str, Any]]) -> str:
    return " | ".join(
        f"{t['name']} : " + " , ".join(f"{c['name']} ({TO_COMPACT_TYPE.get(c['type'], 'text')})" for c in t["columns"])
        for t in tables
    )


def tables_to_schema(tables: List[Dict[str, Any]], fmt: str) -> str:
    usable = [t for t in tables if t["name"] and t["columns"]]
    return tables_to_ddl(usable) if fmt == "ddl" else tables_to_compact(usable)


# ---------------------------- validation -----------------------------------
def validate_tables(tables: List[Dict[str, Any]]) -> List[str]:
    """Human-readable problems in a schema (empty list = looks fine)."""
    issues, seen = [], set()
    for t in tables:
        name = t["name"]
        if not name:
            issues.append("A table has no name (it is skipped until you name it).")
            continue
        if name.lower() in seen:
            issues.append(f"Duplicate table name `{name}`.")
        seen.add(name.lower())
        if not t["columns"]:
            issues.append(f"Table `{name}` has no columns (it is skipped until you add one).")
        col_names = [c["name"].lower() for c in t["columns"]]
        for dup in sorted({c for c in col_names if col_names.count(c) > 1}):
            issues.append(f"Table `{name}` has more than one column called `{dup}`.")

    known = {t["name"].lower(): {c["name"].lower() for c in t["columns"]} for t in tables if t["name"]}
    for t in tables:
        for c in t["columns"]:
            if not c["ref"]:
                continue
            rt, rc = parse_ref(c["ref"])
            if not rt:
                issues.append(f"`{t['name']}.{c['name']}`: reference `{c['ref']}` should look like `table.column`.")
            elif rt.lower() not in known:
                issues.append(f"`{t['name']}.{c['name']}` references table `{rt}`, which is not in the schema.")
            elif rc and rc.lower() not in known[rt.lower()]:
                issues.append(f"`{t['name']}.{c['name']}` references `{rt}.{rc}`, but `{rt}` has no such column.")
    return issues


# ---------------------------- file readers ---------------------------------
def read_sqlite_schema(data: bytes) -> List[Dict[str, Any]]:
    """Reads tables, types, keys and foreign keys from an uploaded SQLite database file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        conn = sqlite3.connect(f"file:{Path(tmp.name).as_posix()}?mode=ro", uri=True)
        try:
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            tables = []
            for name in names:
                quoted = '"' + name.replace('"', '""') + '"'
                fks = {r[3].lower(): format_ref(r[2], r[4] or "") for r in conn.execute(f"PRAGMA foreign_key_list({quoted})")}
                columns = [
                    {"name": r[1], "type": normalize_type(r[2]), "pk": r[5] > 0, "not_null": bool(r[3]),
                     "ref": fks.get(r[1].lower(), ""), "extra": ""}
                    for r in conn.execute(f"PRAGMA table_info({quoted})")
                ]
                if columns:
                    tables.append({"name": name, "columns": columns})
            return tables
        finally:
            conn.close()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def read_csv_schema(filename: str, data: bytes) -> List[Dict[str, Any]]:
    """One CSV file = one table; column types are inferred from the first 1,000 rows."""
    df = pd.read_csv(io.BytesIO(data), nrows=1000)
    columns = []
    for col in df.columns:
        series = df[col].dropna()
        if pd.api.types.is_bool_dtype(df[col]):
            col_type = "BOOLEAN"
        elif pd.api.types.is_integer_dtype(df[col]):
            col_type = "INTEGER"
        elif pd.api.types.is_float_dtype(df[col]):
            col_type = "REAL"
        elif len(series) and series.astype(str).head(50).str.match(r"^\d{4}-\d{2}-\d{2}").all():
            col_type = "DATE"
        else:
            col_type = "TEXT"
        columns.append({"name": clean_ident(str(col)) or "column", "type": col_type,
                        "pk": False, "not_null": False, "ref": "", "extra": ""})
    return [{"name": clean_ident(Path(filename).stem).lower() or "table", "columns": columns}]


# ---------------------------------------------------------------------------
# Prompt Formatting & Post-Processing (Mirrors train_qlora.py)
# ---------------------------------------------------------------------------
def build_system_prompt() -> str:
    """Matches the exact training system prompt from train_qlora.py."""
    return (
        "You are a professional, schema-aware Text-to-SQL assistant. "
        "Given the target database identifier and its schema, generate a single "
        "syntactically valid SQLite query to answer the question. "
        "Return only the executable SQL query without explanations or markdown."
    )


def format_user_prompt(db_id: str, db_schema: str, question: str) -> str:
    """Matches the exact training user prompt template from train_qlora.py."""
    return f"""Database ID:
{db_id.strip()}

Database Schema:
{db_schema.strip()}

Question:
{question.strip()}
"""


def clean_sql(raw_output: str) -> str:
    """Extracts clean SQL from model completion, stripping markdown fences."""
    text = raw_output.strip()
    if not text:
        return ""

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


def validate_sql_syntax(sql: str) -> Tuple[bool, str]:
    """Tests SQL syntax validity using an in-memory SQLite EXPLAIN parser."""
    if not sql:
        return False, "Empty SQL query."
    conn = sqlite3.connect(":memory:")
    try:
        # EXPLAIN parses syntax without needing the underlying database tables
        conn.execute(f"EXPLAIN {sql}")
        return True, "Valid SQLite Syntax"
    except sqlite3.OperationalError as e:
        err = str(e)
        if "no such table" in err.lower() or "no such column" in err.lower():
            return True, "Valid SQLite Syntax (Grammar passed)"
        return False, f"SQLite Syntax Error: {err}"
    except Exception as e:
        return False, f"Syntax Check Error: {e}"
    finally:
        conn.close()


def check_schema_alignment(sql: str, schema_text: str) -> List[str]:
    """Checks that every table used in FROM/JOIN exists in the supplied schema."""
    warnings = []
    if not sql or not schema_text:
        return warnings

    known = {t["name"].lower() for t in parse_schema_text(schema_text)}
    ctes = {m.lower() for m in re.findall(r"\b([A-Za-z_]\w*)\s+AS\s*\(", sql, re.IGNORECASE)}
    referenced = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)", sql, re.IGNORECASE)

    seen = set()
    for tbl in referenced:
        low = tbl.lower()
        if low in seen or low in ctes or low in ("select", "where", "with", "as"):
            continue
        seen.add(low)
        # Exact table-name match when the schema could be parsed; plain substring search otherwise
        missing = low not in known if known else low not in schema_text.lower()
        if missing:
            warnings.append(f"Table `{tbl}` referenced in query was not found in the supplied schema.")
    return warnings


# ---------------------------------------------------------------------------
# Lazy Cached Model Loading (Loaded once per Streamlit session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_qlora_model(base_model_id: str, adapter_path: str):
    """Loads the base model in 4-bit NF4 and attaches the trained LoRA adapter."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter_dir = Path(adapter_path)
    weights_file = adapter_dir / "adapter_model.safetensors"
    config_file = adapter_dir / "adapter_config.json"

    if not adapter_dir.exists() or not (weights_file.exists() or (adapter_dir / "adapter_model.bin").exists()):
        raise FileNotFoundError(
            f"Trained adapter weights not found at `{adapter_dir.resolve()}`.\n"
            f"Please run `python train_qlora.py` first to produce the fine-tuned adapter!"
        )

    if not config_file.exists():
        raise FileNotFoundError(f"Missing `adapter_config.json` inside `{adapter_dir.resolve()}`.")

    is_cuda = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if is_cuda and torch.cuda.is_bf16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    ) if is_cuda else None

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        quantization_config=bnb_config,
        device_map={"": 0} if is_cuda else "cpu",
        trust_remote_code=True,
    )
    model_dtype = compute_dtype if is_cuda else torch.float32
    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=model_dtype, **load_kwargs)
    except TypeError:  # older transformers name this argument `torch_dtype`
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=model_dtype, **load_kwargs)

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Session State & Callbacks
#   builder_tables      : [{"uid": int, "df": DataFrame}]  (initial grid contents per table)
#   tname_<uid>         : table-name widget state
#   cols_<uid>          : column-grid widget state (st.data_editor)
#   schema_content      : the schema text that is sent to the model (SQL tab text area)
#   _last_synced_schema : last text generated from the builder; if the text differs from it,
#                         the user typed/pasted something by hand
# Callbacks run BEFORE the script reruns, which is the only safe place to overwrite
# the value of a widget (the text area / grids) that already exists on the page.
# ---------------------------------------------------------------------------
def flash(kind: str, message: str) -> None:
    st.session_state.setdefault("_flash", []).append((kind, message))


def render_flash() -> None:
    for kind, message in st.session_state.pop("_flash", []):
        getattr(st, kind)(message)


def _next_uid() -> int:
    st.session_state["_uid"] = st.session_state.get("_uid", 0) + 1
    return st.session_state["_uid"]


def columns_to_df(columns: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame({
        "Column": pd.Series([c["name"] for c in columns], dtype="object"),
        "Type": pd.Series([c["type"] for c in columns], dtype="object"),
        "PK": pd.Series([bool(c["pk"]) for c in columns], dtype=bool),
        "Not Null": pd.Series([bool(c["not_null"]) for c in columns], dtype=bool),
        "References": pd.Series([c.get("ref", "") for c in columns], dtype="object"),
        "Extra": pd.Series([c.get("extra", "") for c in columns], dtype="object"),
    })


def df_to_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    columns = []
    for _, row in df.iterrows():
        name = clean_ident(row.get("Column"))
        if not name:  # blank rows are ignored
            continue
        col_type = row.get("Type")
        columns.append({
            "name": name,
            "type": col_type if col_type in SQL_TYPES else "TEXT",
            "pk": as_bool(row.get("PK")),
            "not_null": as_bool(row.get("Not Null")),
            "ref": as_str(row.get("References")),
            "extra": as_str(row.get("Extra")),
        })
    return columns


def collect_builder_tables(edited: Optional[Dict[int, pd.DataFrame]] = None) -> List[Dict[str, Any]]:
    """Current builder content as schema tables (uses live grid edits when given)."""
    tables = []
    for t in st.session_state.get("builder_tables", []):
        df = (edited or {}).get(t["uid"], t["df"])
        tables.append({"name": clean_ident(st.session_state.get(f"tname_{t['uid']}", "")), "columns": df_to_columns(df)})
    return tables


def builder_schema_from_state() -> str:
    return tables_to_schema(collect_builder_tables(), st.session_state.get("schema_format", DEFAULT_SCHEMA_FORMAT))


def set_builder_tables(tables: List[Dict[str, Any]]) -> None:
    for t in st.session_state.get("builder_tables", []):
        for prefix in ("tname_", "cols_", "rm_"):
            st.session_state.pop(f"{prefix}{t['uid']}", None)
    new_tables = []
    for t in tables:
        uid = _next_uid()
        st.session_state[f"tname_{uid}"] = t["name"]
        new_tables.append({"uid": uid, "df": columns_to_df(t["columns"])})
    st.session_state["builder_tables"] = new_tables
    st.session_state["_last_synced_schema"] = builder_schema_from_state()


def load_tables(tables: List[Dict[str, Any]]) -> None:
    """Puts tables into the builder AND regenerates the schema text in the selected format."""
    set_builder_tables(tables)
    st.session_state["schema_content"] = st.session_state["_last_synced_schema"]


def add_table() -> None:
    existing = {clean_ident(st.session_state.get(f"tname_{t['uid']}", "")).lower() for t in st.session_state["builder_tables"]}
    n = len(existing) + 1
    while f"table_{n}" in existing:
        n += 1
    uid = _next_uid()
    st.session_state[f"tname_{uid}"] = f"table_{n}"
    starter = [
        {"name": "id", "type": "INTEGER", "pk": True, "not_null": False, "ref": "", "extra": ""},
        {"name": "name", "type": "TEXT", "pk": False, "not_null": False, "ref": "", "extra": ""},
    ]
    st.session_state["builder_tables"].append({"uid": uid, "df": columns_to_df(starter)})


def remove_table(uid: int) -> None:
    st.session_state["builder_tables"] = [t for t in st.session_state["builder_tables"] if t["uid"] != uid]
    for prefix in ("tname_", "cols_", "rm_"):
        st.session_state.pop(f"{prefix}{uid}", None)


def clear_schema() -> None:
    load_tables([])


def import_text_to_builder() -> None:
    tables = parse_schema_text(st.session_state.get("schema_content", ""))
    if not tables:
        flash("warning", "No tables found. Use `CREATE TABLE name (...)` or `table : col (type) , ... | table2 : ...`.")
        return
    load_tables(tables)
    flash("success", f"Loaded {len(tables)} table(s) into the builder and normalized the schema text.")


def on_format_change() -> None:
    """Switching the format converts the schema; hand-typed text is parsed first so it isn't lost."""
    text = st.session_state.get("schema_content", "").strip()
    if text and text != st.session_state.get("_last_synced_schema", "").strip():
        tables = parse_schema_text(text)
        if tables:
            load_tables(tables)
            flash("info", "Converted your schema text to the selected format.")
        else:
            flash("warning", "Could not parse the schema text, so it was left unchanged.")
            st.session_state["_last_synced_schema"] = builder_schema_from_state()  # keep the builder from overwriting it


def load_uploaded_files() -> None:
    files = st.session_state.get("schema_uploads") or []
    tables: List[Dict[str, Any]] = []
    for f in files:
        ext = Path(f.name).suffix.lower()
        try:
            if ext in (".sql", ".txt"):
                found = parse_schema_text(f.getvalue().decode("utf-8", errors="replace"))
            elif ext in (".db", ".sqlite", ".sqlite3"):
                found = read_sqlite_schema(f.getvalue())
            elif ext == ".csv":
                found = read_csv_schema(f.name, f.getvalue())
            else:
                found = []
        except Exception as err:
            flash("error", f"Could not read `{f.name}`: {err}")
            continue
        if found:
            tables.extend(found)
        else:
            flash("warning", f"No tables found in `{f.name}`.")
    if tables:
        load_tables(tables)
        flash("success", f"Loaded {len(tables)} table(s) from {len(files)} file(s). Review them in the builder tab.")


def on_preset_change() -> None:
    """Callback triggered whenever user selects a different preset from the dropdown."""
    chosen_key = st.session_state["preset_selector"]
    if chosen_key in EXAMPLE_SCHEMAS:
        load_tables(parse_schema_text(EXAMPLE_SCHEMAS[chosen_key]["schema"]))
        st.session_state["question_content"] = EXAMPLE_SCHEMAS[chosen_key]["sample_question"]
        st.session_state["db_id_content"] = EXAMPLE_SCHEMAS[chosen_key]["db_id"]


default_preset_name = "E-Commerce & Orders"
st.session_state.setdefault("schema_format", DEFAULT_SCHEMA_FORMAT)
st.session_state.setdefault("_last_synced_schema", "")
if "builder_tables" not in st.session_state:
    load_tables(parse_schema_text(EXAMPLE_SCHEMAS[default_preset_name]["schema"]))
if "question_content" not in st.session_state:
    st.session_state["question_content"] = EXAMPLE_SCHEMAS[default_preset_name]["sample_question"]
if "db_id_content" not in st.session_state:
    st.session_state["db_id_content"] = EXAMPLE_SCHEMAS[default_preset_name]["db_id"]


# ---------------------------------------------------------------------------
# Sidebar: Controls & Best Practices (Clean, No Engine Specs)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Generation Controls")
    max_tokens = st.slider("Max New Tokens", min_value=64, max_value=512, value=256, step=32)
    max_context_length = st.slider("Max Context Length (Tokens)", min_value=512, max_value=2048, value=1024, step=128)

    st.divider()
    st.markdown("### 💡 Schema Tips")
    st.markdown(
        """
        * **Use the same schema format the model was trained on.** The builder's *Schema text format*
          switch converts between compact and DDL notation in one click.
        * **Use descriptive names** (e.g., `order_date`, `total_amount`) to disambiguate intent.
        * **Upload instead of typing:** a SQLite `.db` file brings in tables, types, keys and
          foreign keys automatically.
        * **Keep it focused:** very large schemas (over ~1,000 tokens) are longer than the training examples.
        """
    )

    st.divider()
    st.markdown(
        """
        <small style="color: #64748B;">
        <b>Safety Notice:</b> Queries are validated for syntax correctness. Always review generated queries before running on production databases.
        </small>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Content Area
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">Generic Text-to-SQL Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Schema-aware, parameter-efficient SQL generation powered by QLoRA fine-tuning.</div>',
    unsafe_allow_html=True,
)

# Preset Selector & Database Identifier
col_preset, col_db_id = st.columns([3, 2])
with col_preset:
    st.selectbox(
        "Load a Sample Schema Preset",
        options=list(EXAMPLE_SCHEMAS.keys()),
        index=0,
        key="preset_selector",
        on_change=on_preset_change,
        help="Quickly populate the schema and question inputs with verified domain examples.",
    )

with col_db_id:
    st.text_input(
        "Database Identifier (`db_id`)",
        key="db_id_content",
        help="A unique identifier for the database schema (e.g., ecommerce_store, company_ops).",
    )

# Section A: Database Schema Input (three ways in, one schema out)
st.markdown("#### A. Database Schema Definition")
render_flash()

tab_builder, tab_sql, tab_upload = st.tabs(["🧩 Visual Builder", "✍️ SQL / Text Editor", "📂 Upload"])

# ---- Tab 1: Visual Builder ------------------------------------------------
# NOTE: this block must run before the text area below is created, because it may update `schema_content`.
with tab_builder:
    st.caption(
        "Add tables and edit their columns in the grids. The schema text is regenerated on every change "
        "(it replaces text typed in the SQL tab; use *Parse into builder* there to keep hand-written SQL)."
    )
    st.radio(
        "Schema text format",
        options=list(SCHEMA_FORMATS.keys()),
        format_func=SCHEMA_FORMATS.get,
        key="schema_format",
        horizontal=True,
        on_change=on_format_change,
        help="The text format sent to the model. Pick the one used in your training data.",
    )

    edited_grids: Dict[int, pd.DataFrame] = {}
    for tbl in list(st.session_state["builder_tables"]):
        uid = tbl["uid"]
        with st.container(border=True):
            name_col, remove_col = st.columns([5, 1])
            with name_col:
                st.text_input("Table name", key=f"tname_{uid}", placeholder="Table name, e.g. customers",
                              label_visibility="collapsed")
            with remove_col:
                st.button("🗑️ Remove", key=f"rm_{uid}", on_click=remove_table, args=(uid,))
            edited_grids[uid] = st.data_editor(
                tbl["df"],
                key=f"cols_{uid}",
                num_rows="dynamic",
                hide_index=True,
                column_config={
                    "Column": st.column_config.TextColumn("Column", help="Column name (spaces become underscores)"),
                    "Type": st.column_config.SelectboxColumn("Type", options=SQL_TYPES, default="TEXT", required=True),
                    "PK": st.column_config.CheckboxColumn("PK", help="Primary key (tick several for a composite key)", default=False),
                    "Not Null": st.column_config.CheckboxColumn("Not Null", default=False),
                    "References": st.column_config.TextColumn(
                        "References (FK)", help="Foreign key target as table.column, e.g. customers.customer_id"),
                    "Extra": st.column_config.TextColumn(
                        "Extra SQL", help="Raw constraints kept as-is in DDL, e.g. DEFAULT 0, UNIQUE, CHECK(status IN ('a','b'))"),
                },
            )

    st.button("➕ Add table", on_click=add_table)

    builder_tables_now = collect_builder_tables(edited_grids)
    new_schema = tables_to_schema(builder_tables_now, st.session_state["schema_format"])
    if new_schema != st.session_state["_last_synced_schema"]:
        st.session_state["schema_content"] = new_schema
        st.session_state["_last_synced_schema"] = new_schema

    builder_issues = validate_tables(builder_tables_now)
    if builder_issues:
        st.warning("**Schema check**\n\n" + "\n".join(f"- {issue}" for issue in builder_issues))

# ---- Tab 2: SQL / Text Editor --------------------------------------------
with tab_sql:
    st.caption(
        "Paste standard `CREATE TABLE` DDL or the compact notation "
        "`table : col (type) , col (type) | table2 : ...`. Edit freely - this exact text is sent to the model."
    )
    schema_input = st.text_area(
        label="Database Schema",
        key="schema_content",
        height=260,
        placeholder="CREATE TABLE employees (\n    employee_id INTEGER PRIMARY KEY,\n    employee_name TEXT NOT NULL,\n    department_id INTEGER,\n    salary REAL\n);",
        label_visibility="collapsed",
    )
    btn_parse, btn_clear, _ = st.columns([1.6, 1.2, 3.2])
    with btn_parse:
        st.button("🔄 Parse into builder", on_click=import_text_to_builder,
                  help="Reads the text above into the visual builder and rewrites it in the selected format")
    with btn_clear:
        st.button("🧹 Clear all", on_click=clear_schema, help="Remove every table and clear the schema text")

# ---- Tab 3: Upload --------------------------------------------------------
with tab_upload:
    st.caption(
        "Bring in an existing schema: a **.sql / .txt** file with `CREATE TABLE` statements, a **SQLite .db** file "
        "(tables, types, keys and foreign keys are read automatically), or one or more **.csv** files "
        "(each file becomes a table, column types are inferred). Only the structure is used - no row data is stored or sent."
    )
    uploads = st.file_uploader(
        "Schema files",
        type=["sql", "txt", "db", "sqlite", "sqlite3", "csv"],
        accept_multiple_files=True,
        key="schema_uploads",
        label_visibility="collapsed",
    )
    st.button("📥 Load into schema", on_click=load_uploaded_files, disabled=not uploads,
              help="Replaces the current schema with the tables found in the uploaded files")

# Live Schema Inspector: parsed tables preview
parsed_tables = parse_schema_metadata(schema_input)
if parsed_tables:
    st.markdown("**Detected Schema Structure:**")
    pills_html = "".join(
        f'<span class="table-pill" title="{html.escape(", ".join(t["columns"]))}">📦 {html.escape(t["name"])} '
        f'<span class="table-pill-cols">&nbsp;({t["count"]} cols)</span></span>'
        for t in parsed_tables
    )
    st.markdown(pills_html, unsafe_allow_html=True)

    # Problems in hand-typed text (builder problems are already shown inside the builder tab)
    if schema_input.strip() != st.session_state["_last_synced_schema"].strip():
        text_issues = validate_tables(parse_schema_text(schema_input))
        if text_issues:
            st.warning("**Schema check**\n\n" + "\n".join(f"- {issue}" for issue in text_issues))
elif schema_input.strip():
    st.warning(
        "⚠️ No tables detected. Use `CREATE TABLE table_name (...);` or `table : col (type) , ... | table2 : ...` "
        "- or build the schema in the **Visual Builder** tab."
    )

# Section B: Natural Language Question
st.markdown("#### B. Business Question")
question_input = st.text_input(
    label="Natural-Language Question",
    key="question_content",
    placeholder="e.g., Which departments have an average employee salary greater than $90,000?",
    label_visibility="collapsed",
)

# Action Button
st.markdown("")
col_btn, _ = st.columns([1.5, 4.5])
with col_btn:
    generate_btn = st.button("⚡ Generate SQL Query", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Inference & SQL Generation Pipeline
# ---------------------------------------------------------------------------
if generate_btn:
    clean_schema = schema_input.strip()
    clean_question = question_input.strip()
    clean_db_id = st.session_state.get("db_id_content", "").strip() or "generic_database"

    if not clean_schema:
        st.error("⚠️ Please provide a database schema. The model requires schema context to ground its query.")
        st.stop()

    if not clean_question:
        st.error("⚠️ Please enter a business question to convert into SQL.")
        st.stop()

    model = None
    tokenizer = None

    with st.spinner("Initializing fine-tuned QLoRA model..."):
        try:
            model, tokenizer = load_qlora_model(
                base_model_id=DEFAULT_BASE_MODEL_ID,
                adapter_path=str(ADAPTER_DIR),
            )
        except FileNotFoundError as fnf_err:
            st.error(f"❌ Model Loading Failure: {fnf_err}")
            st.stop()
        except torch.cuda.OutOfMemoryError:
            st.error("❌ GPU Out-Of-Memory (OOM): Close other GPU processes to free VRAM.")
            st.stop()
        except Exception as load_err:
            st.error(f"❌ Failed to load model or adapter: {load_err}")
            st.stop()

    # Exact training prompt structure
    system_prompt = build_system_prompt()
    user_prompt = format_user_prompt(clean_db_id, clean_schema, clean_question)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer([prompt_text], return_tensors="pt", add_special_tokens=False).to(model.device)
    input_token_count = inputs.input_ids.shape[1]

    if input_token_count > max_context_length:
        st.warning(
            f"⚠️ Context Length Warning: Your input prompt contains {input_token_count} tokens "
            f"(configured limit: {max_context_length}). The model was trained on prompts of up to 1,024 tokens, "
            f"so accuracy may drop - try removing tables that the question does not need."
        )

    # Inference (Greedy, Deterministic)
    start_time = time.perf_counter()
    with st.spinner("Generating schema-grounded SQL query..."):
        try:
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,          # greedy decoding
                    temperature=None,         # silence "sampling flag" warnings
                    top_p=None,
                    top_k=None,
                    repetition_penalty=1.0,   # Qwen's default 1.1 would penalize repeated SQL tokens
                    pad_token_id=tokenizer.pad_token_id,
                )
            latency = time.perf_counter() - start_time
        except torch.cuda.OutOfMemoryError:
            st.error("❌ GPU OOM during generation. Reduce 'Max New Tokens' in the sidebar.")
            st.stop()
        except Exception as gen_err:
            st.error(f"❌ Generation Error: {gen_err}")
            st.stop()

    # Decode only newly generated tokens
    generated_tokens = generated_ids[0][input_token_count:]
    raw_completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    generated_sql = clean_sql(raw_completion)

    # -----------------------------------------------------------------------
    # Section C: Generated SQL Output
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### C. Generated SQL Query")

    if generated_sql:
        st.code(generated_sql, language="sql")

        col_dl, col_stat = st.columns([1.2, 3.8])
        with col_dl:
            st.download_button(
                label="📥 Download .sql File",
                data=f"-- Generated by Generic Text-to-SQL Assistant\n-- Database ID: {clean_db_id}\n-- Question: {clean_question}\n-- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n{generated_sql}\n",
                file_name="generated_query.sql",
                mime="text/plain",
                use_container_width=True,
            )
        with col_stat:
            st.caption(
                f"Generated **{len(generated_tokens)} tokens** in **{latency:.2f} seconds** "
                f"({len(generated_tokens)/max(latency, 1e-9):.1f} tokens/sec)."
            )
    else:
        st.warning("⚠️ The model produced an empty completion or non-SQL output.")
        if raw_completion:
            with st.expander("Inspect Raw Output"):
                st.text(raw_completion)

    # -----------------------------------------------------------------------
    # Section D: Query Validation Status
    # -----------------------------------------------------------------------
    st.markdown("### D. Query Validation Status")

    if generated_sql:
        col_syntax, col_schema = st.columns(2)

        # 1. Syntax Validation
        is_syntax_valid, syntax_msg = validate_sql_syntax(generated_sql)
        with col_syntax:
            if is_syntax_valid:
                st.success(f"✅ **Syntax Check:** {syntax_msg}")
            else:
                st.error(f"❌ **Syntax Check:** {syntax_msg}")

        # 2. Schema Reference Alignment Heuristic
        schema_warnings = check_schema_alignment(generated_sql, clean_schema)
        with col_schema:
            if not schema_warnings:
                st.info("ℹ️ **Schema Alignment:** All referenced tables appear in the supplied schema.")
            else:
                for w in schema_warnings:
                    st.warning(f"⚠️ **Schema Warning:** {w}")

        st.markdown(
            """
            <div class="schema-info-box">
            <b>Validation vs. Execution Notice:</b><br>
            Syntax validation confirms that the query conforms to SQLite grammatical rules. 
            <i>Execution accuracy</i> requires running against an active database with live data. 
            Never execute unverified queries on production systems without human review.
            </div>
            """,
            unsafe_allow_html=True,
        )