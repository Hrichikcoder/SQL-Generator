"""
sql_evaluator.py
Multi-Metric SQL Evaluation Engine for Spider Benchmark

Implements:
1. Token-aware SQL Normalization:
   - Normalizes keyword casing (UPPERCASE) and whitespace formatting.
   - Strictly preserves string literals (contents, casing, internal punctuation).
   - Preserves identifiers (table and column names).
   - Keeps structurally meaningful differences (e.g. COUNT(*) vs COUNT(DISTINCT col)) as distinct.
2. Legacy Normalization:
   - Preserves original string-regex metric for explicit reference and auditability.
3. Database-backed Execution Accuracy:
   - Executes reference and generated queries on real SQLite database instances.
   - Categorizes and records execution errors (table not found, column not found, syntax error).
   - Compares result sets (order-sensitive when ORDER BY is present, multiset otherwise).
"""

import collections
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Comprehensive SQL keywords in Spider / SQLite dialect
SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "HAVING", "LIMIT",
    "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "CROSS", "ON", "AS",
    "DISTINCT", "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE", "GLOB",
    "BETWEEN", "EXISTS", "UNION", "ALL", "INTERSECT", "EXCEPT",
    "CASE", "WHEN", "THEN", "ELSE", "END", "ASC", "DESC",
    "COUNT", "SUM", "AVG", "MIN", "MAX", "CAST", "SUBSTR", "LENGTH",
    "WITH", "RECURSIVE", "VALUES", "DEFAULT", "COLLATE"
}


def legacy_normalize_for_em(sql: str) -> str:
    """Original legacy normalization logic from earlier baseline evaluation.

    Preserved for direct backward compatibility and reference.
    NOTE: Indiscriminately lowercases string literals and destructively strips
    whitespace around commas and parentheses.
    """
    if not sql:
        return ""
    s = sql.strip().rstrip(";").strip().replace('"', "'")
    s = " ".join(s.split()).lower()
    s = s.replace("<>", "!=")
    s = re.sub(r"\s*([(),])\s*", r"\1", s)
    s = re.sub(r"\s*(!=|<=|>=|=|<|>)\s*", r"\1", s)
    return s


def tokenize_sql(sql: str) -> List[Tuple[str, str]]:
    """Lexes SQL string into a stream of (token_type, token_value) tuples.

    Token types:
      - 'KEYWORD': SQL keywords and clauses (e.g. SELECT, WHERE, COUNT)
      - 'IDENTIFIER': Table and column names, table aliases
      - 'STRING': Single or double-quoted string literals ('Texas', "ohio")
      - 'NUMBER': Integer or float numeric constants (2000, 3.14)
      - 'OPERATOR': Comparison and arithmetic operators (=, !=, <, >, +, -)
      - 'PUNCTUATION': Parentheses, commas, dots
    """
    if not sql:
        return []

    s = sql.strip().rstrip(";").strip()
    tokens = []
    i = 0
    n = len(s)

    while i < n:
        c = s[i]

        # Whitespace
        if c.isspace():
            i += 1
            continue

        # Single-quoted string literal: '...'
        if c == "'":
            j = i + 1
            val = ["'"]
            while j < n:
                if s[j] == "'":
                    val.append("'")
                    if j + 1 < n and s[j + 1] == "'":  # Escaped single quote
                        val.append("'")
                        j += 2
                        continue
                    else:
                        j += 1
                        break
                else:
                    val.append(s[j])
                    j += 1
            tokens.append(('STRING', "".join(val)))
            i = j
            continue

        # Double-quoted string literal / identifier: "..."
        if c == '"':
            j = i + 1
            val = ['"']
            while j < n:
                if s[j] == '"':
                    val.append('"')
                    if j + 1 < n and s[j + 1] == '"':  # Escaped double quote
                        val.append('"')
                        j += 2
                        continue
                    else:
                        j += 1
                        break
                else:
                    val.append(s[j])
                    j += 1
            # In Spider queries, double quotes almost uniformly wrap string literals (e.g. WHERE river_name = "ohio")
            tokens.append(('STRING', "".join(val)))
            i = j
            continue

        # Punctuation & Parentheses
        if c in '(),':
            tokens.append(('PUNCTUATION', c))
            i += 1
            continue

        # Comparison operators
        if s[i:i + 2] in ('!=', '<>', '<=', '>='):
            op = s[i:i + 2]
            if op == '<>':
                op = '!='
            tokens.append(('OPERATOR', op))
            i += 2
            continue

        if c in '=<>+-*/%':
            tokens.append(('OPERATOR', c))
            i += 1
            continue

        # Dot separator (table.column)
        if c == '.':
            tokens.append(('PUNCTUATION', '.'))
            i += 1
            continue

        # Numbers
        if c.isdigit():
            j = i
            while j < n and (s[j].isdigit() or s[j] == '.'):
                j += 1
            tokens.append(('NUMBER', s[i:j]))
            i = j
            continue

        # Identifiers or Keywords
        if c.isalnum() or c == '_':
            j = i
            while j < n and (s[j].isalnum() or s[j] == '_'):
                j += 1
            word = s[i:j]
            if word.upper() in SQL_KEYWORDS:
                tokens.append(('KEYWORD', word.upper()))
            else:
                tokens.append(('IDENTIFIER', word))
            i = j
            continue

        # Any fallback character
        tokens.append(('SYMBOL', c))
        i += 1

    return tokens


def normalize_sql(sql: str, lowercase_identifiers: bool = False) -> str:
    """Robust, token-aware SQL normalization for Spider Text-to-SQL.

    Guarantees:
      1. Keyword casing is uniformly normalized to UPPERCASE (e.g. SELECT, FROM, WHERE).
      2. Whitespace is regularized without compressing or altering code inside string literals.
      3. String literals retain their exact characters, casing, and punctuation.
      4. Function call parentheses are normalized (e.g. COUNT(DISTINCT col) vs COUNT ( DISTINCT col )).
      5. Operators (=, !=, <, >) have standard single-space padding.
      6. If lowercase_identifiers=True, table and column names are compared case-insensitively
         matching SQLite's case-insensitive identifier resolution.
    """
    if not sql:
        return ""

    tokens = tokenize_sql(sql)
    if not tokens:
        return ""

    out = []
    prev_type = None
    prev_val = None

    for idx, (ttype, tval) in enumerate(tokens):
        need_space = True

        if prev_type is None:
            need_space = False
        elif tval == '.':
            need_space = False
        elif prev_val == '.':
            need_space = False
        elif tval == ',':
            need_space = False
        elif tval == '(':
            # Function calls: COUNT(, SUM(, AVG(, MIN(, MAX(
            if prev_type in ('KEYWORD', 'IDENTIFIER') and prev_val.upper() in (
                'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'CAST', 'SUBSTR', 'LENGTH'
            ):
                need_space = False
            elif prev_val == '(':
                need_space = False
        elif prev_val == '(':
            need_space = False
        elif tval == ')':
            need_space = False

        if ttype == 'STRING':
            # Normalize outer quotes to single quotes standard, but PRESERVE contents exactly
            content = tval[1:-1]
            if tval.startswith('"') and tval.endswith('"'):
                content = content.replace('""', '"')
            normalized_val = f"'{content}'"
        elif ttype == 'KEYWORD':
            normalized_val = tval.upper()
        elif ttype == 'IDENTIFIER':
            normalized_val = tval.lower() if lowercase_identifiers else tval
        else:
            normalized_val = tval

        if need_space and out:
            out.append(" ")

        out.append(normalized_val)
        prev_type = ttype
        prev_val = tval

    return "".join(out).strip()


def check_syntax(sql: str) -> Tuple[bool, str]:
    """Validates SQL query syntax using the SQLite parser EXPLAIN command."""
    if not sql:
        return False, "Empty SQL output"
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(f"EXPLAIN {sql}")
        return True, ""
    except sqlite3.OperationalError as e:
        err = str(e)
        if "no such table" in err or "no such column" in err:
            return True, ""
        return False, err
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def execute_sql(db_id: str, sql: str, db_dir: Path) -> Tuple[bool, Optional[List[Tuple[Any, ...]]], str]:
    """Executes a SQL query in read-only mode against the target database SQLite file.

    Returns:
      (success: bool, result_set: list of tuples or None, error_message: str)
    """
    if not sql:
        return False, None, "Empty SQL query"

    db_path = db_dir / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        # Also check direct file if not in subdirectory
        alt_path = db_dir / f"{db_id}.sqlite"
        if alt_path.exists():
            db_path = alt_path
        else:
            return False, None, f"Database file not found: {db_path}"

    uri_path = f"file:{db_path.resolve().as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri_path, uri=True, timeout=5.0)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return True, rows, ""
    except sqlite3.OperationalError as e:
        err = str(e)
        return False, None, f"OperationalError: {err}"
    except Exception as e:
        return False, None, f"{type(e).__name__}: {str(e)}"


def compare_result_sets(res_ref: List[Tuple[Any, ...]], res_gen: List[Tuple[Any, ...]], is_ordered: bool = False) -> bool:
    """Compares execution result sets between reference and generated SQL queries.

    - If is_ordered=True (query contains ORDER BY), order must match strictly.
    - Otherwise, compares as multisets (order-insensitive, accounting for duplicates).
    """
    if res_ref is None or res_gen is None:
        return False

    if is_ordered:
        return res_ref == res_gen

    if len(res_ref) != len(res_gen):
        return False

    try:
        # Multiset comparison
        counter_ref = collections.Counter(res_ref)
        counter_gen = collections.Counter(res_gen)
        return counter_ref == counter_gen
    except TypeError:
        # Fallback if unhashable elements occur
        return res_ref == res_gen


def evaluate_query_pair(
    db_id: str,
    ref_sql: str,
    gen_sql: str,
    db_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Performs full multi-metric evaluation of generated SQL against reference SQL."""
    # 1. Syntax check
    is_syntax_valid, syntax_err = check_syntax(gen_sql)

    # 2. String & Token Exact Match Metrics
    raw_match = (ref_sql.strip() == gen_sql.strip())
    legacy_em = (legacy_normalize_for_em(ref_sql) == legacy_normalize_for_em(gen_sql))
    robust_em = (normalize_sql(ref_sql, lowercase_identifiers=False) == normalize_sql(gen_sql, lowercase_identifiers=False))
    schema_em = (normalize_sql(ref_sql, lowercase_identifiers=True) == normalize_sql(gen_sql, lowercase_identifiers=True))

    # 3. Execution-based accuracy (if database directory available)
    ref_exec_ok = False
    ref_exec_err = ""
    res_ref = None

    gen_exec_ok = False
    gen_exec_err = ""
    res_gen = None

    is_exec_match = False

    if db_dir and db_dir.exists():
        ref_exec_ok, res_ref, ref_exec_err = execute_sql(db_id, ref_sql, db_dir)
        gen_exec_ok, res_gen, gen_exec_err = execute_sql(db_id, gen_sql, db_dir)

        if ref_exec_ok and gen_exec_ok:
            is_ordered = ("ORDER BY" in ref_sql.upper())
            is_exec_match = compare_result_sets(res_ref, res_gen, is_ordered=is_ordered)

    return {
        "syntax_valid": is_syntax_valid,
        "syntax_error": syntax_err,
        "raw_exact_match": raw_match,
        "legacy_exact_match": legacy_em,
        "normalized_exact_match": robust_em,
        "schema_exact_match": schema_em,
        "ref_executed": ref_exec_ok,
        "ref_exec_error": ref_exec_err,
        "gen_executed": gen_exec_ok,
        "gen_exec_error": gen_exec_err,
        "execution_match": is_exec_match,
        "normalized_ref_sql": normalize_sql(ref_sql, lowercase_identifiers=False),
        "normalized_gen_sql": normalize_sql(gen_sql, lowercase_identifiers=False),
    }
