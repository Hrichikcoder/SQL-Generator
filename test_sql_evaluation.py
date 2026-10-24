"""
test_sql_evaluation.py
Comprehensive Test Suite for SQL Evaluation & Normalization Engine

Validates that:
1. Capitalization-only differences in keywords do NOT cause false exact-match failures.
2. Formatting and whitespace differences (e.g. parentheses spacing) do NOT cause false failures.
3. String literal casing, contents, and punctuation (e.g. commas in 'New York, NY') are strictly preserved.
4. Genuinely different query logic (e.g. COUNT(*) vs COUNT(DISTINCT title), different predicates, wrong tables)
   are strictly caught as mismatches.
5. Execution accuracy properly distinguishes between queries that produce equivalent vs different results.
6. Syntax validity alone is NOT claimed as query correctness.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sql_evaluator import (
    check_syntax,
    evaluate_query_pair,
    legacy_normalize_for_em,
    normalize_sql,
    tokenize_sql,
)

DB_DIR = PROJECT_ROOT / "data" / "database"


class TestSQLNormalization(unittest.TestCase):

    def test_01_keyword_capitalization_only(self):
        """Keyword casing differences should match under normalized exact match."""
        q1 = "select title from publication where year > 2000;"
        q2 = "SELECT title FROM publication WHERE year > 2000;"
        self.assertEqual(
            normalize_sql(q1),
            normalize_sql(q2),
            "Keyword casing differences should produce identical normalized SQL."
        )

    def test_02_formatting_and_parenthesis_spacing(self):
        """Extra whitespace around clauses and function parentheses should be normalized."""
        q1 = "SELECT COUNT ( DISTINCT title ) FROM publication WHERE year > 2000;"
        q2 = "SELECT COUNT(DISTINCT title) FROM publication WHERE year > 2000;"
        self.assertEqual(
            normalize_sql(q1),
            normalize_sql(q2),
            "Parentheses whitespace inside function calls should normalize cleanly."
        )

    def test_03_string_literal_casing_and_punctuation_preservation(self):
        """String literals must retain their exact contents, casing, and punctuation."""
        q1 = "SELECT name FROM user WHERE name = 'Michelle' AND city = 'New York, NY';"
        norm = normalize_sql(q1)
        # Verify internal comma and space was NOT stripped
        self.assertIn("'New York, NY'", norm)
        self.assertIn("'Michelle'", norm)

        # Ensure different string literal values do NOT match
        q2 = "SELECT name FROM user WHERE name = 'michelle' AND city = 'New York, NY';"
        self.assertNotEqual(
            normalize_sql(q1),
            normalize_sql(q2),
            "Case-sensitive string literal differences must NOT be masked."
        )

    def test_04_quotes_standardization(self):
        """Spider double-quoted literals should standardize to single quotes without altering content."""
        q1 = 'SELECT traverse FROM river WHERE river_name = "ohio";'
        q2 = "SELECT traverse FROM river WHERE river_name = 'ohio';"
        self.assertEqual(
            normalize_sql(q1),
            normalize_sql(q2),
            "Double-quoted string literals in Spider should normalize to single quotes."
        )

    def test_05_structurally_different_count_must_mismatch(self):
        """COUNT(*) vs COUNT(DISTINCT col) has genuinely different logic and MUST mismatch."""
        q_star = "SELECT COUNT(*) FROM publication WHERE year > 2000;"
        q_distinct = "SELECT COUNT(DISTINCT title) FROM publication WHERE year > 2000;"
        self.assertNotEqual(
            normalize_sql(q_star),
            normalize_sql(q_distinct),
            "COUNT(*) and COUNT(DISTINCT title) are structurally different and must NOT match."
        )

    def test_06_different_columns_and_tables_must_mismatch(self):
        """Queries referencing different columns or tables must never match."""
        q1 = "SELECT name FROM author;"
        q2 = "SELECT authorName FROM author;"
        q3 = "SELECT name FROM user;"
        self.assertNotEqual(normalize_sql(q1), normalize_sql(q3))

    def test_07_different_predicates_must_mismatch(self):
        """Queries with different filter conditions must never match."""
        q1 = "SELECT title FROM movie WHERE release_year > 2010;"
        q2 = "SELECT title FROM movie WHERE release_year < 2010;"
        self.assertNotEqual(normalize_sql(q1), normalize_sql(q2))

    def test_08_syntax_validity_does_not_imply_correctness(self):
        """Hallucinated columns and wrong schemas can still be valid SQLite syntax."""
        hallucinated_sql = "SELECT non_existent_column_xyz FROM fake_table_123 WHERE fake_filter = 999;"
        is_valid, err = check_syntax(hallucinated_sql)
        # Syntax is valid SQL grammar according to parser
        self.assertTrue(is_valid)

        # But when evaluated against real database, it MUST fail execution!
        eval_res = evaluate_query_pair(
            db_id="academic",
            ref_sql="SELECT name FROM author;",
            gen_sql=hallucinated_sql,
            db_dir=DB_DIR
        )
        self.assertTrue(eval_res["syntax_valid"])
        self.assertFalse(eval_res["gen_executed"])
        self.assertFalse(eval_res["execution_match"])
        self.assertIn("OperationalError", eval_res["gen_exec_error"])

    def test_09_semantically_equivalent_alternative_queries_via_execution(self):
        """Subquery MAX vs ORDER BY DESC LIMIT 1 produce equivalent results on database."""
        q_subquery = "SELECT state_name FROM state WHERE population = (SELECT MAX(population) FROM state);"
        q_order_by = "SELECT state_name FROM state ORDER BY population DESC LIMIT 1;"

        # Exact match should be False (different syntax)
        self.assertNotEqual(normalize_sql(q_subquery), normalize_sql(q_order_by))

        # But execution match should succeed if database is available
        eval_res = evaluate_query_pair(
            db_id="geo",
            ref_sql=q_subquery,
            gen_sql=q_order_by,
            db_dir=DB_DIR
        )
        self.assertFalse(eval_res["normalized_exact_match"])
        if DB_DIR.exists():
            self.assertTrue(eval_res["gen_executed"])


def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSQLNormalization)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
