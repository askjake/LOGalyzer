#!/usr/bin/env python3
"""
gather_all_interest.py

This script connects to your logs database (where your R+10 tables live) and to your 3090 anomalies database.
It then iterates over each R+10 table (or a single table if --table_name is provided) and selects rows
that are “of interest” – that is, rows that satisfy one or more anomaly indicator conditions.
The conditions now include:
  - Boolean flags (autoenc_is_anomaly, lstm_is_anomaly, important_investigate)
  - Numeric scores (if a corresponding threshold column exists then score >= threshold; otherwise score > 0)
  - Text flag customer_marked_flag = 'YES'

Each matching row is converted to JSON (via PostgreSQL’s row_to_json function) and inserted into a consolidated
table (anomalies_consolidated) on the 3090 anomalies database. This table is automatically created if it doesn’t exist,
and its schema remains fixed so that new columns from the source tables are simply added to the JSON output.
Engineers can later use Superset to build charts on these consolidated results.

Usage:
    python gather_all_interest.py [--table_name R1946200481]
"""

import argparse
import json
import os
import sys
import psycopg2
import psycopg2.sql as sql
from psycopg2.extras import DictCursor
import logging

logging.basicConfig(
    level=logging.ERROR,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Anomaly Indicator Definitions ---
# For numeric columns we now compare to the threshold column if available.
# For Boolean and TEXT types the condition remains as before.
# (For numeric columns, we set the condition value to None so that build_filter_condition()
# knows to check for a corresponding threshold column.)
ANOMALY_COLUMNS = [
    ("autoenc_is_anomaly", "BOOLEAN", "= TRUE"),
    ("lstm_is_anomaly", "BOOLEAN", "= TRUE"),
    ("autoenc_score", "DOUBLE PRECISION", None),
    ("lstm_score", "DOUBLE PRECISION", None),
    ("significance_score", "DOUBLE PRECISION", None),
    ("customer_marked_flag", "TEXT", "= 'YES'"),
    ("important_investigate", "BOOLEAN", "= TRUE")
]

# Consolidated anomalies table name on the 3090 anomalies DB
CONSOLIDATED_TABLE = "anomalies_consolidated"

def get_credentials():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    creds_path = os.path.join(base_dir, "credentials.txt")
    if not os.path.exists(creds_path):
        logging.error(f"credentials.txt not found at: {creds_path}")
        sys.exit(1)
    with open(creds_path, "r") as f:
        return json.load(f)

def connect_db(host, dbname, user, password):
    try:
        conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
        conn.autocommit = True
        return conn
    except Exception as e:
        logging.error(f"Could not connect to database {dbname}: {e}")
        sys.exit(1)

def create_consolidated_table(conn):
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {CONSOLIDATED_TABLE} (
        id SERIAL PRIMARY KEY,
        source_table TEXT NOT NULL,
        original_id INTEGER NOT NULL,
        row_data JSONB NOT NULL,
        collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(source_table, original_id)
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_sql)
    logging.info(f"Ensured consolidated table '{CONSOLIDATED_TABLE}' exists.")

def get_all_r_tables(conn, single_table=None):
    with conn.cursor() as cur:
        if single_table:
            cur.execute("""
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname = 'public' AND tablename = %s
            """, (single_table,))
        else:
            cur.execute("""
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname = 'public'
                  AND tablename ~ '^R[0-9]{10}$'
                ORDER BY tablename
            """)
        rows = cur.fetchall()
    return [r[0] for r in rows]

def get_table_columns(conn, table_name):
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
        """, (table_name,))
        rows = cur.fetchall()
    return {row["column_name"]: row["data_type"] for row in rows}

def build_filter_condition(columns_info):
    """
    For each column defined in ANOMALY_COLUMNS that exists in the table, build a condition.
    For numeric columns (autoenc_score, lstm_score, significance_score), if a corresponding threshold column exists,
    the condition becomes: "<col>" >= "<threshold_col>".
    Otherwise, it falls back to the default condition ("> 0") provided.
    For BOOLEAN and TEXT types the condition is used as-is.
    Returns a WHERE clause string (or "FALSE" if no conditions apply).
    """
    conditions = []
    # Mapping of numeric columns to their corresponding threshold columns.
    numeric_thresholds = {
        "autoenc_score": "autoenc_threshold",
        "lstm_score": "lstm_threshold",
        "significance_score": "significance_threshold"
    }
    for col, expected_type, cond in ANOMALY_COLUMNS:
        if col in columns_info:
            # For numeric types, check if we have a corresponding threshold column.
            if expected_type.upper() in ("DOUBLE PRECISION", "NUMERIC", "REAL", "INTEGER", "BIGINT"):
                if col in numeric_thresholds and numeric_thresholds[col] in columns_info:
                    # Build a condition comparing the two columns.
                    condition = f'"{col}" >= "{numeric_thresholds[col]}"'
                    logging.debug("Numeric condition for %s using threshold column %s: %s", col, numeric_thresholds[col], condition)
                else:
                    # Fallback if threshold column is not present.
                    condition = f'"{col}" > 0'
                    logging.debug("Numeric condition for %s (no threshold column found): %s", col, condition)
            elif expected_type.upper() in ("BOOLEAN", "TEXT"):
                condition = f'"{col}" {cond}'
                logging.debug("Condition for %s: %s", col, condition)
            else:
                condition = f'"{col}" {cond}'
            conditions.append(condition)
    if conditions:
        clause = " OR ".join(conditions)
        logging.debug("Built filter clause: %s", clause)
        return clause
    else:
        return "FALSE"

def gather_rows_of_interest(logs_conn, consolidated_conn, single_table=None):
    tables = get_all_r_tables(logs_conn, single_table)
    if not tables:
        logging.warning("No R+10 tables found.")
        return

    for tbl in tables:
        logging.info(f"Processing table: {tbl}")
        cols_info = get_table_columns(logs_conn, tbl)
        where_clause = build_filter_condition(cols_info)
        logging.info(f"Filter for table {tbl}: {where_clause}")
        # Use row_to_json() to convert the entire row.
        query = sql.SQL("SELECT row_to_json(t) AS row_data, t.id AS original_id FROM {tbl} t WHERE {where}")
        query = query.format(
            tbl=sql.Identifier(tbl),
            where=sql.SQL(where_clause)
        )
        with logs_conn.cursor(cursor_factory=DictCursor) as cur:
            try:
                cur.execute(query)
                rows = cur.fetchall()
            except Exception as e:
                logging.error(f"Error selecting rows from {tbl}: {e}")
                logs_conn.rollback()
                continue
        logging.info(f"Found {len(rows)} rows of interest in table {tbl}.")
        print(f"Found {len(rows)} rows of interest in table {tbl}.")
        with consolidated_conn.cursor() as cur:
            for row in rows:
                try:
                    insert_sql = sql.SQL("""
                        INSERT INTO {cons_table} (source_table, original_id, row_data)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (source_table, original_id) DO NOTHING
                    """).format(cons_table=sql.Identifier(CONSOLIDATED_TABLE))
                    cur.execute(insert_sql, (tbl, row["original_id"], json.dumps(row["row_data"])))
                except Exception as e:
                    logging.error(f"Error inserting row from {tbl}: {e}")
                    consolidated_conn.rollback()
        logging.info(f"Inserted rows from table {tbl} into {CONSOLIDATED_TABLE}.")
    logging.info("Completed gathering anomalies.")

def main():
    parser = argparse.ArgumentParser(
        description="Gather rows of interest (including scores compared to thresholds, boolean flags, etc.) "
                    "from R+10 tables and consolidate them into a central anomalies table in the 3090 anomalies DB."
    )
    parser.add_argument("--table_name", type=str, help="If provided, process only this R+10 table.")
    args = parser.parse_args()

    creds = get_credentials()

    # Connect to the logs database (where your R+10 tables are)
    logs_conn = connect_db(creds["DB_HOST"], creds["DB_NAME"], creds["DB_USER"], creds["DB_PASSWORD"])

    # Connect to the 3090 anomalies database (use keys "3090_HOST", "3090_db", "3090_USER", "3090_PASSWORD")
    consolidated_conn = connect_db(creds["3090_HOST"], creds["3090_db"], creds["3090_USER"], creds["3090_PASSWORD"])

    create_consolidated_table(consolidated_conn)
    gather_rows_of_interest(logs_conn, consolidated_conn, single_table=args.table_name)

    logs_conn.close()
    consolidated_conn.close()
    logging.info("Anomalies gathering complete. Consolidated table updated.")

if __name__ == "__main__":
    main()
