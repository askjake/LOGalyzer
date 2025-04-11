#!/usr/bin/env python3
"""
numeric_groomer.py

Script to 'groom' or 'enrich' R+10 tables so that candidate columns known to hold
numeric data are stored as numeric types. For each candidate column:
  - If the column exists as text/character varying, convert it to numeric and
    overwrite invalid values with a default.
  - If the column exists as numeric, update any NULL values to the default.
  - If the column does not exist, add it with a default value.

Usage:
  python numeric_groomer.py --db main
  python numeric_groomer.py --db happy --table_name R1955706171

By default, it scans all R+10 tables in the chosen DB. If --table_name is specified,
only that one table is groomed.
"""

import argparse
import sys
import json
import psycopg2
import psycopg2.sql as sql
import os

DEFAULT_NUMERIC_VALUE = 0.0  # Default value for invalid or NULL entries


def get_all_r_tables(conn):
    query = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = 'public'
      AND tablename ~ '^R[0-9]{10}'
    ORDER BY tablename
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return [r[0] for r in rows]


def get_column_datatype(conn, table_name, column_name):
    """
    Returns the PostgreSQL data_type for the given column (case-insensitive) or None if not found.
    """
    q = """
    SELECT data_type
    FROM information_schema.columns
    WHERE table_name = %s AND lower(column_name) = lower(%s)
    """
    with conn.cursor() as cur:
        cur.execute(q, (table_name, column_name))
        row = cur.fetchone()
    return row[0] if row else None


def ensure_column_numeric(conn, table_name, column_name, numeric_type="DOUBLE PRECISION",
                          default=DEFAULT_NUMERIC_VALUE):
    dtype = get_column_datatype(conn, table_name, column_name)
    if dtype:
        dtype = dtype.lower()
        if dtype in ("text", "character varying"):
            print(f"[{table_name}] Converting existing column '{column_name}' from {dtype} to {numeric_type}...")
            convert_column_to_numeric(conn, table_name, column_name, numeric_type)
        elif "int" in dtype or "real" in dtype or "double" in dtype or "numeric" in dtype:
            print(f"[{table_name}] Updating NULL values in numeric column '{column_name}' to {default}...")
            with conn.cursor() as cur:
                update_sql = sql.SQL("UPDATE {} SET {} = %s WHERE {} IS NULL").format(
                    sql.Identifier(table_name),
                    sql.Identifier(column_name),
                    sql.Identifier(column_name)
                )
                cur.execute(update_sql, (default,))
            conn.commit()
        else:
            print(f"[{table_name}] Column '{column_name}' exists with type {dtype}; no action taken.")
    else:
        print(
            f"[{table_name}] Column '{column_name}' does not exist. Adding as {numeric_type} with default {default}...")
        with conn.cursor() as cur:
            add_sql = sql.SQL("ALTER TABLE {} ADD COLUMN {} {} DEFAULT %s").format(
                sql.Identifier(table_name),
                sql.Identifier(column_name),
                sql.SQL(numeric_type)
            )
            cur.execute(add_sql, (default,))
        conn.commit()


def convert_column_to_numeric(conn, table_name, column_name, numeric_type="DOUBLE PRECISION"):
    new_col = column_name + "_new"
    with conn.cursor() as cur:
        # 1) Add new column with default value.
        add_sql = sql.SQL("ALTER TABLE {} ADD COLUMN {} {} DEFAULT %s").format(
            sql.Identifier(table_name),
            sql.Identifier(new_col),
            sql.SQL(numeric_type)
        )
        cur.execute(add_sql, (DEFAULT_NUMERIC_VALUE,))

        # 2) Copy data: if the value matches a numeric pattern, cast it; else use default.
        update_sql = sql.SQL("""
            UPDATE {} 
            SET {} = CASE
                WHEN {} ~ '^[+-]?[0-9]+(\\.[0-9]+)?$'
                THEN {}::float
                ELSE %s
            END
        """).format(
            sql.Identifier(table_name),
            sql.Identifier(new_col),
            sql.Identifier(column_name),
            sql.Identifier(column_name)
        )
        cur.execute(update_sql, (DEFAULT_NUMERIC_VALUE,))

        # 3) Drop the old column.
        drop_sql = sql.SQL("ALTER TABLE {} DROP COLUMN {}").format(
            sql.Identifier(table_name),
            sql.Identifier(column_name)
        )
        cur.execute(drop_sql)

        # 4) Rename new column to the original column name.
        rename_sql = sql.SQL("ALTER TABLE {} RENAME COLUMN {} TO {}").format(
            sql.Identifier(table_name),
            sql.Identifier(new_col),
            sql.Identifier(column_name)
        )
        cur.execute(rename_sql)
    conn.commit()
    print(
        f"  -> Converted column '{column_name}' to {numeric_type} in table '{table_name}', overwriting invalid values with {DEFAULT_NUMERIC_VALUE}.")


def groom_table(conn, table_name, numeric_candidates):
    for col in numeric_candidates:
        ensure_column_numeric(conn, table_name, col, numeric_type="DOUBLE PRECISION", default=DEFAULT_NUMERIC_VALUE)


def main():
    parser = argparse.ArgumentParser(
        description="Groom R+10 tables so that candidate numeric columns are stored as numeric and invalid/NULL values are overwritten with a default.")
    parser.add_argument("--db", choices=["main", "happy"], default="main",
                        help="Which database to connect to: main or happy. (Default main)")
    parser.add_argument("--table_name", type=str, default=None,
                        help="If set, only groom this table; else all R+10 tables.")
    args = parser.parse_args()

    cred_path = "credentials.txt"
    if not os.path.exists(cred_path):
        print("❌ Missing credentials.txt!")
        sys.exit(1)
    with open(cred_path, "r") as f:
        creds = json.load(f)

    if args.db == "happy":
        db_host = creds["DB_HOST"]
        db_name = creds["HAPPY_PATH_DB"]
        db_user = creds["HAPPY_PATH_USER"]
        db_pass = creds["HAPPY_PATH_PASSWORD"]
        print(f"[INFO] Using HAPPY_PATH DB: {db_name}")
    else:
        db_host = creds["DB_HOST"]
        db_name = creds["DB_NAME"]
        db_user = creds["DB_USER"]
        db_pass = creds["DB_PASSWORD"]
        print(f"[INFO] Using MAIN DB: {db_name}")

    try:
        conn = psycopg2.connect(host=db_host, database=db_name, user=db_user, password=db_pass)
    except Exception as e:
        print(f"❌ Could not connect to {db_name}: {e}")
        sys.exit(1)

    # Updated candidate list; note we include "malloc_occupied" and now "freed_up_bytes"
    known_numeric_candidates = [
        "sgs_duration_ms",
        "significance_score",
        "autoenc_score",
        "autoenc_threshold",
        "freed_up_bytes",
        "malloc_occupied",
        "lstm_score",
        "lstm_threshold",
        "worker_thread_version",
        "sgs_return_code",
        # etc.
    ]

    if args.table_name:
        tables = [args.table_name]
    else:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname = 'public'
                  AND tablename ~ '^R[0-9]{10}'
                ORDER BY tablename
            """)
            rows = cur.fetchall()
            tables = [r[0] for r in rows]

    if not tables:
        print("[WARN] No matching R+10 tables found.")
        conn.close()
        sys.exit(0)

    for tbl in tables:
        print(f"[INFO] Grooming table '{tbl}' ...")
        groom_table(conn, tbl, known_numeric_candidates)

    conn.close()
    print("[DONE] Grooming complete.")


if __name__ == "__main__":
    main()
