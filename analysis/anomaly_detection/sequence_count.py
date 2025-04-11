#!/usr/bin/env python3

import os
import json
import psycopg2
import matplotlib.pyplot as plt
import time
from datetime import datetime

# --- Load credentials ---
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
creds_path = os.path.join(base_dir, "credentials.txt")

with open(creds_path, "r") as f:
    creds = json.load(f)

DB_CONFIG = {
    "host": creds.get("DB_HOST"),
    "dbname": creds.get("HAPPY_PATH_DB"),
    "user": creds.get("HAPPY_PATH_USER"),
    "password": creds.get("HAPPY_PATH_PASSWORD")
}

# --- Create or update tracking table with row counts progressively ---
def update_tracking_table_progressively():
    with psycopg2.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS table_row_counts (
                table_name TEXT PRIMARY KEY,
                row_count BIGINT,
                last_updated TIMESTAMP DEFAULT now()
            );
        """)
        conn.commit()

        cur.execute("""
            SELECT tablename
            FROM pg_catalog.pg_tables
            WHERE schemaname = 'public'
              AND tablename ~ '^R\\d{10}$';
        """)
        tables = cur.fetchall()

        for (tablename,) in tables:
            cur.execute(f'SELECT COUNT(*) FROM public."{tablename}"')
            row_count = cur.fetchone()[0]

            cur.execute("""
                INSERT INTO table_row_counts (table_name, row_count, last_updated)
                VALUES (%s, %s, now())
                ON CONFLICT (table_name)
                DO UPDATE SET row_count = EXCLUDED.row_count, last_updated = now();
            """, (tablename, row_count))
            conn.commit()
            yield tablename, row_count

# --- Plot real-time bar graph ---
def plot_counts_progressively():
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    table_data = {}

    while True:
        for tablename, row_count in update_tracking_table_progressively():
            table_data[tablename] = row_count

            # Only keep top 20 by row count
            sorted_data = sorted(table_data.items(), key=lambda x: x[1], reverse=True)[:20]
            table_names, row_counts = zip(*sorted_data)

            ax.clear()
            ax.barh(table_names, row_counts)
            ax.set_xlabel("Row Count")
            ax.set_title(f"Top 20 Table Row Counts (Updated: {datetime.now():%Y-%m-%d %H:%M:%S})")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.draw()
            plt.pause(1)

        time.sleep(30)

if __name__ == "__main__":
    plot_counts_progressively()
