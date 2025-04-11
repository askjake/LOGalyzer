#!/usr/bin/env python3
import os
import json
import sys
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone

########################################################################
# 1) Setup: Database & Model
########################################################################

def create_sqlalchemy_engine() -> Engine:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    credentials_file = os.path.join(base_dir, "credentials.txt")
    try:
        with open(credentials_file, "r") as f:
            creds = json.load(f)
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        sys.exit(1)
    db_url = f"postgresql+psycopg2://{creds['DB_USER']}:{creds['DB_PASSWORD']}@{creds['DB_HOST']}/{creds['DB_NAME']}"
    return create_engine(db_url, echo=False)

def ensure_embeddings_table(engine: Engine, embed_table="stb_logs_embeddings"):
    create_table_query = text(f"""
    CREATE TABLE IF NOT EXISTS {embed_table} (
        source_table TEXT NOT NULL,
        log_id INTEGER,
        timestamp TIMESTAMPTZ NOT NULL,
        timestamp_imputed BOOLEAN DEFAULT FALSE,
        embedding JSONB,
        PRIMARY KEY (source_table, log_id)
    );
    """)
    with engine.begin() as conn:
        conn.execute(create_table_query)
    with engine.begin() as conn:
        conn.execute(text(f"""
            ALTER TABLE {embed_table} ADD COLUMN IF NOT EXISTS timestamp_imputed BOOLEAN DEFAULT FALSE;
        """))

def upsert_embeddings_for_table(engine: Engine, df: pd.DataFrame, embed_table="stb_logs_embeddings"):
    if df.empty:
        return 0
    upsert_q = text(f"""
        INSERT INTO {embed_table} (source_table, log_id, timestamp, timestamp_imputed, embedding)
        VALUES (:src, :logid, :ts, :imputed, :emb)
        ON CONFLICT (source_table, log_id) 
        DO UPDATE SET embedding = EXCLUDED.embedding, 
                      timestamp = EXCLUDED.timestamp,
                      timestamp_imputed = EXCLUDED.timestamp_imputed;
    """)
    count = 0
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(upsert_q, {
                "src": row["source_table"],
                "logid": int(row["id"]),
                "ts": row["timestamp"],
                "imputed": row["timestamp_imputed"],
                "emb": json.dumps(row["embedding"])
            })
            count += 1
    return count

def process_table(table: str, engine: Engine) -> dict:
    """
    Embeds logs for the specified table if new or updated logs exist,
    then upserts them into the 'stb_logs_embeddings' table.
    """
    result = {"table": table, "processed": 0, "updated": False}
    try:
        with engine.connect() as conn:
            source_query = text(f'SELECT MAX("timestamp"), COUNT(*) FROM "{table}";')
            source_res = conn.execute(source_query).fetchone()
            source_max = source_res[0] if source_res and source_res[0] else None
            embed_query = text("""SELECT MAX(timestamp), COUNT(*) FROM stb_logs_embeddings WHERE source_table = :t;""")
            embed_res = conn.execute(embed_query, {"t": table}).fetchone()
            embed_max = embed_res[0] if embed_res else None

        # Only embed new logs if there's a newer timestamp than what's in stb_logs_embeddings
        process_flag = (embed_max is None) or (source_max and source_max > embed_max)
        if not process_flag:
            print(f"[INFO] Table {table} is up-to-date. Skipping.")
            return result

        query = f"""
            SELECT id, "timestamp", message, '{table}' AS source_table
            FROM "{table}"
            ORDER BY "timestamp" ASC
            LIMIT 500000;
        """
        df = pd.read_sql(query, engine)
        if df.empty:
            print(f"[WARN] Table {table} is empty.")
            return result

        # Convert timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce', utc=True)
        df = embed_logs(df)  # see embed_logs function below
        upsert_count = upsert_embeddings_for_table(engine, df)
        print(f"[INFO] Processed table {table}: {upsert_count} embeddings upserted.")
        result["processed"] = upsert_count
        result["updated"] = True
    except Exception as e:
        print(f"[ERROR] Failed processing table {table}: {e}")
    return result

def fetch_all_embeddings(engine: Engine, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Pulls up to 500k embedded logs from the 'stb_logs_embeddings' table.
    Filters by optional start/end date, then sorts by timestamp ascending.
    """
    query = text("""
    SELECT log_id, source_table, timestamp, embedding
    FROM stb_logs_embeddings
    WHERE (:start_date IS NULL OR timestamp >= :start_date)
      AND (:end_date IS NULL OR timestamp <= :end_date)
    ORDER BY timestamp ASC
    LIMIT 500000;
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"start_date": start_date, "end_date": end_date})

    # Convert embedding from JSON only if it's a string; if it's already a list, leave it.
    df["embedding"] = df["embedding"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df

########################################################################
# 2) Embedding & Logging
########################################################################
def create_embedding(text: str, model_instance) -> list:
    return model_instance.encode([text])[0].tolist()

def embed_logs(df: pd.DataFrame, model_instance=None) -> pd.DataFrame:
    """
    Converts 'message' column into a vector embedding. If 'timestamp' is NaT, we replace it
    with the current time (and mark 'timestamp_imputed'=True).
    """
    if df.empty:
        return df
    current_time = datetime.now(timezone.utc)

    def fix_timestamp(x):
        # If x is NaT or a string "NaT", replace with current_time
        if pd.isna(x) or (isinstance(x, str) and x.strip().lower() == "nat"):
            return current_time
        return x

    df["timestamp"] = df["timestamp"].apply(fix_timestamp)
    df["timestamp_imputed"] = df["timestamp"].apply(lambda x: x == current_time)

    if model_instance is None:
        model_instance = SentenceTransformer("all-MiniLM-L6-v2")

    df["embedding"] = df["message"].apply(lambda msg: create_embedding(msg or "", model_instance))
    return df

########################################################################
# 3) New "Landscape" Plot
########################################################################
def plot_embedding_landscape(df: pd.DataFrame):
    """
    Creates a 3D "landscape" where:
      X-axis = time,
      Y-axis = numeric-coded source_table (or RXID),
      Z-axis = embedding norm (the magnitude of the embedding).

    The idea is that 'normal' logs form a relatively flat area, while
    unusual logs (with distinct embeddings) form "peaks" or "valleys."
    """
    if df.empty:
        print("[WARN] No embeddings to plot for the landscape.")
        return

    # 1) Convert timestamp to numeric
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "embedding"])

    # 2) Compute embedding norm
    df["embedding_norm"] = df["embedding"].apply(lambda e: float(np.linalg.norm(e)))

    # 3) Convert source_table to a numeric code so we can plot along a "Y-axis"
    #    We'll keep the original text in 'source_table' for hover info.
    table_codes = {tbl: i for i, tbl in enumerate(sorted(df["source_table"].unique()))}
    df["table_code"] = df["source_table"].map(table_codes)

    # 4) Convert the actual timestamp to numeric (UNIX timestamp in seconds) or keep it as is for 3D
    df["timestamp_sec"] = df["timestamp"].apply(lambda t: t.timestamp())

    print(f"[INFO] Plotting {len(df)} logs in a 3D 'landscape' ...")

    fig = px.scatter_3d(
        df,
        x="timestamp_sec",
        y="table_code",
        z="embedding_norm",
        color="source_table",  # color by source_table so we can see each table distinctly
        hover_data=["log_id", "timestamp", "source_table"],
        title="3D Landscape of Embeddings (Time vs SourceTable vs Embedding Norm)"
    )

    fig.update_traces(marker=dict(size=3))  # Make points smaller so they don't overlap
    fig.update_layout(
        scene=dict(
            xaxis_title="Time (Unix Seconds)",
            yaxis_title="Source Table (numeric code)",
            zaxis_title="Embedding Norm",
        ),
        legend_title="Source Table",
        template="plotly_dark"
    )
    fig.show()

########################################################################
# 4) Main
########################################################################
def main():
    parser = argparse.ArgumentParser(description="Generate a 3D 'landscape' plot for STB log embeddings.")
    parser.add_argument("-n", "--num_tables", type=int, default=5, help="Number of tables to process simultaneously.")
    parser.add_argument("-s", "--start_date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("-e", "--end_date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("-m", "--model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence Transformer model to use for embedding")
    parser.add_argument("--no_embed", action="store_true",
                        help="If set, bypass new embedding generation and only plot existing embeddings.")
    args = parser.parse_args()

    engine = create_sqlalchemy_engine()
    if not engine:
        print("[ERR] No engine created. Exiting.")
        return

    ensure_embeddings_table(engine)

    if not args.no_embed:
        # 1) Collect STB log tables
        with engine.connect() as conn:
            table_query = text("""
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname = 'public'
                  AND tablename ~ '^R[0-9]{10}$';
            """)
            t_result = conn.execute(table_query)
            tables = [row[0] for row in t_result.fetchall()]

        if not tables:
            print("[WARN] No STB log tables found.")
            return

        new_tables = []
        update_tables = []
        with engine.connect() as conn:
            for t in tables:
                embed_q = text("SELECT COUNT(*) FROM stb_logs_embeddings WHERE source_table = :t")
                try:
                    e_res = conn.execute(embed_q, {"t": t}).fetchone()
                    if e_res[0] == 0:
                        new_tables.append(t)
                    else:
                        update_tables.append(t)
                except Exception as e:
                    print(f"[WARN] Could not check embeddings for {t}: {e}")

        prioritized_tables = new_tables + update_tables
        print(f"[INFO] Processing {len(prioritized_tables)} tables "
              f"({len(new_tables)} new, {len(update_tables)} to update)")

        model = SentenceTransformer(args.model)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = []
        with ThreadPoolExecutor(max_workers=args.num_tables) as executor:
            future_to_table = {executor.submit(process_table, t, engine): t for t in prioritized_tables}
            for future in as_completed(future_to_table):
                table = future_to_table[future]
                try:
                    rdict = future.result()
                    results.append(rdict)
                except Exception as exc:
                    print(f"[ERROR] Table {table} generated an exception: {exc}")

        for r in results:
            print(f"[SUMMARY] Table {r['table']} => processed: {r['processed']}, updated={r['updated']}")
    else:
        print("[INFO] Skipping embedding step; using existing embeddings only.")

    # 2) Fetch embeddings from DB
    embed_df = fetch_all_embeddings(engine, args.start_date, args.end_date)
    if embed_df.empty:
        print("[INFO] No embeddings found to plot.")
        return

    # 3) Plot the new 3D "landscape"
    plot_embedding_landscape(embed_df)


if __name__ == "__main__":
    main()
