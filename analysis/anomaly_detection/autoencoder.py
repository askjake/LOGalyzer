#!/usr/bin/env python3
"""
analysis/anomaly_detection/autoencoder.py - Dual-DB Global Autoencoder Anomaly Detection

TRAIN mode (--mode train --happy_path):
  - Connects to HAPPY_PATH_DB (or MAIN DB if specified via --db)
  - Scans all R+10 tables to detect a global set of numeric features.
  - If none are found, it calls the numeric groomer to convert candidate text columns.
  - Then, it fetches these columns (filling missing values with 0) from all tables and aggregates them.
  - Scales the aggregated data and trains (or fine-tunes) a single global autoencoder model.
  - Stores the model and its feature pipeline in the chosen DB.

ANALYZE mode (--mode analyze):
  - Connects to MAIN DB for logs and to HAPPY_PATH_DB to load the stored global model & pipeline.
  - For each table in the main DB, it fetches the global set of numeric features.
  - **New:** Before updating anomaly scores, it checks the “data” field and—if any expected numeric fields (e.g. freed_up_bytes, malloc_occupied, etc.) are missing—adds them automatically.
  - Applies the stored scaler, computes reconstruction errors and marks anomalies.
  - Updates each table with anomaly scores/flags.

Usage examples:
  python autoencoder.py --mode train --happy_path --table_name R1946890461
  python autoencoder.py --mode analyze --table_name R1946890461
"""

import os
import sys
import io
import json
import argparse
import datetime
import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
import logging
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.ERROR,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DEFAULT_NUMERIC_VALUE = 0.0

###############################################################################
# Helper: Run Numeric Groomer if No Numeric Columns Found
###############################################################################
def run_numeric_groomer(db_choice):
    """
    Calls numeric_groomer.py (located at ingestion/linking/numeric_groomer.py)
    to convert candidate text columns into numeric types.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    groomer_path = os.path.join(base_dir, "ingestion", "linking", "numeric_groomer.py")
    cmd = ["python", groomer_path, "--db", db_choice]
    logging.info(f"[Autoencoder] Running numeric groomer: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"[Autoencoder] Numeric groomer failed: {e}")
        sys.exit(1)

###############################################################################
# 0) DB Table for Model + Pipeline Storage
###############################################################################
def ensure_enrichdb_table(pg_conn, enrichdb_table="encoded_master"):
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS "{enrichdb_table}" (
      input_dim INT NOT NULL,
      latent_dim INT NOT NULL,
      model_data BYTEA NOT NULL,
      last_updated TIMESTAMP NOT NULL,
      PRIMARY KEY (input_dim, latent_dim)
    );
    """
    with pg_conn.cursor() as cur:
        cur.execute(create_sql)
    pg_conn.commit()

def parse_schema_and_table(table_name: str):
    table_name = table_name.strip().strip('"')
    if '.' in table_name:
        parts = table_name.split('.', 1)
        schema_part = parts[0].strip().strip('"') or 'public'
        table_part = parts[1].strip().strip('"')
    else:
        schema_part = 'public'
        table_part = table_name
    return schema_part, table_part

###############################################################################
# 1) Global Autoencoder Model Definition
###############################################################################
class LogAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

###############################################################################
# 2) Model & Pipeline Persistence (Using pickle)
###############################################################################
def load_model_from_db(pg_conn, input_dim, latent_dim, enrichdb_table="encoded_master", device="cpu"):
    key = (input_dim, latent_dim)
    logging.info(f"[Autoencoder] Loading model with key={key} from '{enrichdb_table}'.")
    select_sql = f"""
    SELECT model_data
    FROM "{enrichdb_table}"
    WHERE input_dim = %s AND latent_dim = %s
    LIMIT 1
    """
    with pg_conn.cursor() as cur:
        try:
            cur.execute(select_sql, key)
            row = cur.fetchone()
            if not row:
                logging.info(f"[Autoencoder] No model found for key={key}.")
                return None, None
            model_bytes = row[0]
        except psycopg2.errors.UndefinedTable:
            pg_conn.rollback()
            logging.info(f"[Autoencoder] Table '{enrichdb_table}' not found. Creating it.")
            ensure_enrichdb_table(pg_conn, enrichdb_table)
            return None, None
        except psycopg2.Error as e:
            pg_conn.rollback()
            logging.error(f"[Autoencoder] Error loading model: {e}")
            return None, None

    try:
        buffer = io.BytesIO(model_bytes)
        loaded_dict = pickle.load(buffer)
        state_dict = loaded_dict["state_dict"]
        pipeline = loaded_dict.get("pipeline", {})
    except Exception as e:
        logging.error(f"[Autoencoder] Error unpickling model: {e}")
        return None, None

    model = LogAutoencoder(input_dim, latent_dim=latent_dim)
    model.load_state_dict(state_dict)
    logging.info(f"[Autoencoder] Model loaded successfully with key={key}.")
    return model, pipeline

def store_model_in_db(pg_conn, model, input_dim, latent_dim, pipeline, enrichdb_table="encoded_master"):
    key = (input_dim, latent_dim)
    logging.info(f"[Autoencoder] Storing model with key={key} in '{enrichdb_table}'.")
    checkpoint = {"state_dict": model.state_dict(), "pipeline": pipeline}
    model_bytes = pickle.dumps(checkpoint)
    now = datetime.datetime.now(datetime.timezone.utc)
    upsert_sql = f"""
    INSERT INTO "{enrichdb_table}" (input_dim, latent_dim, model_data, last_updated)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (input_dim, latent_dim)
    DO UPDATE SET model_data = EXCLUDED.model_data,
                  last_updated = EXCLUDED.last_updated;
    """
    with pg_conn.cursor() as cur:
        try:
            cur.execute(upsert_sql, (input_dim, latent_dim, psycopg2.Binary(model_bytes), now))
            pg_conn.commit()
        except psycopg2.errors.UndefinedTable:
            pg_conn.rollback()
            logging.info(f"[Autoencoder] Table '{enrichdb_table}' not found. Creating it.")
            ensure_enrichdb_table(pg_conn, enrichdb_table)
            cur.execute(upsert_sql, (input_dim, latent_dim, psycopg2.Binary(model_bytes), now))
            pg_conn.commit()
        except psycopg2.Error as e:
            pg_conn.rollback()
            logging.error(f"[Autoencoder] Error storing model: {e}")
    logging.info(f"[Autoencoder] Model stored/updated with key={key} in '{enrichdb_table}'.")

###############################################################################
# 3) Data Handling: Global Numeric Feature Detection & Data Fetching
###############################################################################
def detect_numeric_columns(pg_conn, table_name):
    """
    Returns a list of numeric column names from the table (ignoring common non-numeric columns).
    (Note: Using table_name as given so that case is preserved.)
    """
    sql_query = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = %s
    """
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(sql_query, (table_name,))
        rows = cur.fetchall()
    numeric = []
    skip = {"id", "timestamp", "event_type", "data", "info", "details"}
    for r in rows:
        col = r["column_name"]
        dt = r["data_type"].lower()
        if col.lower() in skip:
            continue
        if "int" in dt or "real" in dt or "double" in dt or "numeric" in dt:
            numeric.append(col)
    return numeric

def detect_global_numeric_columns(pg_conn, tables):
    """
    Returns a sorted list (for consistency) of the union of numeric columns across given tables.
    """
    global_numeric = set()
    for t in tables:
        cols = detect_numeric_columns(pg_conn, t)
        global_numeric.update(cols)
    return sorted(list(global_numeric))

def fetch_global_features(pg_conn, table_name, global_cols, start_time=None, end_time=None):
    if global_cols:
        cols = ", " + ", ".join([f'"{col}"' for col in global_cols])
    else:
        cols = ""
    sql_query = f"""
    SELECT id, timestamp{cols}
    FROM "{table_name}"
    WHERE timestamp IS NOT NULL
    ORDER BY timestamp ASC
    """
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(sql_query)
        rows = cur.fetchall()
    if not rows:
        return np.array([]), []
    row_ids = []
    data_matrix = []
    for r in rows:
        row_ids.append(r["id"])
        row_vec = []
        for col in global_cols:
            try:
                val = float(r[col]) if r[col] is not None else 0.0
            except Exception:
                val = 0.0
            row_vec.append(val)
        data_matrix.append(row_vec)
    data_matrix = np.array(data_matrix, dtype=np.float32)
    return data_matrix, row_ids

###############################################################################
# 4) Training the Global Autoencoder
###############################################################################
def train_autoencoder(model, train_data, epochs=50, batch_size=32, learning_rate=1e-3, table_label="Global"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"[Autoencoder][{table_label}] Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")
    return model

###############################################################################
# 5) Anomaly Detection
###############################################################################
def detect_anomalies(model, data, threshold=None):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32)
        reconstructed = model(x)
        mse = ((x - reconstructed) ** 2).mean(dim=1).numpy()
    if threshold is None:
        threshold = np.mean(mse) + 2 * np.std(mse)
    anomalies = mse > threshold
    return anomalies, mse, threshold

###############################################################################
# 6) Update Table with Autoencoder Anomaly Columns
###############################################################################
def ensure_anomaly_columns(pg_conn, table_name):
    alter_sqls = [
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS autoenc_score DOUBLE PRECISION',
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS autoenc_threshold DOUBLE PRECISION',
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS autoenc_is_anomaly BOOLEAN'
    ]
    with pg_conn.cursor() as cur:
        for sql_cmd in alter_sqls:
            cur.execute(sql_cmd)
    pg_conn.commit()

def store_anomalies_in_table(pg_conn, table_name, row_ids, anomalies, mse_scores, threshold):
    update_sql = f"""
    UPDATE "{table_name}"
    SET autoenc_score = %s,
        autoenc_threshold = %s,
        autoenc_is_anomaly = %s
    WHERE id = %s
    """
    with pg_conn.cursor() as cur:
        for i, rid in enumerate(row_ids):
            cur.execute(update_sql, (float(mse_scores[i]), float(threshold), bool(anomalies[i]), rid))
    pg_conn.commit()

###############################################################################
# 7) Global Model Training & Analysis Functions
###############################################################################
def get_all_user_tables(pg_conn):
    query = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = 'public'
      AND tablename ~ '^R[0-9]{10}$'
    """
    with pg_conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return [r[0] for r in rows]

def train_global_autoencoder(pg_conn, global_cols, seq_len=16, latent_dim=16, epochs=50, batch_size=32):
    tables = get_all_user_tables(pg_conn)
    if not tables:
        logging.error("[TRAIN][Global] No tables found in the target DB.")
        return None, None, None

    all_data = []
    for t in tables:
        data, _ = fetch_global_features(pg_conn, t, global_cols)
        if data.size > 0:
            logging.info(f"[TRAIN][Global] {t}: {data.shape[0]} rows.")
            all_data.append(data)
        else:
            logging.info(f"[TRAIN][Global] {t}: No data.")
    if not all_data:
        logging.error("[TRAIN][Global] No data aggregated.")
        return None, None, None

    combined_data = np.vstack(all_data)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)
    input_dim = scaled_data.shape[1]
    logging.info(f"[TRAIN][Global] Global data shape: {scaled_data.shape}")

    sequences = []
    n = scaled_data.shape[0]
    for i in range(n - seq_len + 1):
        seq = scaled_data[i: i + seq_len]
        sequences.append(seq)
    if not sequences:
        logging.error("[TRAIN][Global] Not enough data to form sequences.")
        return None, None, None

    model, pipeline_info = load_model_from_db(pg_conn, input_dim, latent_dim, enrichdb_table="encoded_master")
    if model is None:
        logging.info(f"[TRAIN][Global] Creating new global autoencoder (input_dim={input_dim}, latent_dim={latent_dim}).")
        model = LogAutoencoder(input_dim, latent_dim=latent_dim)
        pipeline_info = {}
    else:
        logging.info("[TRAIN][Global] Loaded existing global autoencoder. Fine-tuning...")

    model = train_autoencoder(model, np.vstack(sequences), epochs=epochs, batch_size=batch_size, table_label="Global")
    pipeline_info["global_cols"] = global_cols
    pipeline_info["scaler_means"] = scaler.mean_.tolist()
    pipeline_info["scaler_stds"] = scaler.scale_.tolist()
    pipeline_info["seq_len"] = seq_len
    pipeline_info["input_dim"] = input_dim
    return model, scaler, pipeline_info

def load_any_autoenc_by_latent(pg_conn, latent_dim, enrichdb_table):
    sel_sql = f"""
    SELECT model_data, input_dim
    FROM "{enrichdb_table}"
    WHERE latent_dim = %s
    ORDER BY last_updated DESC
    LIMIT 1
    """
    with pg_conn.cursor() as cur:
        try:
            cur.execute(sel_sql, (latent_dim,))
            row = cur.fetchone()
            if not row:
                return None, None
            model_bytes, in_dim = row
        except Exception as e:
            logging.error(f"[Autoencoder] Error loading model by latent_dim: {e}")
            return None, None
    try:
        buffer = io.BytesIO(model_bytes)
        loaded_dict = pickle.load(buffer)
        state_dict = loaded_dict["state_dict"]
        pipeline = loaded_dict.get("pipeline", {})
        model = LogAutoencoder(in_dim, latent_dim=latent_dim)
        model.load_state_dict(state_dict)
        logging.info(f"[Autoencoder] Loaded global model with input_dim={in_dim}, latent_dim={latent_dim}.")
        return model, pipeline
    except Exception as e:
        logging.error(f"[Autoencoder] Unpickling error: {e}")
        return None, None

###############################################################################
# New Helper Functions for Auto-Enrichment of Missing Columns
###############################################################################
def determine_col_type(sample_value):
    """Return "DOUBLE PRECISION" if sample_value can be cast to float; otherwise "TEXT"."""
    try:
        float(sample_value)
        return "DOUBLE PRECISION"
    except (ValueError, TypeError):
        return "TEXT"

def get_table_columns(pg_conn, schema_name, table_name):
    q = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
    """
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(q, (schema_name, table_name))
        rows = cur.fetchall()
    return {row["column_name"]: row["data_type"] for row in rows}

def add_missing_extracted_columns(pg_conn, schema_name, table_name, extracted_keys):
    """
    For each key in extracted_keys, if the column does not exist in the table,
    add it using a type determined from a sample value.
    """
    existing = get_table_columns(pg_conn, schema_name, table_name)
    with pg_conn.cursor() as cur:
        for key, sample_val in extracted_keys.items():
            if key not in existing:
                col_type = determine_col_type(sample_val)
                sql_add = f'ALTER TABLE "{schema_name}"."{table_name}" ADD COLUMN "{key}" {col_type} DEFAULT %s'
                logging.info(f"[GROOM] Adding missing column '{key}' ({col_type}) to {schema_name}.{table_name} with default {DEFAULT_NUMERIC_VALUE}.")
                cur.execute(sql_add, (DEFAULT_NUMERIC_VALUE,))
    pg_conn.commit()

def enrich_table(pg_conn, table_name, trained_model, relevant_cols):
    """
    Ensures that all columns from relevant_cols exist.
    Then, for each row in the table, parses the "data" field and collects any extra keys.
    Any missing extracted columns are added and the row is updated with the parsed values.
    """
    schema_part, table_part = parse_schema_and_table(table_name)
    # Ensure fixed columns exist:
    from ingestion.linking.enricher_bert import add_col_if_missing, predict_event_type, parse_data_fields
    with pg_conn.cursor() as cur:
        for c in relevant_cols:
            add_col_if_missing(cur, schema_part, table_part, c)
        add_col_if_missing(cur, schema_part, table_part, "significance_score", col_type="DOUBLE PRECISION")
    pg_conn.commit()

    # Fetch rows (include the "data" column for extraction)
    col_select = ", ".join(f'"{c}"' for c in relevant_cols if c != "event_type") + ", id, event_type, data"
    q = f'SELECT {col_select} FROM "{schema_part}"."{table_part}"'
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(q)
        rows = cur.fetchall()
    if not rows:
        logging.info(f"Table '{table_name}' has no rows, skipping enrichment.")
        return
    updates_pred = []
    updates_parse = []
    extracted_keys_union = {}
    for r in rows:
        row_id = r["id"]
        current_label = r.get("event_type") or ""
        if not current_label.strip():
            row_dict = {k: r[k] for k in relevant_cols if k in r}
            predicted = predict_event_type(row_dict, trained_model, relevant_cols)
            if predicted != "UNKNOWN":
                updates_pred.append((predicted, row_id))
        if "data" in r:
            extracted = parse_data_fields(r["data"])
            if extracted:
                updates_parse.append((row_id, extracted))
                for key, val in extracted.items():
                    if key not in extracted_keys_union and val is not None:
                        extracted_keys_union[key] = val
    if extracted_keys_union:
        add_missing_extracted_columns(pg_conn, schema_part, table_part, extracted_keys_union)
    with pg_conn.cursor() as cur:
        for row_id, exdict in updates_parse:
            set_parts = []
            vals = []
            for k, v in exdict.items():
                set_parts.append(f'"{k}"=%s')
                vals.append(v)
            if set_parts:
                sql_up = f'UPDATE "{schema_part}"."{table_part}" SET {",".join(set_parts)} WHERE id=%s'
                vals.append(row_id)
                cur.execute(sql_up, vals)
        if updates_pred:
            up_sql = f'UPDATE "{schema_part}"."{table_part}" SET "event_type"=%s WHERE id=%s'
            cur.executemany(up_sql, updates_pred)
    pg_conn.commit()
    logging.info(f"[{table_name}] => {len(updates_pred)} new event_type predictions, {len(updates_parse)} parse updates.")

###############################################################################
# 8) Main CLI: Dual-DB Global Model
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Global Autoencoder-based anomaly detection with dual-DB support.")
    parser.add_argument("--table_name", type=str, help="Analyze only this table. Otherwise, process all tables.")
    parser.add_argument("--enrichdb_table", type=str, default=None,
                        help="DB table to store the autoencoder model and pipeline.")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension for the autoencoder.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length for building sequences.")
    parser.add_argument("--start_time", type=str, default=None, help="Optional start time (ISO format).")
    parser.add_argument("--end_time", type=str, default=None, help="Optional end time (ISO format).")
    parser.add_argument("--plot", action="store_true", help="If set, generate MSE distribution plots.")
    parser.add_argument("--mode", choices=["train", "analyze"], default="analyze",
                        help="Operation mode: train or analyze.")
    parser.add_argument("--db", choices=["main", "happy"], default="happy",
                        help="Which DB to use for training logs. For analyze mode, still loads the model from happy by default.")

    args = parser.parse_args()

    if args.enrichdb_table is None:
        args.enrichdb_table = f"encoded_master{args.latent_dim}"

    cred_path = "credentials.txt"
    if not os.path.exists(cred_path):
        print("❌ Missing credentials.txt!")
        sys.exit(1)
    with open(cred_path, "r") as f:
        creds = json.load(f)

    if args.mode == "train":
        if args.db == "happy":
            db_host = creds["DB_HOST"]
            db_name = creds["HAPPY_PATH_DB"]
            db_user = creds["HAPPY_PATH_USER"]
            db_pass = creds["HAPPY_PATH_PASSWORD"]
            logging.info(f"[TRAIN] Using HAPPY_PATH DB: {db_name}")
        else:
            db_host = creds["DB_HOST"]
            db_name = creds["DB_NAME"]
            db_user = creds["DB_USER"]
            db_pass = creds["DB_PASSWORD"]
            logging.info(f"[TRAIN] Using MAIN DB: {db_name}")

        pg_conn = psycopg2.connect(host=db_host, database=db_name, user=db_user, password=db_pass)
        ensure_enrichdb_table(pg_conn, args.enrichdb_table)

        if args.table_name:
            tables = [args.table_name]
        else:
            tables = get_all_user_tables(pg_conn)

        # Determine numeric columns across tables.
        global_cols = detect_global_numeric_columns(pg_conn, tables)
        if not global_cols:
            logging.warning("[TRAIN] No numeric columns found. Attempting to run numeric groomer...")
            run_numeric_groomer(args.db)
            tables = get_all_user_tables(pg_conn)
            global_cols = detect_global_numeric_columns(pg_conn, tables)
            if not global_cols:
                logging.error("[TRAIN] Still found no numeric columns after grooming. Exiting.")
                pg_conn.close()
                sys.exit(1)

        logging.info(f"[TRAIN] Global numeric columns: {global_cols}")

        all_data = []
        for t in tables:
            data, _ = fetch_global_features(pg_conn, t, global_cols)
            if data.size > 0:
                logging.info(f"[TRAIN] {t}: {data.shape[0]} rows.")
                all_data.append(data)
            else:
                logging.info(f"[TRAIN] {t}: No data or no numeric columns matched.")
        if not all_data:
            logging.error("[TRAIN] No data aggregated from any table, so cannot train.")
            pg_conn.close()
            sys.exit(1)

        combined_data = np.vstack(all_data)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)
        input_dim = scaled_data.shape[1]

        sequences = []
        n = scaled_data.shape[0]
        for i in range(n - args.seq_len + 1):
            seq = scaled_data[i: i + args.seq_len]
            sequences.append(seq)
        if not sequences:
            logging.error("[TRAIN] Not enough data to form sequences.")
            pg_conn.close()
            sys.exit(1)

        model, pipeline_info = load_model_from_db(pg_conn, input_dim, args.latent_dim, enrichdb_table=args.enrichdb_table)
        if model is None:
            logging.info(f"[TRAIN] Creating new global autoencoder (input_dim={input_dim}, latent_dim={args.latent_dim}).")
            model = LogAutoencoder(input_dim, latent_dim=args.latent_dim)
            pipeline_info = {}
        else:
            logging.info("[TRAIN] Loaded existing global autoencoder. Fine-tuning...")

        model = train_autoencoder(model, np.vstack(sequences),
                                  epochs=args.epochs,
                                  batch_size=args.batch_size,
                                  table_label="Global")
        pipeline_info["global_cols"] = global_cols
        pipeline_info["scaler_means"] = scaler.mean_.tolist()
        pipeline_info["scaler_stds"] = scaler.scale_.tolist()
        pipeline_info["seq_len"] = args.seq_len
        pipeline_info["input_dim"] = input_dim

        store_model_in_db(pg_conn, model, input_dim, args.latent_dim, pipeline_info,
                          enrichdb_table=args.enrichdb_table)
        print("[TRAIN] Global autoencoder model stored.")
        pg_conn.close()

    else:
        # ANALYZE mode:
        main_db_host = creds["DB_HOST"]
        main_db_name = creds["DB_NAME"]
        main_db_user = creds["DB_USER"]
        main_db_pass = creds["DB_PASSWORD"]
        logging.info(f"[ANALYZE] Using MAIN DB for logs: {main_db_name}")
        pg_conn_main = psycopg2.connect(host=main_db_host, database=main_db_name, user=main_db_user,
                                        password=main_db_pass)
        happy_db_host = creds["DB_HOST"]
        happy_db_name = creds["HAPPY_PATH_DB"]
        happy_db_user = creds["HAPPY_PATH_USER"]
        happy_db_pass = creds["HAPPY_PATH_PASSWORD"]
        logging.info(f"[ANALYZE] Loading global autoencoder model from HAPPY_PATH DB: {happy_db_name}")
        pg_conn_happy = psycopg2.connect(host=happy_db_host, database=happy_db_name, user=happy_db_user,
                                         password=happy_db_pass)
        model, pipeline_info = load_any_autoenc_by_latent(pg_conn_happy, args.latent_dim, args.enrichdb_table)
        if model is None or pipeline_info is None:
            print("[ANALYZE] No stored global autoencoder model found in HAPPY_PATH DB; aborting.")
            pg_conn_main.close()
            pg_conn_happy.close()
            sys.exit(1)
        global_cols = pipeline_info.get("global_cols", [])
        if not global_cols:
            print("[ANALYZE] No global feature columns stored in pipeline info; aborting.")
            pg_conn_main.close()
            pg_conn_happy.close()
            sys.exit(1)
        means = np.array(pipeline_info.get("scaler_means", []), dtype=np.float32)
        stds = np.array(pipeline_info.get("scaler_stds", []), dtype=np.float32)
        scaler = StandardScaler()
        scaler.mean_ = means
        scaler.scale_ = stds
        scaler.n_features_in_ = len(means)
        scaler.var_ = stds ** 2

        if args.table_name:
            tables = [args.table_name]
        else:
            tables = get_all_user_tables(pg_conn_main)
        for t in tables:
            # First, call enrich_table to ensure any extra parsed numeric columns are present.
            try:
                enrich_table(pg_conn_main, t, model, pipeline_info.get("global_cols", []))
            except psycopg2.Error as e:
                logging.error(f"Error enriching table {t}: {e}")
                pg_conn_main.rollback()
                continue

            data, row_ids = fetch_global_features(pg_conn_main, t, global_cols)
            if data.size == 0:
                print(f"[ANALYZE] No data in table {t}, skipping.")
                continue
            scaled_data = scaler.transform(data)
            sequences = []
            n = scaled_data.shape[0]
            for i in range(n - pipeline_info.get("seq_len", args.seq_len) + 1):
                seq = scaled_data[i: i + pipeline_info.get("seq_len", args.seq_len)]
                sequences.append(seq)
            if not sequences:
                print(f"[ANALYZE] Not enough data in {t} for sequences, skipping.")
                continue
            anomalies, mse_scores, threshold = detect_anomalies(model, np.vstack(sequences))
            logging.info(f"[ANALYZE][{t}] Threshold: {threshold:.4f}, #Anomalies: {np.sum(anomalies)}")
            print(f"[ANALYZE][{t}] Threshold: {threshold:.4f}, #Anomalies: {np.sum(anomalies)}")
            ensure_anomaly_columns(pg_conn_main, t)
            store_anomalies_in_table(pg_conn_main, t, row_ids, anomalies, mse_scores, threshold)
            if args.plot:
                fig_path = f"mse_dist_{t}.png"
                plt.figure(figsize=(8, 6))
                plt.hist(mse_scores, bins=50, alpha=0.7, label="Reconstruction MSE")
                plt.axvline(x=threshold, color='red', linestyle='--', label=f"Threshold={threshold:.4f}")
                plt.title(f"MSE Distribution - {t}")
                plt.xlabel("MSE")
                plt.ylabel("Frequency")
                plt.legend()
                plt.tight_layout()
                plt.savefig(fig_path)
                logging.info(f"[ANALYZE][{t}] MSE plot saved to {fig_path}")
            print(f"[ANALYZE][{t}] => anomalies={np.sum(anomalies)}, threshold={threshold:.4f}")
        pg_conn_main.close()
        pg_conn_happy.close()
        logging.info("[ANALYZE][Global] Analysis complete.")

        # Optionally, run anomaly gathering
        try:
            if args.table_name:
                subprocess.run(["python", "ingestion/linking/importance_collector.py", "--table_name", args.table_name],
                               check=True)
            else:
                subprocess.run(["python", "ingestion/linking/importance_collector.py"], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Gather anomalies error: {e}")

if __name__ == "__main__":
    main()
