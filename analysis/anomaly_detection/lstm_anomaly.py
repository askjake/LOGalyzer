#!/usr/bin/env python3
"""
lstm_anomaly.py - Dual-DB approach for training vs. analysis.
In TRAIN mode (--mode train --happy_path), logs are read from the happy_path DB and the LSTM model is stored there.
In ANALYZE mode (--mode analyze), logs are read from the main DB while the LSTM model is loaded from the happy_path DB.
Known-good logs (in happy_path) are used to train a model that is then applied to analyze and update tables in the main DB.

Usage:
  # Train mode – use happy_path DB for logs and model storage:
  python lstm_anomaly.py --mode train --happy_path --table_name R1946890461

  # Analyze mode – logs come from the main DB but the model is loaded from happy_path DB:
  python lstm_anomaly.py --mode analyze --table_name R1946890461
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

###############################################################################
# 0) Utility: Ensure the LSTM Model Table exists
###############################################################################
def ensure_enrichdb_table(pg_conn, enrichdb_table="lstm_master"):
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS "{enrichdb_table}" (
      model_type TEXT NOT NULL,
      input_dim INT NOT NULL,
      hidden_size INT NOT NULL,
      seq_len INT NOT NULL,
      model_data BYTEA NOT NULL,
      last_updated TIMESTAMP NOT NULL
    );
    """
    with pg_conn.cursor() as cur:
        cur.execute(create_sql)
    # Create a unique index to ensure one model per key
    with pg_conn.cursor() as cur:
        index_name = f"{enrichdb_table}_uniq_idx"
        create_index_sql = f"""
        CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
        ON "{enrichdb_table}" (model_type, input_dim, hidden_size, seq_len);
        """
        cur.execute(create_index_sql)
    pg_conn.commit()

###############################################################################
# 1) LSTM Model Definition
###############################################################################
class LogLSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Use output from the final time step to predict the next feature vector
        out = self.fc(out[:, -1, :])
        return out

###############################################################################
# 2) Model Persistence: Load/Store Model from a specified DB
###############################################################################
def load_lstm_from_db(pg_conn, input_dim, hidden_size, seq_len,
                      enrichdb_table="lstm_master", device="cpu", model_type="LSTM_GLOBAL"):
    key = (model_type, input_dim, hidden_size, seq_len)
    logging.info(f"[LSTM] Attempting to load model with key={key} from table '{enrichdb_table}'.")
    select_sql = f"""
    SELECT model_data
    FROM "{enrichdb_table}"
    WHERE model_type = %s
      AND input_dim = %s
      AND hidden_size = %s
      AND seq_len = %s
    LIMIT 1
    """
    with pg_conn.cursor() as cur:
        try:
            cur.execute(select_sql, key)
        except psycopg2.errors.UndefinedTable:
            pg_conn.rollback()
            logging.info(f"[LSTM] Table '{enrichdb_table}' not found. Creating it.")
            ensure_enrichdb_table(pg_conn, enrichdb_table)
            cur.execute(select_sql, key)
        except psycopg2.Error as e:
            pg_conn.rollback()
            logging.error(f"[LSTM] Error loading model: {e}")
            return None
        row = cur.fetchone()
        if not row:
            logging.info(f"[LSTM] No existing model found in '{enrichdb_table}' for key={key}.")
            return None
        model_bytes = row[0]
    buffer = io.BytesIO(model_bytes)
    checkpoint = torch.load(buffer, map_location=device)
    model = LogLSTMAnomalyDetector(input_dim, hidden_size)
    model.load_state_dict(checkpoint["state_dict"])
    logging.info(f"[LSTM] Model with key={key} loaded successfully.")
    return model

def store_lstm_in_db(pg_conn, model, input_dim, hidden_size, seq_len,
                     enrichdb_table="lstm_master", model_type="LSTM_GLOBAL"):
    key = (model_type, input_dim, hidden_size, seq_len)
    logging.info(f"[LSTM] Storing model with key={key} in '{enrichdb_table}'.")
    checkpoint = {"state_dict": model.state_dict()}
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    model_bytes = buffer.getvalue()
    now = datetime.datetime.now(datetime.timezone.utc)
    upsert_sql = f"""
    INSERT INTO "{enrichdb_table}" (model_type, input_dim, hidden_size, seq_len, model_data, last_updated)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (model_type, input_dim, hidden_size, seq_len)
    DO UPDATE SET model_data = EXCLUDED.model_data,
                  last_updated = EXCLUDED.last_updated;
    """
    with pg_conn.cursor() as cur:
        try:
            cur.execute(upsert_sql, (
                model_type, input_dim, hidden_size, seq_len,
                psycopg2.Binary(model_bytes), now
            ))
            pg_conn.commit()
        except psycopg2.errors.UndefinedTable:
            pg_conn.rollback()
            logging.info(f"[LSTM] Table '{enrichdb_table}' not found. Creating it.")
            ensure_enrichdb_table(pg_conn, enrichdb_table)
            cur.execute(upsert_sql, (
                model_type, input_dim, hidden_size, seq_len,
                psycopg2.Binary(model_bytes), now
            ))
            pg_conn.commit()
        except psycopg2.Error as e:
            pg_conn.rollback()
            logging.error(f"[LSTM] Error storing model: {e}")
    logging.info(f"[LSTM] Model with key={key} stored/updated in '{enrichdb_table}'.")

###############################################################################
# 3) Sequence Dataset & Training Functions
###############################################################################
class LogSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # The final row is the prediction target.
        return seq[:-1], seq[-1]

def train_lstm_anomaly(model, sequences, epochs=10, batch_size=32, learning_rate=1e-3, table_name=None):
    dataset = LogSequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, target in dataloader:
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(dataset)
        if table_name:
            print(f"[LSTM][{table_name}] Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")
        else:
            print(f"[LSTM] Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")
    return model

###############################################################################
# 4) Anomaly Detection Function
###############################################################################
def detect_lstm_anomalies(model, sequences):
    model.eval()
    losses = []
    with torch.no_grad():
        for seq in sequences:
            input_seq = seq[:-1].unsqueeze(0)  # shape: [1, seq_len-1, input_dim]
            target = seq[-1].unsqueeze(0)      # shape: [1, input_dim]
            out = model(input_seq)
            mse_loss = nn.functional.mse_loss(out, target, reduction='mean').item()
            losses.append(mse_loss)
    raw_losses = np.array(losses, dtype=np.float32)
    if raw_losses.size == 0:
        return np.array([], dtype=bool), raw_losses, 0.0, raw_losses
    min_val, max_val = raw_losses.min(), raw_losses.max()
    if max_val > min_val:
        norm_losses = (raw_losses - min_val) / (max_val - min_val)
    else:
        norm_losses = raw_losses.copy()
    threshold = np.mean(norm_losses) + 2.0 * np.std(norm_losses)
    anomalies = norm_losses > threshold
    return anomalies, norm_losses, threshold, raw_losses

###############################################################################
# 5) Build Feature Vectors from Logs
###############################################################################
def get_table_columns(pg_conn, table_name):
    query = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = %s
    """
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query, (table_name.lower(),))
        rows = cur.fetchall()
    return set(r["column_name"].lower() for r in rows)

def build_feature_vectors(pg_conn, table_name, seq_len=16, scale_data=True):
    columns = get_table_columns(pg_conn, table_name)
    select_parts = ["id"]
    # Feature 1: Length of the data field
    if "data" in columns:
        select_parts.append("COALESCE(LENGTH(data), 0) AS data_len")
    else:
        select_parts.append("0 AS data_len")
    # Feature 2: Timestamp (for time difference computation)
    if "timestamp" in columns:
        select_parts.append("timestamp")
    else:
        select_parts.append("NULL as timestamp")
    sql_query = f"""
      SELECT {', '.join(select_parts)}
      FROM "{table_name}"
      WHERE timestamp IS NOT NULL
      ORDER BY timestamp ASC
    """
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(sql_query)
        rows = cur.fetchall()
    if not rows:
        return [], []
    row_ids = [row["id"] for row in rows]
    data_lens = []
    timestamps = []
    for r in rows:
        data_lens.append(float(r["data_len"]))
        try:
            t = datetime.datetime.fromisoformat(r["timestamp"])
        except Exception:
            t = None
        timestamps.append(t)
    time_diffs = [0.0]
    for i in range(1, len(timestamps)):
        if timestamps[i] and timestamps[i-1]:
            diff_s = (timestamps[i] - timestamps[i-1]).total_seconds()
        else:
            diff_s = 0.0
        time_diffs.append(diff_s)
    features = np.column_stack((data_lens, time_diffs))
    if scale_data and len(features) > 1:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    sequences = []
    n = len(features)
    for i in range(n - seq_len + 1):
        block = features[i: i + seq_len]
        block_tensor = torch.tensor(block, dtype=torch.float32)
        sequences.append(block_tensor)
    return row_ids, sequences

###############################################################################
# 6) Update Table with LSTM Anomaly Columns
###############################################################################
def ensure_lstm_columns(pg_conn, table_name):
    alter_cmds = [
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS lstm_score DOUBLE PRECISION',
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS lstm_threshold DOUBLE PRECISION',
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS lstm_is_anomaly BOOLEAN',
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS lstm_issue TEXT'
    ]
    with pg_conn.cursor() as cur:
        for cmd in alter_cmds:
            cur.execute(cmd)
    pg_conn.commit()

def store_lstm_anomalies(pg_conn, table_name, row_ids, anomalies, norm_losses, threshold, seq_len=16):
    update_sql = f"""
    UPDATE "{table_name}"
    SET lstm_score = %s,
        lstm_threshold = %s,
        lstm_is_anomaly = %s,
        lstm_issue = %s
    WHERE id = %s
    """
    if len(anomalies) == 0:
        logging.info(f"[LSTM] No sequences => no anomalies to store in {table_name}.")
        return
    try:
        with pg_conn.cursor() as cur:
            for i, anom in enumerate(anomalies):
                final_idx = i + seq_len - 1
                row_id = row_ids[final_idx]
                score_val = float(norm_losses[i])
                diff = score_val - threshold
                if not anom:
                    issue_str = "normal"
                elif diff > 0.2:
                    issue_str = "critical"
                elif diff > 0.1:
                    issue_str = "warning"
                else:
                    issue_str = "anomaly"
                cur.execute(update_sql, (score_val, float(threshold), bool(anom), issue_str, row_id))
        pg_conn.commit()
    except psycopg2.Error as e:
        logging.error(f"[LSTM] DB error updating anomalies in table {table_name}: {e}")
        pg_conn.rollback()

###############################################################################
# 7) Global LSTM: Train/Analyze Functions
###############################################################################
def get_all_r_tables(pg_conn):
    query = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname='public'
      AND tablename ~ '^R[0-9]{10}$'
    """
    with pg_conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return [r[0] for r in rows]

def train_global_lstm_model(pg_conn, seq_len=16, hidden_size=16, epochs=10,
                            enrichdb_table="lstm_master", model_type="LSTM_GLOBAL"):
    tables = get_all_r_tables(pg_conn)
    if not tables:
        print("[TRAIN][Global] No R-tables found. Aborting.")
        return None
    all_seq = []
    for t in tables:
        row_ids, seqs = build_feature_vectors(pg_conn, t, seq_len=seq_len)
        if seqs:
            all_seq.extend(seqs)
            print(f"[TRAIN][Global] {t}: {len(seqs)} sequences.")
        else:
            print(f"[TRAIN][Global] {t}: No sequences collected.")
    if not all_seq:
        print("[TRAIN][Global] No sequences aggregated.")
        return None
    input_dim = all_seq[0].shape[1]
    model = load_lstm_from_db(pg_conn, input_dim, hidden_size, seq_len,
                              enrichdb_table=enrichdb_table, model_type=model_type)
    if model is None:
        print(f"[TRAIN][Global] Creating new LSTM model (dim={input_dim}, hidden={hidden_size}, seq_len={seq_len}).")
        model = LogLSTMAnomalyDetector(input_dim, hidden_size)
    else:
        print("[TRAIN][Global] Loaded existing global model. Fine-tuning...")
    model = train_lstm_anomaly(model, all_seq, epochs=epochs, table_name="GLOBAL_MODEL")
    return model

def analyze_global_lstm(pg_conn, model, seq_len=16, plot=False,
                        enrichdb_table="lstm_master", hidden_size=16):
    if model is None:
        print("[ANALYZE][Global] No model. Exiting.")
        return
    tables = get_all_r_tables(pg_conn)
    for t in tables:
        row_ids, seqs = build_feature_vectors(pg_conn, t, seq_len=seq_len)
        if not seqs:
            print(f"[ANALYZE][Global] {t}: No sequences => skip.")
            continue
        anom, norm_l, thresh, raw_l = detect_lstm_anomalies(model, seqs)
        print(f"[ANALYZE][Global] {t}: threshold={thresh:.4f}, anomalies={np.sum(anom)}")
        ensure_lstm_columns(pg_conn, t)
        store_lstm_anomalies(pg_conn, t, row_ids, anom, norm_l, thresh, seq_len=seq_len)
        if plot:
            outpath = f"global_lstm_mse_{t}.png"
            plt.figure()
            plt.hist(raw_l, bins=50, alpha=0.7, label="Raw MSE")
            plt.axvline(x=np.mean(raw_l) + 2*np.std(raw_l),
                        color='r', linestyle='--', label="Threshold")
            plt.title(f"Global LSTM MSE Distribution - {t}")
            plt.xlabel("MSE")
            plt.ylabel("Frequency")
            plt.legend()
            plt.savefig(outpath)
            print(f"[ANALYZE][Global] MSE chart saved => {outpath}")

###############################################################################
# 8) Single-Table LSTM: Train/Analyze Functions
###############################################################################
def train_table_lstm_model(pg_conn, table_name, seq_len=16, hidden_size=16,
                           epochs=10, enrichdb_table="lstm_master", model_type="LSTM"):
    ensure_enrichdb_table(pg_conn, enrichdb_table)
    row_ids, seqs = build_feature_vectors(pg_conn, table_name, seq_len=seq_len)
    if not seqs:
        print(f"[TRAIN][{table_name}] => No sequences found, skipping training.")
        return None
    input_dim = seqs[0].shape[1]
    model = load_lstm_from_db(pg_conn, input_dim, hidden_size, seq_len,
                              enrichdb_table=enrichdb_table, model_type=model_type)
    if model is None:
        print(f"[TRAIN][{table_name}] Creating new LSTM (dim={input_dim}, hidden={hidden_size}, seq_len={seq_len}).")
        model = LogLSTMAnomalyDetector(input_dim, hidden_size)
    else:
        print(f"[TRAIN][{table_name}] Loaded existing model, continuing training...")
    model = train_lstm_anomaly(model, seqs, epochs=epochs, table_name=table_name)
    return model

def analyze_table_lstm(pg_conn, table_name, model, seq_len=16, plot=False,
                       enrichdb_table="lstm_master", model_type="LSTM"):
    if model is None:
        print(f"[ANALYZE][{table_name}] => No model, aborting.")
        return
    row_ids, seqs = build_feature_vectors(pg_conn, table_name, seq_len=seq_len)
    if not seqs:
        print(f"[ANALYZE][{table_name}] => No sequences, skipping.")
        return
    anom, norm_l, thresh, raw_l = detect_lstm_anomalies(model, seqs)
    print(f"[ANALYZE][{table_name}] => threshold={thresh:.4f}, anomalies={np.sum(anom)}")
    ensure_lstm_columns(pg_conn, table_name)
    store_lstm_anomalies(pg_conn, table_name, row_ids, anom, norm_l, thresh, seq_len=seq_len)
    if plot:
        outpath = f"table_lstm_mse_{table_name}.png"
        plt.figure()
        plt.hist(raw_l, bins=50, alpha=0.7, label="Raw MSE")
        plt.axvline(x=np.mean(raw_l) + 2*np.std(raw_l),
                    color='r', linestyle='--', label="Threshold")
        plt.title(f"Table LSTM MSE Distribution - {table_name}")
        plt.xlabel("MSE")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(outpath)
        print(f"[ANALYZE][{table_name}] MSE chart saved => {outpath}")

###############################################################################
# 9) MAIN: Dual-DB Mode for Training vs. Analysis
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="LSTM-based anomaly detection with dual-DB support.")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length for building sequences.")
    parser.add_argument("--hidden_size", type=int, default=16, help="Hidden dimension for LSTM.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--plot", action="store_true", help="If set, produce MSE distribution plots.")
    parser.add_argument("--enrichdb_table", type=str, default="lstm_master", help="Table to store the LSTM model.")
    parser.add_argument("--mode", choices=["train", "analyze"], default="analyze",
                        help="Operation mode: train or analyze.")
    parser.add_argument("--table_name", type=str, default=None,
                        help="If specified, single-table mode; otherwise, global mode is used.")
    # In our design, training always uses the happy_path DB and analysis uses the main DB for logs.
    parser.add_argument("--happy_path", action="store_true",
                        help="If set, use the happy_path DB for training/logs and model storage (for train mode).")
    args = parser.parse_args()

    cred_path = "credentials.txt"
    if not os.path.exists(cred_path):
        print("❌ Missing credentials.txt!")
        sys.exit(1)
    with open(cred_path) as f:
        creds = json.load(f)

    if args.mode == "train":
        # Connect to happy_path DB for logs and model storage.
        db_host = creds["DB_HOST"]
        db_name = creds["HAPPY_PATH_DB"]
        db_user = creds["HAPPY_PATH_USER"]
        db_pass = creds["HAPPY_PATH_PASSWORD"]
        logging.info(f"[TRAIN] Using happy_path DB: {db_name}")
        pg_conn = psycopg2.connect(host=db_host, database=db_name, user=db_user, password=db_pass)
        ensure_enrichdb_table(pg_conn, args.enrichdb_table)
        if args.table_name is None:
            print("[TRAIN] => Global LSTM model mode.")
            model = train_global_lstm_model(pg_conn,
                                            seq_len=args.seq_len,
                                            hidden_size=args.hidden_size,
                                            epochs=args.epochs,
                                            enrichdb_table=args.enrichdb_table,
                                            model_type="LSTM_GLOBAL")
            if model:
                store_lstm_in_db(pg_conn, model,
                                 input_dim=2,
                                 hidden_size=args.hidden_size,
                                 seq_len=args.seq_len,
                                 enrichdb_table=args.enrichdb_table,
                                 model_type="LSTM_GLOBAL")
                print("[TRAIN] => Global model stored in happy_path DB.")
        else:
            print(f"[TRAIN] => Single-table mode: {args.table_name}")
            model = train_table_lstm_model(pg_conn,
                                           table_name=args.table_name,
                                           seq_len=args.seq_len,
                                           hidden_size=args.hidden_size,
                                           epochs=args.epochs,
                                           enrichdb_table=args.enrichdb_table,
                                           model_type="LSTM")
            if model:
                store_lstm_in_db(pg_conn, model,
                                 input_dim=2,
                                 hidden_size=args.hidden_size,
                                 seq_len=args.seq_len,
                                 enrichdb_table=args.enrichdb_table,
                                 model_type="LSTM")
                print(f"[TRAIN] => Model for {args.table_name} stored in happy_path DB.")
        pg_conn.close()

    else:
        # Analyze mode:
        # - Logs are read from the main DB.
        # - The LSTM model is loaded from the happy_path DB.
        main_db_host = creds["DB_HOST"]
        main_db_name = creds["DB_NAME"]
        main_db_user = creds["DB_USER"]
        main_db_pass = creds["DB_PASSWORD"]
        logging.info(f"[ANALYZE] Using main DB for logs: {main_db_name}")
        pg_conn_main = psycopg2.connect(host=main_db_host, database=main_db_name, user=main_db_user, password=main_db_pass)
        happy_db_host = creds["DB_HOST"]
        happy_db_name = creds["HAPPY_PATH_DB"]
        happy_db_user = creds["HAPPY_PATH_USER"]
        happy_db_pass = creds["HAPPY_PATH_PASSWORD"]
        logging.info(f"[ANALYZE] Loading model from happy_path DB: {happy_db_name}")
        pg_conn_happy = psycopg2.connect(host=happy_db_host, database=happy_db_name, user=happy_db_user, password=happy_db_pass)
        if args.table_name is None:
            model = load_lstm_from_db(pg_conn_happy,
                                      input_dim=2,
                                      hidden_size=args.hidden_size,
                                      seq_len=args.seq_len,
                                      enrichdb_table=args.enrichdb_table,
                                      model_type="LSTM_GLOBAL")
            analyze_global_lstm(pg_conn_main, model,
                                seq_len=args.seq_len,
                                plot=args.plot,
                                enrichdb_table=args.enrichdb_table,
                                hidden_size=args.hidden_size)
        else:
            model = load_lstm_from_db(pg_conn_happy,
                                      input_dim=2,
                                      hidden_size=args.hidden_size,
                                      seq_len=args.seq_len,
                                      enrichdb_table=args.enrichdb_table,
                                      model_type="LSTM")
            analyze_table_lstm(pg_conn_main, args.table_name, model,
                               seq_len=args.seq_len, plot=args.plot)
        pg_conn_main.close()
        pg_conn_happy.close()

if __name__ == "__main__":
    main()
