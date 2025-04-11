#!/usr/bin/env python3
"""
train_model_web.py

This script runs a Flask web server that displays the progress of training a global machine
learning model from all tables in chatbotdb whose names match '^R[0-9]{10}$'. It gathers
all rows (without filtering on lstm_is_anomaly) so that the model sees the entire log context,
including events that might lead up to anomalies. It then trains a single global RandomForest
model, serializes it, and stores it in enrichdb. Throughout the process, debug messages are
logged and streamed to a web page using Server-Sent Events (SSE).

The web GUI displays:
  1. Learning progress over each table processed.
  2. Details during the training iteration (e.g. vectorized feature matrix shape,
     sample text and label, and top feature importances).
  3. A simple explanation of what is being learned.

Usage:
  python train_model_web.py
"""

import os
import json
import time
import base64
import pickle
import threading
import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor
from flask import Flask, Response, render_template_string, stream_with_context

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.ensemble import RandomForestClassifier

    sklearn_installed = True
except ImportError:
    sklearn_installed = False
    print("[WARN] scikit-learn not installed. ML-based classification won't run automatically.")

# --- Global progress tracking ---
app = Flask(__name__)
progress_log = []  # List to store progress messages
progress_lock = threading.Lock()


def log_progress(message: str):
    with progress_lock:
        progress_log.append(message)
    print(message)


# --- Credentials & DB Connections ---
def load_credentials():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    creds_path = os.path.join(base_dir, "credentials.txt")
    if not os.path.exists(creds_path):
        log_progress(f"[ERROR] credentials.txt not found at {creds_path}")
        return None
    with open(creds_path, "r") as f:
        return json.load(f)


def connect_db(host, dbname, user, password):
    try:
        conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
        return conn
    except Exception as e:
        log_progress(f"[ERROR] Unable to connect to database {dbname} at {host}: {e}")
        return None


# --- Data Gathering Functions ---
def get_all_r_tables(pg_conn):
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


def gather_table_data(pg_conn, table_name):
    query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        try:
            cur.execute(query)
            rows = cur.fetchall()
            log_progress(f"[DEBUG] Table {table_name}: gathered {len(rows)} rows.")
            return rows
        except Exception as e:
            log_progress(f"[ERROR] Error gathering data from table {table_name}: {e}")
            pg_conn.rollback()
            return []


def combine_training_data(pg_conn, tables):
    all_data = []
    for tbl in tables:
        rows = gather_table_data(pg_conn, tbl)
        all_data.extend(rows)
        log_progress(f"[INFO] Processed table {tbl}, total rows so far: {len(all_data)}")
    log_progress(f"[INFO] Total training rows gathered: {len(all_data)}")
    return all_data


def extract_features_and_labels(data_rows):
    texts = []
    labels = []
    for row in data_rows:
        # Combine all column values into a single text string.
        row_text = " ".join([str(val) for val in row.values() if val is not None])
        texts.append(row_text)
        # Ensure event_type is a string; use "UNKNOWN" if missing or empty.
        label = row.get("event_type")
        if label is None or not str(label).strip():
            label = "UNKNOWN"
        else:
            label = str(label)
        labels.append(label)
    if texts:
        log_progress(f"[DEBUG] Sample training text (first 10000 chars): {texts[0][:1000]}...")
        log_progress(f"[DEBUG] Sample label: {labels[0]}")
    return texts, labels


# --- Model Training & Storage ---
def train_global_model(texts, labels):
    log_progress(f"[ML] Starting training on {len(texts)} samples.")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    log_progress(f"[ML] Vectorized text into feature matrix of shape {X.shape}.")
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, labels)
    log_progress("[ML] RandomForest model training complete.")
    # Log top 10 features if available.
    if hasattr(clf, 'feature_importances_'):
        import numpy as np
        importances = clf.feature_importances_
        feature_names = vectorizer.get_feature_names_out()
        indices = np.argsort(importances)[::-1]
        topN = min(10, len(indices))
        log_progress("[ML] Top feature importances:")
        for idx in indices[:topN]:
            log_progress(f"    {feature_names[idx]}: {importances[idx]:.4f}")
    return vectorizer, clf


def serialize_model(vectorizer, clf):
    model_tuple = (vectorizer, clf)
    pickled = pickle.dumps(model_tuple)
    b64_encoded = base64.b64encode(pickled).decode('utf-8')
    return b64_encoded


def store_model_in_enrichdb(enrich_conn, model_blob, model_name="global_model"):
    create_query = """
    CREATE TABLE IF NOT EXISTS ml_models (
        model_name TEXT PRIMARY KEY,
        model_blob TEXT
    )
    """
    with enrich_conn.cursor() as cur:
        cur.execute(create_query)
        enrich_conn.commit()
    upsert_query = """
    INSERT INTO ml_models (model_name, model_blob)
    VALUES (%s, %s)
    ON CONFLICT (model_name)
    DO UPDATE SET model_blob = EXCLUDED.model_blob
    """
    with enrich_conn.cursor() as cur:
        cur.execute(upsert_query, (model_name, model_blob))
        enrich_conn.commit()
    log_progress(f"[INFO] Model '{model_name}' stored in enrichdb.")


# --- Flask Web Server for Progress ---
@app.route("/")
def index():
    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Global Model Training Progress</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; }
          #progress { border: 1px solid #ccc; padding: 10px; height: 400px; overflow-y: scroll; background: #f9f9f9; }
          button { padding: 10px 20px; font-size: 16px; }
        </style>
      </head>
      <body>
        <h1>Global Model Training Progress</h1>
        <button onclick="startTraining()">Start Training</button>
        <h2>Progress Log:</h2>
        <div id="progress"></div>
        <script>
          var evtSource = new EventSource("/progress");
          evtSource.onmessage = function(e) {
            var progressDiv = document.getElementById("progress");
            progressDiv.innerHTML += e.data + "<br/>";
            progressDiv.scrollTop = progressDiv.scrollHeight;
          };
          function startTraining() {
            fetch("/start")
              .then(response => response.text())
              .then(text => alert(text));
          }
        </script>
      </body>
    </html>
    """
    return html


@app.route("/start")
def start_training():
    thread = threading.Thread(target=training_process)
    thread.start()
    return "Training started in background."


@app.route("/progress")
def progress():
    def generate():
        last_idx = 0
        while True:
            with progress_lock:
                if last_idx < len(progress_log):
                    for msg in progress_log[last_idx:]:
                        yield f"data: {msg}\n\n"
                    last_idx = len(progress_log)
            time.sleep(1)

    return Response(generate(), mimetype="text/event-stream")


# --- Background Training Process ---
def training_process():
    log_progress("=== Starting Global Model Training Process ===")
    creds = load_credentials()
    if not creds:
        log_progress("[ERROR] No credentials loaded. Aborting training.")
        return

    # Connect to logs DB (chatbotdb)
    pg_conn = connect_db(creds["DB_HOST"], creds["DB_NAME"], creds["DB_USER"], creds["DB_PASSWORD"])
    if not pg_conn:
        log_progress("[ERROR] Could not connect to logs DB. Aborting.")
        return

    # Connect to enrich DB
    enrich_conn = connect_db(creds["ENRICH_DB_HOST"], creds["ENRICH_DB"], creds["ENRICH_DB_USER"],
                             creds["ENRICH_DB_PASSWORD"])
    if not enrich_conn:
        log_progress("[ERROR] Could not connect to enrich DB. Aborting.")
        pg_conn.close()
        return

    # Get all tables matching '^R[0-9]{10}$'
    tables = get_all_r_tables(pg_conn)
    if not tables:
        log_progress("[ERROR] No tables matching '^R[0-9]{10}$' found in chatbotdb.")
        pg_conn.close()
        enrich_conn.close()
        return
    log_progress(f"[DEBUG] Found tables: {tables}")

    # Gather all training data (all rows from all tables)
    all_data = combine_training_data(pg_conn, tables)
    if not all_data:
        log_progress("[ERROR] No training data found across tables.")
        pg_conn.close()
        enrich_conn.close()
        return

    # Extract features and labels
    texts, labels = extract_features_and_labels(all_data)

    # (Optional) Iterate multiple times and log progress per iteration.
    # For simplicity we do a single training iteration here.
    log_progress("[INFO] Starting training iteration 1.")
    vectorizer, clf = train_global_model(texts, labels)
    log_progress("[INFO] Training iteration 1 complete.")

    # Serialize and store the model in enrichdb
    model_blob = serialize_model(vectorizer, clf)
    store_model_in_enrichdb(enrich_conn, model_blob, model_name="global_model")

    log_progress("=== Global Model Training Process Complete ===")
    pg_conn.close()
    enrich_conn.close()


# --- Main Entry Point: Run Flask Web Server ---
if __name__ == "__main__":
    if not sklearn_installed:
        log_progress("[WARN] scikit-learn not installed. Exiting.")
    else:
        app.run(debug=True, host="0.0.0.0", port=4000)
