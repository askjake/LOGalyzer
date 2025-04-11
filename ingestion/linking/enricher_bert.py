#!/usr/bin/env python3
"""
ingestion/linking/enricher_bert.py

Refactored global log enrichment script that leverages:
1. BERT-based embeddings (using TF-IDF vectorizer) for textual columns.
2. Forced inclusion of domain-knowledge columns (e.g. svc_name, video_resolution, connection_status, etc.).
3. A hybrid feature vector combining:
    - TF-IDF text features (with dimensionality reduction via TruncatedSVD)
    - Structured features from domain columns, where numeric features are scaled and categorical features hashed via FeatureHasher.
4. A single global model is trained on all R+10 tables from a chosen DB (main or HAPPY_PATH)
   and then used to update each table’s event_type and significance_score.
5. The entire enrichment pipeline is stored persistently in the ml_config table (under key "enricher_bert_model")
   so that subsequent runs use the same feature space.

Usage examples:
    python enricher_bert.py --happy_path
    python enricher_bert.py --table_name R1898891907
"""

import os
import re
import json
import logging
import argparse
import subprocess
import numpy as np
import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mutual_info_score
import pickle
import base64

try:
    import torch
    from transformers import AutoTokenizer, AutoModel

    transformers_installed = True
except ImportError:
    logging.error("[ERROR] transformers not installed. BERT embeddings won't be used.")
    transformers_installed = False

# Logging configuration
logging.basicConfig(
    level=logging.ERROR,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DEFAULT_NUMERIC_VALUE = 0.0

###############################################################################
# 1) Database and Table Utilities
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
# 2) Regex-based Data Extraction & Domain Knowledge Helpers
###############################################################################
INTERFACE_STATUS_REGEX = re.compile(r'Interface (\w+) (Connected|Not Connected)\(?(-?\d+)?\)?')
AUTO_IP_THREAD_REGEX = re.compile(r'Need to start Auto IP thread for (\w+), thread \(([^)]+)\), event (-?\d+)')
ARPING_REGEX = re.compile(r'Arping for (\d+\.\d+\.\d+\.\d+)')
PACKET_INFO_REGEX = re.compile(r'(Destin\.|Source) MAC: ([0-9A-Fa-f:]+)')
ARP_DETAILS_REGEX = re.compile(r'(Sender|Target) (IP|MAC)\s*:\s*([0-9A-Fa-f:.]+)')
SGS_MSG_REGEX = re.compile(r'SGS Msg <(\d+):(\w+)> took <(\d+) ms> SGS Return Code:<(\d+):(\w+)> rx_id:<([\w-]+)>')
MAC_SELECT_THREAD_REGEX = re.compile(r'MAC Select thread running on <(\w+)> interface')
WORKER_THREAD_REGEX = re.compile(r'starting thread:<(\w+ thread)>, version:<(\d+)>')

MALLOC_OCCUPIED_REGEX = re.compile(r"This is the total size of memory occupied by chunks handed out by malloc:(\d+)")
FREE_CHUNKS_REGEX = re.compile(r"This is the total size of memory occupied by free \(not in use\) chunks\.:(\d+)")
TOPMOST_CHUNK_REGEX = re.compile(r"This is the size of the top-most releasable chunk.*?:(\d+)")
FREED_UP_BYTES_REGEX = re.compile(r"Freed up bytes:\s*(\d+)")
USED_MEMORY_BEFORE_REGEX = re.compile(r"Used memory before GC:\s*(\d+)")
USED_MEMORY_AFTER_REGEX = re.compile(r"Used memory after GC:\s*(\d+)")
ALLOCATED_MMAP_REGEX = re.compile(r"Allocated\s*(\d+)\s*bytes\s*in\s*(\d+)\s*chunks\.\s*\(via mmap\)")
ALLOCATED_MALLOC_REGEX = re.compile(r"Allocated\s*(\d+)\s*bytes\s.*\(via malloc\)")

MARK_LOGS_PATTERN = "ES_STB_MSG_TYPE_VIDEO_DUMP_BRCM_PROC"
DUMP_FILE_PATTERN = "DUMPING FILE "


def extract_svc_name(line_data):
    match = re.search(r"svc_name\s*=\s*([\w\-]+)", line_data, re.IGNORECASE)
    return match.group(1) if match else None


def extract_video_resolution(line_data):
    match = re.search(r"video resolution[:\s]+(\d{3,4}x\d{3,4})", line_data, re.IGNORECASE)
    return match.group(1) if match else None


def parse_display_drop_value(line_data):
    match = re.search(r"Display Drop Detected\s*=\s*(\d+)", line_data)
    return int(match.group(1)) if match else 0


def parse_data_fields(data_value: str) -> dict:
    extracted = {}
    data_str = str(data_value or "").strip().strip('"')
    if MARK_LOGS_PATTERN in data_str or DUMP_FILE_PATTERN in data_str:
        extracted["customer_marked_flag"] = "YES"
        logging.info("⚠️ **** LOGS MARKED for customer review ****")
    m_interface = INTERFACE_STATUS_REGEX.search(data_str)
    if m_interface:
        extracted["interface"] = m_interface.group(1)
        extracted["connection_status"] = m_interface.group(2)
        extracted["interface_status_code"] = m_interface.group(3)
    m_auto_ip = AUTO_IP_THREAD_REGEX.search(data_str)
    if m_auto_ip:
        extracted["auto_ip_interface"] = m_auto_ip.group(1)
        extracted["auto_ip_thread"] = m_auto_ip.group(2)
        extracted["auto_ip_event"] = m_auto_ip.group(3)
    m_arping = ARPING_REGEX.search(data_str)
    if m_arping:
        extracted["arping_ip"] = m_arping.group(1)
    m_packet = PACKET_INFO_REGEX.search(data_str)
    if m_packet:
        extracted["packet_mac_type"] = m_packet.group(1)
        extracted["packet_mac"] = m_packet.group(2)
    m_arp_details = ARP_DETAILS_REGEX.search(data_str)
    if m_arp_details:
        extracted["arp_details_role"] = m_arp_details.group(1)
        extracted["arp_details_type"] = m_arp_details.group(2)
        extracted["arp_details_value"] = m_arp_details.group(3)
    m_sgs = SGS_MSG_REGEX.search(data_str)
    if m_sgs:
        extracted["sgs_msg_id"] = m_sgs.group(1)
        extracted["sgs_msg_type"] = m_sgs.group(2)
        extracted["sgs_duration_ms"] = m_sgs.group(3)
        extracted["sgs_return_code"] = m_sgs.group(4)
        extracted["sgs_return_desc"] = m_sgs.group(5)
        extracted["sgs_rx_id"] = m_sgs.group(6)
    m_mac_select = MAC_SELECT_THREAD_REGEX.search(data_str)
    if m_mac_select:
        extracted["mac_select_interface"] = m_mac_select.group(1)
    m_worker = WORKER_THREAD_REGEX.search(data_str)
    if m_worker:
        extracted["worker_thread"] = m_worker.group(1)
        try:
            extracted["worker_thread_version"] = int(m_worker.group(2))
        except ValueError:
            extracted["worker_thread_version"] = m_worker.group(2)
    m_malloc = MALLOC_OCCUPIED_REGEX.search(data_str)
    if m_malloc:
        extracted["malloc_occupied"] = m_malloc.group(1)
        extracted["malloc_occupied_num"] = m_malloc.group(1)
    m_free_chunks = FREE_CHUNKS_REGEX.search(data_str)
    if m_free_chunks:
        extracted["free_chunks"] = m_free_chunks.group(1)
        extracted["free_chunks_num"] = m_free_chunks.group(1)
    m_topmost = TOPMOST_CHUNK_REGEX.search(data_str)
    if m_topmost:
        extracted["topmost_chunk"] = m_topmost.group(1)
        extracted["topmost_chunk_num"] = m_topmost.group(1)
    m_freed = FREED_UP_BYTES_REGEX.search(data_str)
    if m_freed:
        extracted["freed_up_bytes"] = m_freed.group(1)
        extracted["freed_up_bytes_num"] = m_freed.group(1)
    m_used_before = USED_MEMORY_BEFORE_REGEX.search(data_str)
    if m_used_before:
        extracted["used_memory_before_gc"] = m_used_before.group(1)
        extracted["used_memory_before_gc_num"] = m_used_before.group(1)
    m_used_after = USED_MEMORY_AFTER_REGEX.search(data_str)
    if m_used_after:
        extracted["used_memory_after_gc"] = m_used_after.group(1)
        extracted["used_memory_after_gc_num"] = m_used_after.group(1)
    m_allocated_mmap = ALLOCATED_MMAP_REGEX.search(data_str)
    if m_allocated_mmap:
        extracted["allocated_mmap_bytes"] = m_allocated_mmap.group(1)
        extracted["allocated_mmap_chunks"] = m_allocated_mmap.group(2)
        extracted["allocated_mmap_bytes_num"] = m_allocated_mmap.group(1)
        extracted["allocated_mmap_chunks_num"] = m_allocated_mmap.group(2)
    m_allocated_malloc = ALLOCATED_MALLOC_REGEX.search(data_str)
    if m_allocated_malloc:
        extracted["allocated_malloc_bytes"] = m_allocated_malloc.group(1)
        extracted["allocated_malloc_bytes_num"] = m_allocated_malloc.group(1)
    svc_name = extract_svc_name(data_str)
    if svc_name:
        extracted["svc_name"] = svc_name
    video_res = extract_video_resolution(data_str)
    if video_res:
        extracted["video_resolution"] = video_res
    return extracted


###############################################################################
# 2) Domain-Knowledge and Always-Text Columns
###############################################################################
FORCED_DOMAIN_COLUMNS = [
    "svc_name",
    "video_resolution",
    "connection_status",
    "category",
    "file_line",
    "function",
    "timestamp",
    "rx_id",
    "software",
    "model",
]
ALWAYS_TEXT_COLS = ["data", "info", "details"]


###############################################################################
# 3) ML Config Table Management
###############################################################################
def ensure_ml_config_table(enrich_conn):
    with enrich_conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ml_config (
                config_key TEXT PRIMARY KEY,
                config_value JSONB
            );
        """)
    enrich_conn.commit()


def load_enricher_config(enrich_conn):
    with enrich_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT config_value FROM ml_config WHERE config_key = 'enricher_settings'")
        row = cur.fetchone()
        if row:
            return row["config_value"]
    return None


def store_enricher_config(enrich_conn, config):
    with enrich_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ml_config (config_key, config_value)
            VALUES ('enricher_settings', %s)
            ON CONFLICT (config_key)
            DO UPDATE SET config_value = EXCLUDED.config_value
        """, (json.dumps(config),))
    enrich_conn.commit()


###############################################################################
# NEW: Persistence of the Enricher Model (Pipeline)
###############################################################################
def load_enricher_model(enrich_conn):
    """
    Retrieves the base64-encoded pickled model from ml_config (under key='enricher_bert_model'),
    decodes it, and unpickles into a Python object. Returns None if not found or decoding fails.
    """
    with enrich_conn.cursor() as cur:
        cur.execute("""
            SELECT config_value
            FROM ml_config
            WHERE config_key = 'enricher_bert_model'
        """)
        row = cur.fetchone()
        if not row:
            return None  # No row => no stored model

        # row[0] is the JSONB data in config_value
        stored_json = row[0]
        if not stored_json or "model_data" not in stored_json:
            return None  # Malformed or missing model data

        try:
            # stored_json["model_data"] should be the base64 string
            b64_model = stored_json["model_data"]
            model_bytes = base64.b64decode(b64_model)
            trained_model = pickle.loads(model_bytes)
            logging.info("[enricher_bert] Loaded stored enricher model from DB successfully.")
            return trained_model
        except Exception as e:
            logging.error(f"[enricher_bert] Error decoding stored enricher model: {e}")
            return None


def store_enricher_model(enrich_conn, trained_model):
    """
    Serializes the trained_model via pickle -> base64, then stores it in ml_config
    under key='enricher_bert_model'. We store it as proper JSONB with
    {"model_data": "<base64 string>"}.
    """
    try:
        # 1) Pickle the model
        model_bytes = pickle.dumps(trained_model)

        # 2) Base64-encode the pickled bytes
        b64_model = base64.b64encode(model_bytes).decode("utf-8")

        # 3) Create a dictionary that will be stored as JSONB
        stored_dict = {"model_data": b64_model}

    except Exception as e:
        logging.error(f"[enricher_bert] Error encoding enricher model: {e}")
        return

    # Insert or update ml_config with JSONB data
    with enrich_conn.cursor() as cur:
        try:
            cur.execute("""
                INSERT INTO ml_config (config_key, config_value)
                VALUES ('enricher_bert_model', %s::jsonb)
                ON CONFLICT (config_key)
                DO UPDATE SET config_value = EXCLUDED.config_value
            """, [json.dumps(stored_dict)])
            enrich_conn.commit()
            logging.info("[enricher_bert] Stored enricher model in DB (enricher_bert_model).")
        except Exception as e:
            enrich_conn.rollback()
            logging.error(f"[enricher_bert] DB error storing model: {e}")
###############################################################################
# 4) Schema Inspection and Relevant Columns
###############################################################################
def get_table_columns(pg_conn, schema_name, table_name):
    q = sql.SQL("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
    """)
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(q, (schema_name, table_name))
        rows = cur.fetchall()
    return {row["column_name"]: row["data_type"] for row in rows}


def detect_relevant_columns_table(pg_conn, table_name, sample_size=2000):
    from sklearn.metrics import mutual_info_score
    schema_part, table_part = parse_schema_and_table(table_name)
    try:
        col_info = get_table_columns(pg_conn, schema_part, table_part)
    except psycopg2.Error as e:
        logging.warning(f"Table {table_name} does not exist; skipping. Error: {e}")
        pg_conn.rollback()
        return []
    text_columns = [c for c, dtype in col_info.items() if ('text' in dtype or 'char' in dtype)]
    if "event_type" in col_info and "event_type" not in text_columns:
        text_columns.append("event_type")
    col_select = ", ".join(f'"{c}"' for c in text_columns)
    q = f'''
        SELECT {col_select}
        FROM "{schema_part}"."{table_part}"
        ORDER BY random()
        LIMIT {sample_size}
    '''
    sample_rows = []
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        try:
            cur.execute(q)
            sample_rows = cur.fetchall()
        except Exception as e:
            logging.warning(f"Sampling {schema_part}.{table_part} failed: {e}")
            return []
    if not sample_rows:
        return text_columns
    event_labels = [(r["event_type"] or "UNKNOWN") for r in sample_rows]
    if all(evt == "UNKNOWN" for evt in event_labels):
        return text_columns
    results = []
    for c in text_columns:
        if c == "event_type":
            continue
        col_vals = [row.get(c) or "NULL" for row in sample_rows]
        s = mutual_info_score(col_vals, event_labels)
        results.append((c, s))
    results.sort(key=lambda x: x[1], reverse=True)
    top_texts = [x[0] for x in results[:5]]
    top_texts.append("event_type")
    return list(set(top_texts))


def detect_global_relevant_columns(pg_conn, tables, sample_size=10000):
    global_cols = set()
    for t in tables:
        top_texts = detect_relevant_columns_table(pg_conn, t, sample_size)
        global_cols.update(top_texts)
    for forced in FORCED_DOMAIN_COLUMNS:
        global_cols.add(forced)
    for txtcol in ALWAYS_TEXT_COLS:
        global_cols.add(txtcol)
    return list(global_cols)


###############################################################################
# 5) Global Model Building: Hybrid Approach
###############################################################################
def get_bert_embedding(text, tokenizer, model):
    """Compute mean-pooled BERT embedding for text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb


def gather_global_training_data(pg_conn, tables, relevant_cols):
    from psycopg2.extras import DictCursor
    all_rows = []
    for t in tables:
        schema_part, table_part = parse_schema_and_table(t)
        try:
            col_info = get_table_columns(pg_conn, schema_part, table_part)
        except psycopg2.Error as e:
            logging.warning(f"Table {t} missing or error: {e}")
            pg_conn.rollback()
            continue
        use_cols = [c for c in relevant_cols if c in col_info]
        if not use_cols:
            logging.warning(f"Table {t} => no relevant columns found.")
            continue
        col_select = ", ".join(f'"{c}"' for c in use_cols)
        q = f'''
            SELECT {col_select}
            FROM "{schema_part}"."{table_part}"
            WHERE "event_type" IS NOT NULL
        '''
        with pg_conn.cursor(cursor_factory=DictCursor) as cur:
            try:
                cur.execute(q)
                rows = cur.fetchall()
                all_rows.extend(rows)
            except Exception as ex:
                logging.warning(f"Reading table {t}: {ex}")
                pg_conn.rollback()
                continue
    return all_rows


def build_global_model(all_data_rows, relevant_cols, config):
    """
    Hybrid approach:
      - For textual features: use TF-IDF with max_features (default=1000) and reduce dimensionality via TruncatedSVD (default=50 dims).
      - For domain features: numeric features are scaled and categorical features hashed.
      - Concatenate both parts and train a RandomForest classifier.
    """
    if not all_data_rows:
        logging.info("No training data found. Not building a model.")
        return None

    def is_numeric(val):
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

    def is_text_col(col: str):
        return (col in ALWAYS_TEXT_COLS) or (col not in FORCED_DOMAIN_COLUMNS)

    text_columns = [c for c in relevant_cols if is_text_col(c)]
    domain_columns = [c for c in relevant_cols if (c in FORCED_DOMAIN_COLUMNS) and c != "event_type"]

    text_samples = []
    labels = []
    domain_samples = []
    for row in all_data_rows:
        label = (row.get("event_type") or "").strip() or row.get("category", "UNKNOWN")
        labels.append(label)
        text_piece = [str(row.get(c)) for c in text_columns if row.get(c)]
        text_samples.append(" ".join(text_piece))
        domain_row = {}
        for c in domain_columns:
            val = row.get(c)
            if is_numeric(val):
                try:
                    domain_row[c] = float(val)
                except:
                    domain_row[c] = str(val)
            else:
                domain_row[c] = str(val) if val is not None else ""
        domain_samples.append(domain_row)

    if not text_samples:
        logging.info("No textual samples found; aborting model building.")
        return None

    max_features = config.get("tfidf_max_features", 1000)
    tfidf = TfidfVectorizer(max_features=max_features)
    X_text_sparse = tfidf.fit_transform(text_samples)
    svd_components = config.get("svd_components", 50)
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    X_text_reduced = svd.fit_transform(X_text_sparse)

    numeric_keys = []
    for c in domain_columns:
        for d in domain_samples:
            if d[c] != "":
                if is_numeric(d[c]):
                    numeric_keys.append(c)
                break
    numeric_keys = list(set(numeric_keys))
    if numeric_keys:
        X_numeric = np.array([[d.get(c, 0) for c in numeric_keys] for d in domain_samples], dtype=float)
        scaler = StandardScaler()
        X_numeric_scaled = scaler.fit_transform(X_numeric)
    else:
        X_numeric_scaled = np.empty((len(domain_samples), 0))
    categorical_keys = [c for c in domain_columns if c not in numeric_keys]
    if categorical_keys:
        from sklearn.feature_extraction import FeatureHasher
        cat_samples = [{col: d.get(col, "") for col in categorical_keys} for d in domain_samples]
        hasher = FeatureHasher(n_features=100, input_type="dict")
        X_categ_encoded = hasher.transform(cat_samples).toarray()
    else:
        X_categ_encoded = np.empty((len(domain_samples), 0))
    try:
        X_domain = np.hstack([X_numeric_scaled, X_categ_encoded])
    except ValueError as e:
        logging.error(f"Error concatenating domain features: {e}")
        return None

    try:
        X_all = np.hstack([X_text_reduced, X_domain])
    except ValueError as e:
        logging.error(f"Error concatenating text and domain features: {e}")
        return None

    Y = np.array(labels)

    model_type = config.get("model_type", "RandomForest")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_all, Y)
    logging.info(f"Trained single global {model_type} model on {len(X_all)} samples.")

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        tfidf_features = tfidf.get_feature_names_out()
        domain_feature_names = []
        if numeric_keys:
            domain_feature_names.extend([f"{col}_numeric" for col in numeric_keys])
        if categorical_keys:
            domain_feature_names.extend([f"hash_{i}" for i in range(100)])
        feature_names = np.concatenate([tfidf_features, domain_feature_names])
        indices = np.argsort(importances)[::-1]
        topN = min(20, len(indices))
        logging.info("Top feature importances:")
        for idx in indices[:topN]:
            logging.info(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    trained_model = {
        "model_version": "1.0",
        "tfidf": tfidf,
        "svd": svd,
        "clf": clf,
        "numeric_keys": numeric_keys,
        "categorical_keys": categorical_keys,
        "scaler": scaler if numeric_keys else None,
        "hasher": hasher if categorical_keys else None,
    }
    return trained_model


###############################################################################
# 6) Enrichment Steps: Predict using the Global Model
###############################################################################
def predict_event_type(row_dict, trained_model, relevant_cols):
    if not trained_model:
        return "UNKNOWN"
    text_cols = []
    for c in relevant_cols:
        if c == "event_type":
            continue
        if c in ALWAYS_TEXT_COLS or c not in FORCED_DOMAIN_COLUMNS:
            val = row_dict.get(c, "")
            if val:
                text_cols.append(str(val))
    combined_text = " ".join(text_cols)
    tfidf = trained_model["tfidf"]
    # IMPORTANT: Apply TF-IDF and then SVD to match training.
    X_text_sparse = tfidf.transform([combined_text])
    X_text_reduced = trained_model["svd"].transform(X_text_sparse)

    numeric_keys = trained_model.get("numeric_keys", [])
    categorical_keys = trained_model.get("categorical_keys", [])
    domain_features = []
    for col in numeric_keys:
        try:
            val = float(row_dict.get(col, 0))
        except:
            val = 0.0
        domain_features.append(val)
    X_numeric = np.array(domain_features).reshape(1, -1) if numeric_keys else np.empty((1, 0))

    cat_sample = {}
    for col in categorical_keys:
        val = row_dict.get(col)
        if val is None:
            val = ""
        elif hasattr(val, "isoformat"):
            val = val.isoformat()
        else:
            val = str(val)
        cat_sample[col] = val
    if categorical_keys and trained_model.get("hasher") is not None:
        hasher = trained_model["hasher"]
        X_cat_enc = hasher.transform([cat_sample]).toarray()
    else:
        X_cat_enc = np.empty((1, 0))

    if numeric_keys and trained_model.get("scaler") is not None:
        scaler = trained_model["scaler"]
        X_numeric_scaled = scaler.transform(X_numeric)
    else:
        X_numeric_scaled = X_numeric

    X_domain = np.hstack([X_numeric_scaled, X_cat_enc])
    X_final = np.hstack([X_text_reduced, X_domain])
    label = trained_model["clf"].predict(X_final)
    return label[0] if len(label) else "UNKNOWN"


def add_col_if_missing(cur, schema_name, table_name, col_name, col_type="TEXT"):
    existing = get_table_columns(cur.connection, schema_name, table_name)
    if col_name in existing:
        return
    sql_add = f'ALTER TABLE "{schema_name}"."{table_name}" ADD COLUMN "{col_name}" {col_type}'
    cur.execute(sql_add)


def groom_table_schema(pg_conn, table_name, required_cols):
    """
    Ensures that the given table has all columns in required_cols.
    Each element of required_cols is a tuple (col_name, col_type).
    """
    schema_part, table_part = parse_schema_and_table(table_name)
    with pg_conn.cursor() as cur:
        for col_name, col_type in required_cols:
            existing = get_table_columns(cur.connection, schema_part, table_part)
            if col_name not in existing:
                sql_add = f'ALTER TABLE "{schema_part}"."{table_part}" ADD COLUMN "{col_name}" {col_type}'
                logging.info(f"[GROOM] Adding missing column {col_name} ({col_type}) to table {table_name}.")
                cur.execute(sql_add)
    pg_conn.commit()

def determine_col_type(sample_value):
    """
    Given a sample value, return a PostgreSQL type.
    If the sample can be cast to float, return "DOUBLE PRECISION";
    otherwise, return "TEXT".
    """
    try:
        float(sample_value)
        return "DOUBLE PRECISION"
    except (ValueError, TypeError):
        return "TEXT"

def get_table_columns(pg_conn, schema_name, table_name):
    """
    Returns a dictionary of {column_name: data_type} for the given table.
    """
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
    add it using a type based on a sample value.
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
    schema_part, table_part = parse_schema_and_table(table_name)
    # Ensure that all columns from relevant_cols exist
    with pg_conn.cursor() as cur:
        for c in relevant_cols:
            add_col_if_missing(cur, schema_part, table_part, c)
        add_col_if_missing(cur, schema_part, table_part, "significance_score", col_type="DOUBLE PRECISION")
    pg_conn.commit()

    # Fetch rows for enrichment (include "data" column to extract extra keys)
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
    # Collect union of keys from all parsed "data" fields
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
                # Build union: if key not seen before, store the sample value
                for key, val in extracted.items():
                    if key not in extracted_keys_union and val is not None:
                        extracted_keys_union[key] = val

    # Ensure any extracted key is present as a column in the table
    if extracted_keys_union:
        add_missing_extracted_columns(pg_conn, schema_part, table_part, extracted_keys_union)

    # Now update rows with extracted data and event_type predictions
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
# 7) Significance Heuristic (Retained)
###############################################################################
def analyze_significance(pg_conn, table_name):
    schema_part, table_part = parse_schema_and_table(table_name)
    with pg_conn.cursor() as cur:
        cur.execute(f'''
            ALTER TABLE "{schema_part}"."{table_part}"
            ADD COLUMN IF NOT EXISTS significance_score DOUBLE PRECISION;
        ''')
    pg_conn.commit()
    numeric_cols = []
    with pg_conn.cursor() as cur:
        cur.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
              AND data_type IN ('integer', 'bigint', 'double precision', 'real', 'numeric');
        """, (schema_part, table_part))
        numeric_cols = [row[0] for row in cur.fetchall()]
    if not numeric_cols:
        logging.warning(f"[SIGNIFICANCE] No numeric columns found in {table_name}, skipping analysis.")
        return
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        query = f'SELECT id, {", ".join(numeric_cols)} FROM "{schema_part}"."{table_part}"'
        cur.execute(query)
        rows = cur.fetchall()
    if not rows:
        logging.warning(f"[SIGNIFICANCE] No rows in {table_name}, skipping analysis.")
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    available_numeric_cols = [col for col in numeric_cols if col in df.columns]
    if not available_numeric_cols:
        logging.warning(f"[SIGNIFICANCE] None of the numeric columns are present in {table_name}.")
        return
    df.dropna(subset=available_numeric_cols, how="all", inplace=True)
    logging.info(f"[SIGNIFICANCE] {len(df)} rows remain after dropping rows with all NaNs in numeric columns.")
    if df.empty:
        logging.warning(f"[SIGNIFICANCE] No valid numeric data in {table_name}.")
        return
    for col in available_numeric_cols:
        col_data = df[col].dropna()
        if not col_data.empty:
            min_val, max_val = col_data.min(), col_data.max()
            if min_val != max_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0
    df["significance_score"] = df[available_numeric_cols].mean(axis=1)
    update_values = [(float(score), int(row_id)) for row_id, score in zip(df["id"], df["significance_score"])]
    if update_values:
        with pg_conn.cursor() as cur:
            update_query = f'''
                UPDATE "{schema_part}"."{table_part}"
                SET significance_score = %s
                WHERE id = %s;
            '''
            cur.executemany(update_query, update_values)
        pg_conn.commit()
        logging.info(f"[SIGNIFICANCE] Updated significance_score for {len(update_values)} rows in {table_name}.")
    highlight_customer_marked(pg_conn, table_name, window_minutes=1, significance_boost=5)


def highlight_customer_marked(pg_conn, table_name, window_minutes=1, significance_boost=5):
    schema_part, table_part = parse_schema_and_table(table_name)
    with pg_conn.cursor() as cur:
        add_col_if_missing(cur, schema_part, table_part, "customer_marked_flag", col_type="TEXT")
        add_col_if_missing(cur, schema_part, table_part, "important_investigate", col_type="BOOLEAN")
        cur.execute(f'''
        ALTER TABLE "{schema_part}"."{table_part}"
        ADD COLUMN IF NOT EXISTS significance_score DOUBLE PRECISION
        ''')
    pg_conn.commit()
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        try:
            cur.execute(f'''
            SELECT id, "timestamp"
            FROM "{schema_part}"."{table_part}"
            WHERE customer_marked_flag='YES'
              AND "timestamp" IS NOT NULL
            ''')
            marked_rows = cur.fetchall()
        except psycopg2.Error as e:
            logging.error(f"Could not fetch customer_marked_flag from {table_name}: {e}")
            pg_conn.rollback()
            return
    if not marked_rows:
        return
    with pg_conn.cursor() as cur:
        for row in marked_rows:
            tstamp = row["timestamp"]
            if not tstamp:
                continue
            sql_update = f'''
            UPDATE "{schema_part}"."{table_part}"
            SET significance_score = COALESCE(significance_score,0)::double precision + %s,
                important_investigate = TRUE
            WHERE "timestamp" BETWEEN %s - INTERVAL '{window_minutes} minutes'
                                 AND %s + INTERVAL '{window_minutes} minutes'
            '''
            cur.execute(sql_update, (significance_boost, tstamp, tstamp))
    pg_conn.commit()
    logging.info(
        f"[MARKED] {len(marked_rows)} row(s) marked for customer review in '{table_name}' with significance boost of +{significance_boost}.")


###############################################################################
# 8) MAIN: Global Enrichment Using the Single Global Model
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Global log enricher using BERT embeddings and TF-IDF for textual data combined with structured domain features."
    )
    parser.add_argument("--table_name", type=str,
                        help="If set, only process that single table. Otherwise, process all R+10 tables.")
    parser.add_argument("--happy_path", action="store_true",
                        help="If set, use the HAPPY_PATH_DB instead of the main DB for both training and model persistence.")
    parser.add_argument("--mode", type=str, choices=["train", "enrich"], default="enrich",
                        help="Mode of operation: train or enrich")
    args = parser.parse_args()

    # Load credentials and set up database connections
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    creds_path = os.path.join(base_dir, "credentials.txt")
    if not os.path.exists(creds_path):
        logging.error("credentials.txt not found.")
        return
    with open(creds_path, "r") as f:
        j = json.load(f)

    # Connect to the appropriate logs DB
    if args.happy_path:
        pg_conn = psycopg2.connect(
            host=j["DB_HOST"],
            dbname=j["HAPPY_PATH_DB"],
            user=j["HAPPY_PATH_USER"],
            password=j["HAPPY_PATH_PASSWORD"]
        )
    else:
        pg_conn = psycopg2.connect(
            host=j["DB_HOST"],
            dbname=j["DB_NAME"],
            user=j["DB_USER"],
            password=j["DB_PASSWORD"]
        )

    # Connect to the enrichment DB
    enrich_conn = psycopg2.connect(
        host=j["ENRICH_DB_HOST"],
        dbname=j["ENRICH_DB"],
        user=j["ENRICH_DB_USER"],
        password=j["ENRICH_DB_PASSWORD"]
    )

    # Load or initialize configuration
    config = load_enricher_config(enrich_conn)
    if not config:
        config = {"model_type": "RandomForest", "relevant_columns": {}}

    # Determine which tables to use:
    if args.table_name:
        tables = [args.table_name.strip()]
    else:
        tables = get_all_user_tables(pg_conn)
    if not tables:
        logging.error("No tables found; exiting.")
        pg_conn.close()
        enrich_conn.close()
        return

    # Determine global (relevant) columns:
    if not config.get("relevant_columns"):
        global_cols = detect_global_relevant_columns(pg_conn, tables, sample_size=10000)
        config["relevant_columns"] = global_cols
    else:
        global_cols = config["relevant_columns"]
        for col in FORCED_DOMAIN_COLUMNS:
            if col not in global_cols:
                global_cols.append(col)
        for tcol in ALWAYS_TEXT_COLS:
            if tcol not in global_cols:
                global_cols.append(tcol)
        config["relevant_columns"] = global_cols

    logging.info(f"[enricher_bert] Using {len(global_cols)} relevant columns => {global_cols}")

    # Gather training data from all tables
    all_train_rows = gather_global_training_data(pg_conn, tables, global_cols)

    if args.mode == "train":
        # In train mode, always build (or rebuild) the model regardless of table-specific enrichment
        if not all_train_rows:
            logging.warning("No training data found; aborting model training.")
        else:
            trained_model = build_global_model(all_train_rows, global_cols, config)
            if trained_model:
                store_enricher_model(enrich_conn, trained_model)
                logging.info("Model training completed and stored.")
            else:
                logging.warning("Model training failed.")
        store_enricher_config(enrich_conn, config)
        pg_conn.close()
        enrich_conn.close()
        return  # Exit after training

    else:
        # In enrich mode, try to load an existing model first
        trained_model = load_enricher_model(enrich_conn)
        if trained_model is None:
            logging.info("[enricher_bert] No stored enricher model found. Building a new one...")
            if not all_train_rows:
                logging.warning("No training data found; skipping model building.")
                store_enricher_config(enrich_conn, config)
                pg_conn.close()
                enrich_conn.close()
                return
            trained_model = build_global_model(all_train_rows, global_cols, config)
            if not trained_model:
                logging.warning("No model built; skipping enrichment.")
                store_enricher_config(enrich_conn, config)
                pg_conn.close()
                enrich_conn.close()
                return
            store_enricher_model(enrich_conn, trained_model)
        else:
            logging.info("[enricher_bert] Using previously stored enricher model.")

        store_enricher_config(enrich_conn, config)

        # Proceed with enrichment for each table:
        for t in tables:
            logging.info(f"Enriching table: {t}")
            try:
                enrich_table(pg_conn, t, trained_model, config["relevant_columns"])
                analyze_significance(pg_conn, t)
            except psycopg2.Error as e:
                logging.error(f"Error processing table {t}: {e}")
                pg_conn.rollback()

        pg_conn.close()
        enrich_conn.close()
        logging.info("✅ enricher_bert: Global enrichment complete.")

        # Optionally, run anomaly gathering (if applicable)

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
