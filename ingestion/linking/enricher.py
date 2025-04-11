#!/usr/bin/env python3
"""
enricher.py

Refactored to always use a single, global ML model for enrichment.
Enhancements:
1. If parse_data_fields() detects "customer_marked_flag=YES", automatically sets event_type="MARKED_LOGS".
2. highlight_customer_marked() is called to boost significance for marked logs.
3. A single global model is trained from all R+10 tables (from the main or HAPPY_PATH DB) to ensure consistency.
4. For records whose event_type contains keywords ('ipll', 'ipvod', or 'llot'), appends ", signal_protector".
5. NEW for VAR: Additional fields (svc_name, video_resolution, connection_status) are extracted and forced into the feature set.
6. The mutual information selection now uses an increased sample size.
7. Uses TfidfVectorizer instead of CountVectorizer for more nuanced feature extraction.
"""

import re
import os
import json
import argparse
import psycopg2
from psycopg2 import sql, Error
import subprocess
from psycopg2.extras import DictCursor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import mutual_info_score
    sklearn_installed = True
except ImportError:
    sklearn_installed = False
    logging.warning("scikit-learn not installed. ML-based classification won't run automatically.")


###############################################################################
# 1) Utility: Fetch All User Tables
###############################################################################
def get_all_user_tables(pg_conn):
    """
    Returns a list of all R+10 tables in the DB.
    If the DB is happy_path, missing tables are skipped.
    """
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


###############################################################################
# 2) Regex-based Data Extraction & New VAR Field Extractors
###############################################################################
INTERFACE_STATUS_REGEX = re.compile(r'Interface (\w+) (Connected|Not Connected)\(?(-?\d+)?\)?')
AUTO_IP_THREAD_REGEX   = re.compile(r'Need to start Auto IP thread for (\w+), thread \(([^)]+)\), event (-?\d+)')
ARPING_REGEX           = re.compile(r'Arping for (\d+\.\d+\.\d+\.\d+)')
PACKET_INFO_REGEX      = re.compile(r'(Destin\.|Source) MAC: ([0-9A-Fa-f:]+)')
ARP_DETAILS_REGEX      = re.compile(r'(Sender|Target) (IP|MAC)\s*:\s*([0-9A-Fa-f:.]+)')
SGS_MSG_REGEX          = re.compile(r'SGS Msg <(\d+):(\w+)> took <(\d+) ms> SGS Return Code:<(\d+):(\w+)> rx_id:<([\w-]+)>')
MAC_SELECT_THREAD_REGEX= re.compile(r'MAC Select thread running on <(\w+)> interface')
WORKER_THREAD_REGEX    = re.compile(r'starting thread:<(\w+ thread)>, version:<(\d+)>')

MALLOC_OCCUPIED_REGEX  = re.compile(r"This is the total size of memory occupied by chunks handed out by malloc:(\d+)")
FREE_CHUNKS_REGEX      = re.compile(r"This is the total size of memory occupied by free \(not in use\) chunks\.:(\d+)")
TOPMOST_CHUNK_REGEX    = re.compile(r"This is the size of the top-most releasable chunk.*?:(\d+)")
FREED_UP_BYTES_REGEX   = re.compile(r"Freed up bytes:\s*(\d+)")
USED_MEMORY_BEFORE_REGEX = re.compile(r"Used memory before GC:\s*(\d+)")
USED_MEMORY_AFTER_REGEX  = re.compile(r"Used memory after GC:\s*(\d+)")
ALLOCATED_MMAP_REGEX     = re.compile(r"Allocated\s*(\d+)\s*bytes\s*in\s*(\d+)\s*chunks\.\s*\(via mmap\)")
ALLOCATED_MALLOC_REGEX   = re.compile(r"Allocated\s*(\d+)\s*bytes\s.*\(via malloc\)")

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

    # NEW: Extract additional VAR fields for enrichment
    svc_name = extract_svc_name(data_str)
    if svc_name:
        extracted["svc_name"] = svc_name
    video_res = extract_video_resolution(data_str)
    if video_res:
        extracted["video_resolution"] = video_res

    return extracted


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
# 4) Table Column Management
###############################################################################
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

def add_col_if_missing(cur, schema_name, table_name, col_name, col_type="TEXT"):
    existing = get_table_columns(cur.connection, schema_name, table_name)
    if col_name in existing:
        return
    sql_add = f'ALTER TABLE "{schema_name}"."{table_name}" ADD COLUMN "{col_name}" {col_type}'
    cur.execute(sql_add)

def rename_and_migrate_column(cur, schema_part, table_part, old_name, new_name):
    old_lower = old_name.lower()
    new_lower = new_name.lower()
    if old_lower == new_lower:
        return
    existing = get_table_columns(cur.connection, schema_part, table_part)
    if new_name in existing:
        return
    logging.info(f"[REFORM] Renaming '{old_name}' => '{new_name}' in {schema_part}.{table_part}")
    try:
        cur.execute(sql.SQL('ALTER TABLE {}.{} ADD COLUMN {} TEXT').format(
            sql.Identifier(schema_part),
            sql.Identifier(table_part),
            sql.Identifier(new_name)
        ))
        copy_query = sql.SQL('UPDATE {}.{} SET {} = {} WHERE {} IS NOT NULL').format(
            sql.Identifier(schema_part),
            sql.Identifier(table_part),
            sql.Identifier(new_name),
            sql.Identifier(old_name),
            sql.Identifier(old_name)
        )
        cur.execute(copy_query)
        logging.info(f"  Copied data from '{old_name}' to '{new_name}'.")
        drop_query = sql.SQL('ALTER TABLE {}.{} DROP COLUMN {}').format(
            sql.Identifier(schema_part),
            sql.Identifier(table_part),
            sql.Identifier(old_name)
        )
        cur.execute(drop_query)
        logging.info(f"  Dropped old column '{old_name}'.")
    except Exception as e:
        logging.error(f"  ❌ Error renaming column {old_name} => {new_name}: {e}")
        cur.connection.rollback()

def add_col(cur, schema_part, table_part, col_name):
    existing_cols = get_table_columns(cur.connection, schema_part, table_part)
    if col_name in existing_cols:
        return
    col_type = "BIGINT" if col_name.endswith("_num") else "TEXT"
    logging.info(f"[REFORM] Adding missing column '{col_name}' ({col_type}) in {schema_part}.{table_part}")
    cur.execute(sql.SQL('ALTER TABLE {}.{} ADD COLUMN {} ' + col_type).format(
        sql.Identifier(schema_part),
        sql.Identifier(table_part),
        sql.Identifier(col_name)
    ))

def refactor_table_for_enrichment(pg_conn, table_name, columns_needed):
    schema_part, table_part = parse_schema_and_table(table_name)
    with pg_conn.cursor() as cur:
        existing_cols = get_table_columns(pg_conn, schema_part, table_part)
        existing_lower_map = {k.lower(): k for k in existing_cols.keys()}
        if "event_type" not in columns_needed:
            columns_needed.append("event_type")
        # Ensure new VAR fields are included
        for new_field in ["svc_name", "video_resolution", "connection_status"]:
            if new_field not in columns_needed:
                columns_needed.append(new_field)
        for c in columns_needed:
            c_lower = c.lower()
            if c_lower in existing_lower_map:
                real_name = existing_lower_map[c_lower]
                if real_name != c:
                    rename_and_migrate_column(cur, schema_part, table_part, real_name, c)
            else:
                add_col(cur, schema_part, table_part, c)
    pg_conn.commit()


###############################################################################
# 5) Detecting "Relevant Columns" from Tables (Increased Sample Size)
###############################################################################
def sample_table_rows(pg_conn, schema_part, table_part, sample_size, col_list):
    col_select = ", ".join(f'"{c}"' for c in col_list)
    q = f'''
        SELECT {col_select}
        FROM "{schema_part}"."{table_part}"
        ORDER BY random()
        LIMIT {sample_size}
    '''
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        try:
            cur.execute(q)
            return cur.fetchall()
        except Exception as e:
            logging.warning(f"Sampling {schema_part}.{table_part} failed: {e}")
            return []

def detect_relevant_columns_table(pg_conn, table_name, sample_size=2000):  # increased sample size
    schema_part, table_part = parse_schema_and_table(table_name)
    try:
        col_info = get_table_columns(pg_conn, schema_part, table_part)
    except psycopg2.Error as e:
        logging.warning(f"Table {table_name} does not exist; skipping. Error: {e}")
        pg_conn.rollback()
        return []
    text_columns = [c for c, dtype in col_info.items() if 'text' in dtype or 'char' in dtype]
    if 'event_type' not in text_columns:
        text_columns.append('event_type')

    sample_rows = sample_table_rows(pg_conn, schema_part, table_part, sample_size, text_columns)
    if not sample_rows:
        return list(set(text_columns))

    from sklearn.metrics import mutual_info_score
    event_labels = [(r['event_type'] or 'UNKNOWN') for r in sample_rows]
    if not any(evt != 'UNKNOWN' for evt in event_labels):
        return text_columns

    scores = []
    for c in text_columns:
        col_vals = [row.get(c) or 'NULL' for row in sample_rows]
        s = mutual_info_score(col_vals, event_labels)
        scores.append((c, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    top3 = [x[0] for x in scores[:3]]
    if 'event_type' not in top3:
        top3.append('event_type')
    return list(set(top3))

def detect_global_relevant_columns(pg_conn, tables, sample_size=10000):
    global_cols = set()
    for t in tables:
        top3 = detect_relevant_columns_table(pg_conn, t, sample_size)
        global_cols.update(top3)
    return list(global_cols)

def compute_global_mi_for_columns(all_data_rows, relevant_cols, min_count=1):
    valid_rows = [r for r in all_data_rows if r.get('event_type')]
    if not valid_rows:
        return {}
    column_scores = {}
    event_labels = [r['event_type'] for r in valid_rows]
    for col in relevant_cols:
        col_vals = [str(r.get(col, 'NULL'))[:10000] for r in valid_rows]
        if len(set(col_vals)) <= min_count:
            column_scores[col] = 0.0
            continue
        score = mutual_info_score(col_vals, event_labels)
        column_scores[col] = score
    return column_scores

def refine_columns_by_mi(config, column_scores, min_mi=0.0001, max_cols=128, stability_threshold=10):
    if 'column_score_history' not in config:
        config['column_score_history'] = {}
    history = config['column_score_history']
    stable_cols = []
    for col, score in column_scores.items():
        if score < min_mi:
            history[col] = history.get(col, 0) + 1
        else:
            history[col] = 0
        if score >= min_mi or history[col] < stability_threshold:
            stable_cols.append(col)
    if 'event_type' not in stable_cols:
        stable_cols.append('event_type')
    if 'master_columns' not in config:
        config['master_columns'] = set(stable_cols)
    else:
        config['master_columns'] = set(config['master_columns']).union(set(stable_cols))
    stable_cols = sorted(stable_cols, key=lambda c: column_scores.get(c, 0), reverse=True)
    new_cols = stable_cols[:max_cols]
    config['column_score_history'] = history
    config['master_columns'] = list(config['master_columns'])
    return new_cols


###############################################################################
# 6) Gathering Training Data => Single Global Model
###############################################################################
def gather_global_training_data(pg_conn, tables, relevant_cols, require_event_type=True):
    """
    Always gather from all R+10 tables for a single global model,
    skipping any that don't exist.
    """
    all_rows = []
    for t in tables:
        try:
            schema_part, table_part = parse_schema_and_table(t)
            actual_cols = list(get_table_columns(pg_conn, schema_part, table_part).keys())
        except psycopg2.Error as e:
            logging.warning(f"Table {t} does not exist; skipping. Error: {e}")
            pg_conn.rollback()
            continue
        selected_cols = [c for c in relevant_cols if c in actual_cols]
        if not selected_cols:
            logging.warning(f"Table {t} does not have any of the relevant columns; skipping.")
            continue
        condition = ''
        if require_event_type:
            condition = 'WHERE "event_type" IS NOT NULL AND "event_type" <> \'\''
        col_select = ', '.join(f'"{c}"' for c in selected_cols)
        q = f'''
            SELECT {col_select}
            FROM "{schema_part}"."{table_part}"
            {condition}
        '''
        with pg_conn.cursor(cursor_factory=DictCursor) as cur:
            try:
                cur.execute(q)
                rows = cur.fetchall()
                all_rows.extend(rows)
            except Exception as ex:
                logging.warning(f"Gather data: table {t} => {ex}")
                pg_conn.rollback()
                continue
    return all_rows

def build_global_model(all_data_rows, relevant_cols, config):
    """
    Builds a single global classifier from the entire DB's data.
    Uses TfidfVectorizer and explicitly includes key parsed fields.
    """
    if not all_data_rows:
        logging.info("No global training data found. No model built.")
        return None, None

    from sklearn.feature_extraction.text import TfidfVectorizer

    X_texts = []
    Y_labels = []
    # Explicitly include extra features: svc_name, video_resolution, connection_status.
    for row in all_data_rows:
        # Build feature text from the selected columns
        #base_text = " ".join((row.get(c) or '') for c in relevant_cols)
        base_text = " ".join(str(row.get(c)) if row.get(c) is not None else '' for c in relevant_cols)

        extras = []
        for field in ["svc_name", "video_resolution", "connection_status"]:
            val = row.get(field)
            if val:
                extras.append(str(val))
        combined_text = " ".join([base_text] + extras)
        X_texts.append(combined_text)
        label = row.get('event_type')
        if label is None or (isinstance(label, str) and label.strip() == ""):
            label = row.get('category')
            if label is None or (isinstance(label, str) and label.strip() == ""):
               label = "UNKNOWN"
        Y_labels.append(str(label))

    # Use TF-IDF for richer representations
    vec = TfidfVectorizer(max_features=10000)
    X = vec.fit_transform(X_texts)

    model_type = config.get('model_type', 'RandomForest')
    if model_type == 'NaiveBayes':
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB()
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

    clf.fit(X, Y_labels)
    logging.info(f"Trained single global {model_type} model on {len(X_texts)} rows total.")
    if hasattr(clf, 'feature_importances_'):
        import numpy as np
        importances = clf.feature_importances_
        feature_names = vec.get_feature_names_out()
        indices = np.argsort(importances)[::-1]
        topN = min(20, len(indices))
        logging.info("Top feature importances:")
        for idx in indices[:topN]:
            logging.info(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    return vec, clf


###############################################################################
# 7) Enrich a Table using the Global Model
###############################################################################
def predict_event_type(clf, vec, text):
    if not text.strip() or (clf is None or vec is None):
        return 'UNKNOWN'
    X = vec.transform([text])
    y = clf.predict(X)
    return y[0] if len(y) else 'UNKNOWN'

def update_signal_protector(pg_conn, schema_part, table_part):
    cols = get_table_columns(pg_conn, schema_part, table_part)
    text_cols = [col for col, dtype in cols.items() if 'char' in dtype or 'text' in dtype]
    if not text_cols:
        logging.info(f"No text columns found in {schema_part}.{table_part} for signal protector update.")
        return

    conditions = []
    for col in text_cols:
        conditions.append(f'("{col}" ILIKE \'%ipll%\' OR "{col}" ILIKE \'%ipvod%\' OR "{col}" ILIKE \'%llot%\')')
    condition_clause = " OR ".join(conditions)
    update_sql = f'''
        UPDATE "{schema_part}"."{table_part}"
        SET "event_type" =
            CASE 
                WHEN "event_type" ILIKE '%signal_protector%' THEN "event_type"
                ELSE COALESCE("event_type", '') || 
                     CASE WHEN "event_type" IS NULL OR "event_type" = '' THEN '' ELSE ', ' END || 
                     'signal_protector'
            END
        WHERE {condition_clause};
    '''
    with pg_conn.cursor() as cur:
        cur.execute(update_sql)
    pg_conn.commit()
    logging.info(f"Updated records with signal protector keywords in {schema_part}.{table_part}.")

def enrich_table(pg_conn, table_name, vectorizer, clf, relevant_cols):
    schema_part, table_part = parse_schema_and_table(table_name)
    refactor_table_for_enrichment(pg_conn, table_name, list(relevant_cols))
    # Ensure VAR fields are present
    with pg_conn.cursor() as cur:
        add_col_if_missing(cur, schema_part, table_part, "svc_name")
        add_col_if_missing(cur, schema_part, table_part, "video_resolution")
    pg_conn.commit()

    needed_cols = set(relevant_cols) | {'id', 'data', 'event_type', 'timestamp', 'category', 'svc_name', 'video_resolution'}
    col_select = ','.join(f'"{c}"' for c in needed_cols)
    q = f'SELECT {col_select} FROM "{schema_part}"."{table_part}"'
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(q)
        rows = cur.fetchall()

    if not rows:
        logging.info(f"Table '{table_name}' has no rows, skipping enrichment.")
        return

    updates_pred = []
    updates_parse = []
    updates_marked_logs = []

    for r in rows:
        row_id = r['id']
        if not r.get('event_type'):
            text_combined = " ".join((r.get(c) or '') for c in relevant_cols)
            pred = predict_event_type(clf, vectorizer, text_combined)
            if pred != 'UNKNOWN':
                updates_pred.append((pred, row_id))
        extracted = parse_data_fields(r.get('data', ''))
        if extracted:
            updates_parse.append((row_id, extracted))
            if extracted.get('customer_marked_flag') == "YES":
                updates_marked_logs.append((row_id, "MARKED_LOGS"))
            if extracted.get('svc_name'):
                updates_parse.append((row_id, {"svc_name": extracted.get('svc_name')}))
            if extracted.get('video_resolution'):
                updates_parse.append((row_id, {"video_resolution": extracted.get('video_resolution')}))

    if updates_parse:
        new_cols = set()
        for _, exdict in updates_parse:
            new_cols.update(exdict.keys())
        if new_cols:
            with pg_conn.cursor() as cur_cols:
                for nc in new_cols:
                    add_col(cur_cols, schema_part, table_part, nc)
            pg_conn.commit()

    if updates_parse:
        with pg_conn.cursor() as curu:
            for row_id, exdict in updates_parse:
                set_parts = []
                vals = []
                for k, v in exdict.items():
                    set_parts.append(f'"{k}"=%s')
                    vals.append(v)
                sets_clause = ",".join(set_parts)
                sql_up = f'UPDATE "{schema_part}"."{table_part}" SET {sets_clause} WHERE "id"=%s'
                vals.append(row_id)
                curu.execute(sql_up, vals)
        pg_conn.commit()
        logging.info(f"{len(updates_parse)} rows updated with parsed data fields in '{table_name}'.")

    if updates_marked_logs:
        with pg_conn.cursor() as curp:
            for (row_id, forced_label) in updates_marked_logs:
                curp.execute(f'''
                  UPDATE "{schema_part}"."{table_part}"
                  SET "event_type"=%s
                  WHERE "id"=%s
                ''', (forced_label, row_id))
        pg_conn.commit()
        logging.info(f"{len(updates_marked_logs)} rows labeled as 'MARKED_LOGS' in '{table_name}'.")

    if updates_pred:
        with pg_conn.cursor() as curp:
            up_sql = sql.SQL('UPDATE {}.{} SET "event_type"=%s WHERE "id"=%s').format(
                sql.Identifier(schema_part),
                sql.Identifier(table_part)
            )
            curp.executemany(up_sql, updates_pred)
        pg_conn.commit()
        logging.info(f"{len(updates_pred)} rows had event_type predicted in '{table_name}'.")

    update_signal_protector(pg_conn, schema_part, table_part)


###############################################################################
# 8) Significance Heuristic
###############################################################################
import numpy as np

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
        query = f'SELECT id, {", ".join(f"{col}" for col in numeric_cols)} FROM "{schema_part}"."{table_part}"'
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
    logging.info(f"[MARKED] {len(marked_rows)} row(s) marked for customer review in '{table_name}' with significance boost of +{significance_boost}.")


###############################################################################
# 9) Master Column List Updater
###############################################################################
def update_master_columns(config, detected_cols):
    if 'master_columns' not in config:
        config['master_columns'] = set(detected_cols)
    else:
        config['master_columns'] = set(config['master_columns']).union(set(detected_cols))
    config['master_columns'].add('event_type')
    config['master_columns'] = list(config['master_columns'])
    return config['master_columns']


###############################################################################
# 10) MAIN (Global Model Only)
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Global enricher: Uses a single global ML model for all logs."
    )
    parser.add_argument("--table_name", type=str,
        help="If set, only process that table; otherwise, process all tables.")
    parser.add_argument("--happy_path", action="store_true",
                        help="Use the HAPPY_PATH_DB instead of the main DB for logs.")
    args = parser.parse_args()

    if not sklearn_installed:
        logging.warning("scikit-learn not installed => no classification.")
        return

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    creds_path = os.path.join(base_dir, "credentials.txt")
    if not os.path.exists(creds_path):
        logging.error(f"credentials.txt not found: {creds_path}")
        return
    with open(creds_path, "r") as f:
        j = json.load(f)

    db_name = j.get("HAPPY_PATH_DB") if args.happy_path else j.get("DB_NAME")
    db_host = j.get("DB_HOST")
    db_user = j.get("DB_USER")
    db_pass = j.get("DB_PASSWORD")

    enrich_host = j.get("ENRICH_DB_HOST")
    enrich_db = j.get("ENRICH_DB")
    enrich_user = j.get("ENRICH_DB_USER")
    enrich_pw = j.get("ENRICH_DB_PASSWORD")

    try:
        pg_conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_pass)
    except Exception as e:
        logging.error(f"logs DB connect fail: {e}")
        return

    try:
        enrich_conn = psycopg2.connect(host=enrich_host, dbname=enrich_db, user=enrich_user, password=enrich_pw)
    except Exception as e:
        logging.error(f"enrich DB connect fail: {e}")
        pg_conn.close()
        return

    ensure_ml_config_table(enrich_conn)
    config = load_enricher_config(enrich_conn)
    if not config:
        config = {"model_type": "RandomForest", "relevant_columns": []}
    if 'column_score_history' not in config:
        config['column_score_history'] = {}

    # Global: gather all R+10 tables
    tables = get_all_user_tables(pg_conn)
    if not tables:
        logging.error("No R+10 tables found in logs DB. Exiting.")
        pg_conn.close()
        enrich_conn.close()
        return

    if not config.get('relevant_columns'):
        global_cols = detect_global_relevant_columns(pg_conn, tables)
        config['relevant_columns'] = global_cols
    else:
        global_cols = config['relevant_columns']
    logging.info(f"Using global relevant columns: {len(global_cols)} columns.")

    # Gather training data from all tables
    all_train_rows = gather_global_training_data(pg_conn, tables, global_cols, require_event_type=True)
    if not all_train_rows:
        logging.warning("No event_type-labeled data found globally; trying without filter.")
        all_train_rows = gather_global_training_data(pg_conn, tables, global_cols, require_event_type=False)

    vec, clf = build_global_model(all_train_rows, global_cols, config)
    if not vec or not clf:
        pg_conn.close()
        enrich_conn.close()
        logging.info("No global model built due to lack of data. Exiting.")
        return

    # Refine columns using mutual information (with new sample size and explicit features)
    col_scores = compute_global_mi_for_columns(all_train_rows, global_cols)
    new_global_cols = refine_columns_by_mi(config, col_scores, min_mi=0.0001, max_cols=128)
    logging.info(f"Refined columns from {len(global_cols)} to {len(new_global_cols)}.")
    config['relevant_columns'] = new_global_cols
    store_enricher_config(enrich_conn, config)

    # Enrich: if a single table is specified, only process that table; otherwise process all tables.
    if args.table_name:
        target_table = args.table_name.strip()
        logging.info(f"Enriching single table with global model: {target_table}")
        try:
            enrich_table(pg_conn, target_table, vec, clf, config['relevant_columns'])
            analyze_significance(pg_conn, target_table)
        except psycopg2.Error as e:
            logging.error(f"Error processing table {target_table}: {e}")
            pg_conn.rollback()
    else:
        logging.info("Enriching ALL R+10 tables with global model...")
        for t in tables:
            logging.info(f"Enriching table: {t}")
            try:
                enrich_table(pg_conn, t, vec, clf, config['relevant_columns'])
                analyze_significance(pg_conn, t)
            except psycopg2.Error as e:
                logging.error(f"Error processing table {t}: {e}")
                pg_conn.rollback()

    store_enricher_config(enrich_conn, config)
    pg_conn.close()
    enrich_conn.close()
    logging.info("Global enrichment complete (single global model). ✅")

    try:
        if args.table_name:
            logging.info(f"Gathering anomalies for single table {args.table_name} ...")
            subprocess.run(["python", "ingestion/linking/importance_collector.py",
                            "--table_name", args.table_name],
                            check=True)
        else:
            logging.info("Gathering anomalies from ALL R-tables.")
            subprocess.run(["python", "ingestion/linking/importance_collector.py"], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Global what_a_do failed: {e}")

    logging.info("Enrichment process finished successfully.")

if __name__ == "__main__":
    main()
