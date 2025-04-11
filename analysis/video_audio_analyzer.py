#!/usr/bin/env python3
"""
video_audio_analyzer.py

Refactored to:
1) Analyze logs from table(s) matching R[0-9]{10}.
2) Group logs by sessions delimited by lines containing "CHANGE_CONTENT".
3) Extract for each session:
   - CHANGE_CONTENT timestamp
   - svc_name and video_resolution (if present)
   - Alerts:
       • Count occurrences of BlackScreen events
       • Count occurrences of VideoFreeze events (both via keyword and via repeated videoPos)
       • Count Display Drop events categorized by value ranges (0-99, 100-299, 300-499, 500-699, 700+)
   - Additional events: tuning errors, standby/dream transitions, HDMI events, trick modes.
4) For each session, store a row in the chart DB table "video_audio_analysis"
   with a JSONB field containing the per-session analysis, specifically labeling black screen
   and video freeze events in the final JSON.
"""

import argparse
import json
import logging
import os
import re
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

# Set up basic logging (debug level)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


###############################################################################
# 1. Database Connection Helpers
###############################################################################

def connect_to_db():
    """
    Connect to the logs DB (containing the R-tables).
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    creds_path = os.path.join(base_dir, "credentials.txt")
    if not os.path.exists(creds_path):
        logging.error("credentials.txt not found at %s", creds_path)
        return None
    try:
        with open(creds_path, "r") as f:
            creds = json.load(f)
        conn = psycopg2.connect(
            host=creds["DB_HOST"],
            database=creds["DB_NAME"],
            user=creds["DB_USER"],
            password=creds["DB_PASSWORD"]
        )
        logging.debug("Connected to logs DB")
        return conn
    except Exception as e:
        logging.error("Logs DB connection failed: %s", e)
        return None


def connect_to_chart_db():
    """
    Connect to the chart DB (e.g. the 3090 DB) where we store analysis in
    `video_audio_analysis`.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    creds_path = os.path.join(base_dir, "credentials.txt")
    if not os.path.exists(creds_path):
        logging.error("credentials.txt not found at %s", creds_path)
        return None
    try:
        with open(creds_path, "r") as f:
            creds = json.load(f)
        conn = psycopg2.connect(
            host=creds["3090_HOST"],
            database=creds["3090_db"],
            user=creds["3090_USER"],
            password=creds["3090_PASSWORD"]
        )
        logging.debug("Connected to chart DB")
        return conn
    except Exception as e:
        logging.error("Chart DB connection failed: %s", e)
        return None


def ensure_analysis_table(chart_conn):
    """
    Ensure the table `video_audio_analysis` exists with an 'id' column
    and a primary key on 'id'.
    """
    with chart_conn.cursor() as cur:
        # 1) Create table if it doesn't exist.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_audio_analysis (
                table_name TEXT,
                analysis_json JSONB,
                analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # 2) Add 'id' column if missing.
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name='video_audio_analysis'
                      AND column_name='id'
                )
                THEN
                    ALTER TABLE video_audio_analysis
                    ADD COLUMN id SERIAL;
                END IF;
            END
            $$;
        """)
        # 3) Ensure primary key on 'id'
        cur.execute("""
            SELECT COUNT(*) 
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.table_name = 'video_audio_analysis'
              AND tc.constraint_type = 'PRIMARY KEY'
              AND kcu.column_name = 'id';
        """)
        pk_count = cur.fetchone()[0]
        if pk_count == 0:
            cur.execute("""
                ALTER TABLE video_audio_analysis
                ADD CONSTRAINT video_audio_analysis_id_pkey
                PRIMARY KEY (id);
            """)
        chart_conn.commit()
        logging.debug("Ensured video_audio_analysis table exists with primary key.")


###############################################################################
# 2. Log Fetching & Table Discovery
###############################################################################

def fetch_all_r_tables(conn):
    """
    Return a list of table names matching the pattern R[0-9]{10}.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tablename
            FROM pg_catalog.pg_tables
            WHERE schemaname = 'public'
              AND tablename ~ '^R[0-9]{10}$'
        """)
        rows = cur.fetchall()
    table_list = [r[0] for r in rows]
    logging.debug("Found R-tables: %s", table_list)
    return table_list


def fetch_logs(conn, table_name, start_time=None, end_time=None):
    query = f'SELECT timestamp, data FROM "{table_name}"'
    filters = []
    if start_time:
        filters.append(f'"timestamp" >= \'{start_time}\'')
    if end_time:
        filters.append(f'"timestamp" <= \'{end_time}\'')
    if filters:
        query += " WHERE " + " AND ".join(filters)
    query += " ORDER BY timestamp ASC"
    logging.debug("Executing query: %s", query)
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query)
        logs = cur.fetchall()
    logging.debug("Fetched %d log rows from %s", len(logs), table_name)
    return logs


###############################################################################
# 3. Helpers for Extracting Data from Log Lines
###############################################################################

def extract_svc_name(line_text):
    match = re.search(r"svc_name\s*=\s*([\w\-]+)", line_text, re.IGNORECASE)
    return match.group(1) if match else None


def extract_video_resolution(line_text):
    match = re.search(r"video\s+resolution[:\s]+(\d{3,4}x\d{3,4})", line_text, re.IGNORECASE)
    return match.group(1) if match else None


def parse_display_drop_value(line_text):
    match = re.search(r"Display Drop Detected\s*=\s*(\d+)", line_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def parse_trick_mode(line_text):
    m = re.search(r"(MOTION_[A-Z_0-9]+)\b", line_text)
    return m.group(1) if m else None


def parse_videoPos(line_text):
    """
    Extract the numeric video position from a line, e.g. "videoPos= 43766.466642"
    """
    m = re.search(r"videopos\s*=\s*([\d\.]+)", line_text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception as e:
            logging.debug("Error parsing videoPos: %s", e)
    return None


###############################################################################
# 4. Session Analysis
###############################################################################

def iso_to_datetime(iso_str):
    try:
        if '+' in iso_str:
            iso_str = re.sub(r'(\+\d{2})(\s|$)', r'\100 ', iso_str)
        return datetime.strptime(iso_str, "%Y-%m-%d %H:%M:%S.%f%z")
    except Exception as e:
        logging.debug("Failed to convert iso string to datetime: %s", e)
        return None


def time_diff_in_seconds(start_iso, end_iso):
    dt_start = iso_to_datetime(start_iso)
    dt_end = iso_to_datetime(end_iso)
    if dt_start and dt_end:
        return (dt_end - dt_start).total_seconds()
    return None


def analyze_var_logs(logs):
    sessions = []
    current_session = None

    # We'll use these variables to track consecutive videoPos values.
    last_videoPos = None
    videoPos_repeat_count = 0
    VIDEO_POS_THRESHOLD = 3  # # of consecutive repeats to consider a freeze

    def start_new_session(timestamp, line):
        return {
            "change_content_timestamp": timestamp,
            "svc_name": extract_svc_name(line),
            "video_resolution": extract_video_resolution(line),
            # We'll track black screen & freeze events in separate lists:
            "black_screen_events": [],
            "video_freeze_events": [],
            "alerts": {
                "BlackScreen": {"count": 0, "first_ts": None, "last_ts": None},
                "VideoFreeze": {"count": 0, "first_ts": None, "last_ts": None},
                "DisplayDrop": {
                    "0_99": {"count": 0, "first_ts": None, "last_ts": None},
                    "100_299": {"count": 0, "first_ts": None, "last_ts": None},
                    "300_499": {"count": 0, "first_ts": None, "last_ts": None},
                    "500_699": {"count": 0, "first_ts": None, "last_ts": None},
                    "700_plus": {"count": 0, "first_ts": None, "last_ts": None}
                }
            },
            "other_events": {
                "tuning_errors": [],
                "standby_dream": [],
                "hdmi_events": [],
                "trick_modes": []
            }
        }

    def update_alert(alert_dict, timestamp):
        """Increment the count and set first/last timestamps."""
        alert_dict["count"] += 1
        if alert_dict["first_ts"] is None:
            alert_dict["first_ts"] = timestamp
        alert_dict["last_ts"] = timestamp

    for row in logs:
        line_text = (row["data"] or "").strip()
        lower_line = line_text.lower()
        ts_str = row["timestamp"]
        logging.debug("Processing line at %s: %s", ts_str, line_text)

        # New session boundary on "CHANGE_CONTENT"
        if "change_content" in lower_line:
            if current_session is not None:
                sessions.append(current_session)
            current_session = start_new_session(ts_str, line_text)
            # Reset videoPos tracking
            last_videoPos = None
            videoPos_repeat_count = 0
            continue

        if current_session is None:
            current_session = start_new_session(ts_str, line_text)
            last_videoPos = None
            videoPos_repeat_count = 0

        # Update video resolution if not set yet.
        if not current_session["video_resolution"]:
            maybe_res = extract_video_resolution(line_text)
            if maybe_res:
                current_session["video_resolution"] = maybe_res

        # Check for BlackScreen events.
        if ("blackscreen" in lower_line or "black screen" in lower_line or "black-screen" in lower_line):
            update_alert(current_session["alerts"]["BlackScreen"], ts_str)
            # Also append a labeled event in the final JSON:
            current_session["black_screen_events"].append({
                "ts": ts_str,
                "line": line_text
            })
            logging.debug("BlackScreen event recorded at %s", ts_str)

        # Check for VideoFreeze events by keywords.
        if ("video freeze" in lower_line or "video-freeze" in lower_line or "frozen screen" in lower_line):
            update_alert(current_session["alerts"]["VideoFreeze"], ts_str)
            current_session["video_freeze_events"].append({
                "ts": ts_str,
                "line": line_text
            })
            logging.debug("VideoFreeze event (keyword) recorded at %s", ts_str)

        # Additionally, check for repeated videoPos values => freeze detection.
        if "videopos=" in lower_line:
            pos = parse_videoPos(line_text)
            if pos is not None:
                if last_videoPos is not None and abs(pos - last_videoPos) < 0.001:
                    videoPos_repeat_count += 1
                    logging.debug("Repeated videoPos detected (%f), count=%d", pos, videoPos_repeat_count)
                    if videoPos_repeat_count >= VIDEO_POS_THRESHOLD:
                        # Mark a video freeze event if not already recorded at this timestamp.
                        update_alert(current_session["alerts"]["VideoFreeze"], ts_str)
                        current_session["video_freeze_events"].append({
                            "ts": ts_str,
                            "line": line_text,
                            "reason": "repeated videoPos"
                        })
                        logging.debug("VideoFreeze event (repeated videoPos) recorded at %s", ts_str)
                        # Reset counter to avoid multiple consecutive triggers.
                        videoPos_repeat_count = 0
                else:
                    last_videoPos = pos
                    videoPos_repeat_count = 0

        # Check for Display Drop events.
        if ("display drop detected" in lower_line or "display-drop detected" in lower_line):
            drop_val = parse_display_drop_value(line_text)
            if drop_val < 100:
                update_alert(current_session["alerts"]["DisplayDrop"]["0_99"], ts_str)
            elif drop_val < 300:
                update_alert(current_session["alerts"]["DisplayDrop"]["100_299"], ts_str)
            elif drop_val < 500:
                update_alert(current_session["alerts"]["DisplayDrop"]["300_499"], ts_str)
            elif drop_val < 700:
                update_alert(current_session["alerts"]["DisplayDrop"]["500_699"], ts_str)
            else:
                update_alert(current_session["alerts"]["DisplayDrop"]["700_plus"], ts_str)
            logging.debug("DisplayDrop alert updated with value %d at %s", drop_val, ts_str)

        # Tuning errors/timeouts.
        if any(kw in lower_line for kw in ["ontunestart", "default tune took", "defaulttune"]):
            current_session["other_events"]["tuning_errors"].append({"ts": ts_str, "line": line_text})
            logging.debug("Tuning error recorded at %s", ts_str)

        # Standby/dream transitions.
        if any(kw in lower_line for kw in ["enter day dream", "exit day dream", "enterstandby"]):
            current_session["other_events"]["standby_dream"].append({"ts": ts_str, "line": line_text})
            logging.debug("Standby/dream event recorded at %s", ts_str)

        # HDMI events.
        if "hdmi" in lower_line:
            current_session["other_events"]["hdmi_events"].append({"ts": ts_str, "line": line_text})
            logging.debug("HDMI event recorded at %s", ts_str)

        # Trick modes.
        tm = parse_trick_mode(line_text)
        if tm and ("MOTION_SKIP_LIVE" not in tm):
            current_session["other_events"]["trick_modes"].append({"ts": ts_str, "mode": tm, "line": line_text})
            logging.debug("Trick mode %s recorded at %s", tm, ts_str)

    if current_session is not None:
        sessions.append(current_session)
        logging.debug("Final session finalized at %s", current_session["change_content_timestamp"])

    logging.info("Total sessions analyzed: %d", len(sessions))
    return sessions


###############################################################################
# 5. Storing Each Session in video_audio_analysis
###############################################################################

def store_sessions_in_db(table_name, sessions):
    """
    Insert each session as a separate row in video_audio_analysis,
    with a JSON representation of the session’s analysis.
    Duplicate sessions (with the same change_content_timestamp) are skipped.
    """
    chart_conn = connect_to_chart_db()
    if not chart_conn:
        logging.error("Could not connect to chart DB for storing analysis.")
        return
    ensure_analysis_table(chart_conn)

    insert_sql = """
    INSERT INTO video_audio_analysis (table_name, analysis_json)
    VALUES (%s, %s::jsonb);
    """
    with chart_conn.cursor() as cur:
        for sess in sessions:
            change_ts = sess.get("change_content_timestamp")
            change_ts_text = str(change_ts) if change_ts is not None else None
            # Avoid duplicates by checking if we already stored this session
            cur.execute("""
                SELECT COUNT(*) FROM video_audio_analysis
                WHERE table_name = %s
                  AND analysis_json->>'change_content_timestamp' = %s;
            """, (table_name, change_ts_text))
            count = cur.fetchone()[0]
            if count == 0:
                cur.execute(insert_sql, (table_name, json.dumps(sess, indent=2, default=str)))
                logging.info("Stored session with timestamp %s", change_ts_text)
            else:
                logging.info("Skipping duplicate session with timestamp %s", change_ts_text)
    chart_conn.commit()
    chart_conn.close()


###############################################################################
# 6. Flask Endpoint for Display
###############################################################################

@app.route("/")
def index():
    chart_conn = connect_to_chart_db()
    if not chart_conn:
        return "Error connecting to chart DB.", 500
    with chart_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            "SELECT table_name, analysis_json, analysis_time FROM video_audio_analysis ORDER BY analysis_time DESC")
        rows = cur.fetchall()
    chart_conn.close()
    return f"""
    <html>
      <head><title>Video Audio Analysis Results</title></head>
      <body>
        <h1>Video/Audio Analysis Results</h1>
        <pre>{json.dumps(rows, indent=2, default=str)}</pre>
      </body>
    </html>
    """


###############################################################################
# MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Analyze logs for VAR events (CHANGE_CONTENT, etc.).")
    parser.add_argument("-t", "--table_name", type=str, default="all",
                        help="Name of an R+10 table to analyze, or 'all' for all matching tables.")
    parser.add_argument("-st", "--start_time", type=str, default=None,
                        help="Start timestamp for logs filter (ISO format).")
    parser.add_argument("-et", "--end_time", type=str, default=None,
                        help="End timestamp for logs filter (ISO format).")
    args = parser.parse_args()

    conn = connect_to_db()
    if not conn:
        logging.error("Could not connect to logs DB.")
        return

    if args.table_name.lower() == "all":
        table_names = fetch_all_r_tables(conn)
    else:
        table_names = [args.table_name]

    for tbl in table_names:
        logs = fetch_logs(conn, tbl, start_time=args.start_time, end_time=args.end_time)
        if not logs:
            logging.warning("No logs found in %s for the specified timeframe.", tbl)
            continue
        sessions = analyze_var_logs(logs)
        if not sessions:
            logging.warning("No sessions found in %s (no CHANGE_CONTENT lines?).", tbl)
            continue
        logging.info("Table %s: Found %d sessions.", tbl, len(sessions))
        store_sessions_in_db(tbl, sessions)

    conn.close()
    logging.info("Done processing tables.")


if __name__ == "__main__":
    main()
