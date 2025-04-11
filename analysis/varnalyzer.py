#!/usr/bin/env python3
"""
varnalyzer.py

Refactored to parse lines from 'data' (fallback to file_line/message) for:
 - Video freeze events
 - CHANGE_CONTENT
 - BlackScreen
 - DisplayDrop
 - Tuning errors, daydream, HDMI, trick modes
Then store results in video_audio_analysis table (chart DB).
"""

import argparse
import json
import logging
import os
import re
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from flask import Flask

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

###############################################################################
# 1. DB Connections & Setup
###############################################################################

def connect_to_db():
    """Connect to the logs DB (the one with R+ tables)."""
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
        return conn
    except Exception as e:
        logging.error("Logs DB connection failed: %s", e)
        return None

def connect_to_chart_db():
    """Connect to the chart DB to store analysis in `video_audio_analysis`."""
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
        return conn
    except Exception as e:
        logging.error("Chart DB connection failed: %s", e)
        return None

def ensure_analysis_table(chart_conn):
    """Ensure that the video_audio_analysis table exists (with id primary key)."""
    with chart_conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_audio_analysis (
                table_name TEXT,
                analysis_json JSONB,
                analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Add 'id' column if missing
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
                    ADD COLUMN id SERIAL PRIMARY KEY;
                END IF;
            END
            $$;
        """)
        chart_conn.commit()

###############################################################################
# 2. Discover & Fetch Logs
###############################################################################

def fetch_all_r_tables(conn):
    """
    Returns a list of table names matching typical R tables.

    If you only want EXACT 10-digit R-tables, use:
        WHERE tablename ~ '^R[0-9]{10}$'

    If you also want ones with `_demo`, `_varapp`, etc., use:
        WHERE tablename ~ '^R[0-9]{10}(_.*)?$'
    """
    with conn.cursor() as cur:
        # Toggle whichever pattern you need:
        # cur.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public' AND tablename ~ '^R[0-9]{10}$'")
        cur.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public' AND tablename ~ '^R[0-9]{10}(_.*)?$'")
        rows = cur.fetchall()
    return [r[0] for r in rows]

def fetch_logs(conn, table_name, start_time=None, end_time=None):
    """Fetch from the given table, ordering by timestamp."""
    query = f'SELECT "timestamp", "data", "file_line", "message" FROM "{table_name}"'
    filters = []
    if start_time:
        filters.append(f'"timestamp" >= %s')
    if end_time:
        filters.append(f'"timestamp" <= %s')
    if filters:
        query += " WHERE " + " AND ".join(filters)
    query += " ORDER BY timestamp ASC"

    with conn.cursor(cursor_factory=DictCursor) as cur:
        if start_time and end_time:
            cur.execute(query, (start_time, end_time))
        elif start_time:
            cur.execute(query, (start_time,))
        elif end_time:
            cur.execute(query, (end_time,))
        else:
            cur.execute(query)
        rows = cur.fetchall()
    logging.info("Fetched %d log rows from %s", len(rows), table_name)
    return rows

###############################################################################
# 3. Analysis Logic
###############################################################################

def parse_line_text(row):
    """
    Return the best “log text” from row for analyzing freeze keywords, etc.
    Priority: data > file_line > message.
    """
    line_text = row["data"]
    if line_text:
        return line_text.strip()
    # fallback
    if row["file_line"]:
        return row["file_line"].strip()
    if row["message"]:
        return row["message"].strip()
    return ""

def extract_svc_name(line_text):
    match = re.search(r"svc_name\s*=\s*([\w\-]+)", line_text, re.IGNORECASE)
    return match.group(1) if match else None

def extract_video_resolution(line_text):
    match = re.search(r"(\d{3,4}x\d{3,4})", line_text, re.IGNORECASE)
    return match.group(1) if match else None

def parse_display_drop_value(line_text):
    match = re.search(r"Display Drop Detected\s*=\s*(\d+)", line_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def parse_trick_mode(line_text):
    # e.g. "MOTION_PLAY", "MOTION_FF_15"
    m = re.search(r"(MOTION_[A-Z0-9_]+)\b", line_text)
    return m.group(1) if m else None

def analyze_logs(rows):
    """
    Build session objects keyed by "CHANGE_CONTENT" lines.
    Collect freeze, blackscreen, display drop, etc.
    """
    sessions = []
    current_session = None

    def start_new_session(ts, line_text):
        return {
            "change_content_timestamp": ts,
            "svc_name": extract_svc_name(line_text),
            "video_resolution": extract_video_resolution(line_text),
            "alerts": {
                "BlackScreen": {"count": 0, "first_ts": None, "last_ts": None},
                "VideoFreeze": {"count": 0, "first_ts": None, "last_ts": None},
                "DisplayDrop": {
                    "0_99":  {"count":0, "first_ts":None, "last_ts":None},
                    "100_299":{"count":0, "first_ts":None, "last_ts":None},
                    "300_499":{"count":0, "first_ts":None, "last_ts":None},
                    "500_699":{"count":0, "first_ts":None, "last_ts":None},
                    "700_plus":{"count":0, "first_ts":None, "last_ts":None},
                }
            },
            "black_screen_events": [],
            "video_freeze_events": [],
            "other_events": {
                "hdmi_events": [],
                "trick_modes": [],
                "standby_dream": [],
                "tuning_errors": []
            }
        }

    def update_alert(alert_dict, ts):
        alert_dict["count"] += 1
        if alert_dict["first_ts"] is None:
            alert_dict["first_ts"] = ts
        alert_dict["last_ts"] = ts

    for row in rows:
        ts_str = row["timestamp"]
        line_text = parse_line_text(row)
        lower_line = line_text.lower()

        # If we see "CHANGE_CONTENT", treat as new session
        if "change_content" in lower_line:
            if current_session:
                sessions.append(current_session)
            current_session = start_new_session(ts_str, line_text)
            continue

        # If no session started yet, start one implicitly
        if not current_session:
            current_session = start_new_session(ts_str, line_text)

        # Check for blackscreen
        if any(x in lower_line for x in ["blackscreen", "black screen", "black-screen"]):
            update_alert(current_session["alerts"]["BlackScreen"], ts_str)
            current_session["black_screen_events"].append({"ts":ts_str, "line":line_text})

        # Check for video freeze text
        if "video freeze" in lower_line or "video freeze was detected" in lower_line:
            update_alert(current_session["alerts"]["VideoFreeze"], ts_str)
            current_session["video_freeze_events"].append({"ts":ts_str, "line":line_text})

        # Check for display drop
        if "display drop" in lower_line:
            drop_val = parse_display_drop_value(line_text)
            if drop_val is not None:
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

        # Tuning errors
        if any(kw in lower_line for kw in ["ontunestart", "default tune", "defaulttune"]):
            current_session["other_events"]["tuning_errors"].append({"ts":ts_str, "line":line_text})

        # Standby/dream
        if any(kw in lower_line for kw in ["enter day dream","exit day dream","enterstandby"]):
            current_session["other_events"]["standby_dream"].append({"ts":ts_str, "line":line_text})

        # HDMI
        if "hdmi" in lower_line:
            current_session["other_events"]["hdmi_events"].append({"ts":ts_str, "line":line_text})

        # Trick mode
        tm = parse_trick_mode(line_text)
        if tm and "MOTION_SKIP_LIVE" not in tm:
            current_session["other_events"]["trick_modes"].append({"ts":ts_str, "mode":tm, "line":line_text})

        # Update svc_name / resolution if not found yet
        if not current_session["svc_name"]:
            maybe_svc = extract_svc_name(line_text)
            if maybe_svc:
                current_session["svc_name"] = maybe_svc
        if not current_session["video_resolution"]:
            maybe_res = extract_video_resolution(line_text)
            if maybe_res:
                current_session["video_resolution"] = maybe_res

    # End: add last session
    if current_session:
        sessions.append(current_session)

    return sessions

###############################################################################
# 4. Storing results
###############################################################################

def build_summary_text(session):
    """
    Create a text summary for the analysis_json["summary"].
    """
    lines = []
    cts = session["change_content_timestamp"]
    svc = session["svc_name"] or "Unknown"
    res = session["video_resolution"] or "Unknown_Resolution"
    lines.append(f"\nCHANGE_CONTENT at {cts} => svc_name={svc}, resolution={res}\n")
    lines.append("Alert Summary:\n")
    lines.append("| Alert Type    | Count | First Occurrence       | Last Occurrence        |")
    lines.append("|---------------|-------|------------------------|------------------------|")

    ddict = session["alerts"]["DisplayDrop"]
    # flatten
    for label, subalert in ddict.items():
        if subalert["count"] > 0:
            lines.append(f"| DisplayDrop {label} | {subalert['count']} | {subalert['first_ts']} | {subalert['last_ts']} |")

    # freeze
    vf = session["alerts"]["VideoFreeze"]
    lines.append(f"| VideoFreeze   | {vf['count']} | {vf['first_ts']} | {vf['last_ts']} |")

    # blackscreen
    bs = session["alerts"]["BlackScreen"]
    lines.append(f"| BlackScreen   | {bs['count']} | {bs['first_ts']} | {bs['last_ts']} |")

    lines.append("\nOther Events:\n")
    if session["other_events"]["trick_modes"]:
        lines.append("Trick modes used:")
        for ev in session["other_events"]["trick_modes"]:
            lines.append(f"  - {ev['ts']}: {ev['mode']}")
    else:
        lines.append("No trick modes used.")

    if session["other_events"]["tuning_errors"]:
        lines.append("Tuning errors:")
        for te in session["other_events"]["tuning_errors"]:
            lines.append(f"  - {te['ts']}: {te['line']}")
    else:
        lines.append("No tuning errors found.")

    if session["other_events"]["standby_dream"]:
        lines.append("Standby/dream events:")
        for ev in session["other_events"]["standby_dream"]:
            lines.append(f"  - {ev['ts']}: {ev['line']}")
    else:
        lines.append("No standby/dream transitions found.")

    if session["other_events"]["hdmi_events"]:
        lines.append("HDMI events:")
        for ev in session["other_events"]["hdmi_events"]:
            lines.append(f"  - {ev['ts']}: {ev['line']}")
    else:
        lines.append("No HDMI events found.")

    if session["video_freeze_events"] or session["black_screen_events"]:
        lines.append("Distribution of Alerts:")
        if session["video_freeze_events"]:
            lines.append(f"  VideoFreeze events: {len(session['video_freeze_events'])}")
        if session["black_screen_events"]:
            lines.append(f"  BlackScreen events: {len(session['black_screen_events'])}")
    else:
        lines.append("No BlackScreen or VideoFreeze events in this session.")

    return "\n".join(lines)

def store_sessions_in_db(table_name, sessions, start_time, end_time):
    chart_conn = connect_to_chart_db()
    if not chart_conn:
        logging.error("No chart DB connection, skipping store.")
        return
    ensure_analysis_table(chart_conn)

    time_range_str = f"{start_time or 'start_not_specified'} => {end_time or 'end_not_specified'}"
    insert_sql = """
        INSERT INTO video_audio_analysis (table_name, analysis_json)
        VALUES (%s, %s::jsonb);
    """

    with chart_conn.cursor() as cur:
        for sess in sessions:
            # Add summary text and time range analyzed info.
            summary_text = build_summary_text(sess)
            sess["summary"] = summary_text
            sess["time_range_analyzed"] = time_range_str

            # Use the session's change_content_timestamp to avoid duplicates.
            cct = sess.get("change_content_timestamp")
            cur.execute("""
                SELECT COUNT(*) FROM video_audio_analysis
                WHERE table_name=%s
                  AND analysis_json->>'change_content_timestamp' = %s
            """, (table_name, str(cct)))
            already = cur.fetchone()[0]
            if already == 0:
                # Use default=str to serialize datetime objects.
                cur.execute(insert_sql, (table_name, json.dumps(sess, default=str)))
                logging.info("Stored session cct=%s in table_name=%s", cct, table_name)
            else:
                logging.info("Skipping duplicate session cct=%s in %s", cct, table_name)

        chart_conn.commit()

    chart_conn.close()

###############################################################################
# 5. Main CLI
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Analyze logs for the Video/Audio Rendering (VAR) module.")
    parser.add_argument("--module", default="var", help="Must be 'VAR' for now.")
    parser.add_argument("--table_name", default="all", help="Specific R-table or 'all' to parse them.")
    parser.add_argument("--start_time", default=None, help="Start time in ISO format (optional).")
    parser.add_argument("--end_time", default=None, help="End time in ISO format (optional).")
    args = parser.parse_args()

    if args.module.strip().upper() != "VAR":
        print(f"This script only analyzes module=VAR. You passed {args.module}.")
        return

    logs_conn = connect_to_db()
    if not logs_conn:
        return

    if args.table_name.lower() == "all":
        table_names = fetch_all_r_tables(logs_conn)
    else:
        table_names = [args.table_name]

    for tbl in table_names:
        rows = fetch_logs(logs_conn, tbl, start_time=args.start_time, end_time=args.end_time)
        if not rows:
            logging.info("No rows found in %s, skipping.", tbl)
            continue

        sessions = analyze_logs(rows)
        logging.info("Analyzed %d sessions in %s.", len(sessions), tbl)
        store_sessions_in_db(tbl, sessions, args.start_time, args.end_time)

    logs_conn.close()

if __name__ == "__main__":
    main()
