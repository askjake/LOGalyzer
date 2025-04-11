#!/usr/bin/env python3
"""
var.py

Parser for various log file formats used for VAR logs, including:
  - Standard format with full date (YYYY-MM-DD HH:MM:SS.mmm)
  - Short-date format (MM-DD HH:MM:SS.mmm) with a guessed year
  - Alternate ISO-like format (YYYY-MM-DDTHH:MM:SS.mmmZ)

Produces output dictionaries matching the expected ingestion schema:

Expected output keys for each log record:
    directory_file: str         (name of the source file)
    category: str               (the “channel” or source of the log)
    timestamp: str or None      (ISO8601 string, or None if not parsed)
    file_line: str              (the full original log line)
    function: str               (the parsed key part before the colon)
    data: str                   (the parsed data part after the colon)
    message: str                (a combination of function and data, or the full line if not parsed)
    event_type: str             ("error", "warning", "info" – based on log level)
    data_hash: str              (unique hash computed per log line)
"""

import re
import sys
import hashlib
from datetime import datetime

# -------------------------------------------------------------------
# Regex patterns
# -------------------------------------------------------------------
LOG_LINE_RE = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+'
    r'(?P<pid>\d+)\s+(?P<tid>\d+)\s+(?P<level>[A-Z])\s+'
    r'(?:[^\s]+\s+-\s+)?'  # e.g., "VAR - " or "KrtKl-g : "
    r'(?P<channel>[^:]+):\s*(?P<rest>.*)$'
)

SHORTDATE_RE = re.compile(
    r'^(?P<month>\d{2})-(?P<day>\d{2})\s+'
    r'(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}\.\d+)\s+'
    r'(?P<pid>\d+)\s+(?P<tid>\d+)\s+(?P<level>[A-Z])\s+'
    r'(?:VAR\s+-\s+)?'
    r'(?P<channel>[^:]+):\s*(?P<rest>.*)$'
)

ALT_RE = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z):\s*(?P<message>.*)$'
)

FALLBACK_TIMESTAMP_RE = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+(?P<rest>.*)$'
)

DEFAULT_YEAR = 2025  # used for short-date lines if year is missing

# -------------------------------------------------------------------
# Helper: unique MD5 hash per line
# -------------------------------------------------------------------
def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# -------------------------------------------------------------------
# Main parse functions
# -------------------------------------------------------------------
def parse_line(line: str):
    """
    Attempt to parse a single log line using:
      1) Full-date format,
      2) Short-date format,
      3) Alternate ISO format,
      4) Fallback to at least extracting timestamp,
      5) Otherwise store entire line as unparsed.

    Returns a dict with keys:
      { timestamp, pid, tid, level, channel, key, value }
      or None if line is empty.
    """
    line = line.rstrip()
    if not line:
        return None

    m = LOG_LINE_RE.match(line)
    if m:
        return _handle_standard_line(m)

    m = SHORTDATE_RE.match(line)
    if m:
        return _handle_shortdate_line(m)

    m = ALT_RE.match(line)
    if m:
        return _handle_alt_iso_line(m)

    # Fallback: try to extract at least a known timestamp
    m = FALLBACK_TIMESTAMP_RE.match(line)
    if m:
        d = m.groupdict()
        return {
            "timestamp": d["timestamp"],
            "pid": "",
            "tid": "",
            "level": "",
            "channel": "fallback",
            "key": "line",
            "value": d["rest"].strip(),
        }

    # If we get here, store entire line
    print(f"[WARNING] No recognized pattern: {line}", file=sys.stderr)
    return {
        "timestamp": None,
        "pid": "",
        "tid": "",
        "level": "",
        "channel": "fallback",
        "key": "message",
        "value": line,
    }


def _handle_standard_line(match):
    d = match.groupdict()
    ts_str = d["timestamp"]
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        d["timestamp"] = dt.isoformat()
    except ValueError:
        d["timestamp"] = None

    rest = d.pop("rest").strip()
    if ":" in rest:
        # Split on the first colon so we get a function-like key vs. data
        key_part, value_part = rest.split(':', 1)
        d["key"] = key_part.strip()
        d["value"] = value_part.strip()
    else:
        d["key"] = rest
        d["value"] = ""
    return d


def _handle_shortdate_line(match):
    d = match.groupdict()
    guessed_str = f"{DEFAULT_YEAR}-{d['month']}-{d['day']} {d['hour']}:{d['minute']}:{d['second']}"
    try:
        dt = datetime.strptime(guessed_str, "%Y-%m-%d %H:%M:%S.%f")
        d["timestamp"] = dt.isoformat()
    except ValueError:
        d["timestamp"] = None

    rest = d.pop("rest").strip()
    if ":" in rest:
        key_part, value_part = rest.split(':', 1)
        d["key"] = key_part.strip()
        d["value"] = value_part.strip()
    else:
        d["key"] = rest
        d["value"] = ""
    for x in ("month", "day", "hour", "minute", "second"):
        d.pop(x, None)
    return d


def _handle_alt_iso_line(match):
    d = match.groupdict()
    iso_str = d["timestamp"]
    try:
        dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        d["timestamp"] = dt.isoformat()
    except ValueError:
        pass
    return {
        "timestamp": d["timestamp"],
        "pid": "",
        "tid": "",
        "level": "",
        "channel": "general",
        "key": "message",
        "value": d["message"].strip(),
    }

# -------------------------------------------------------------------
# Transform into final ingestion schema
# -------------------------------------------------------------------
def transform_log(parsed: dict, original_line: str, original_file: str, line_no: int):
    """
    Convert the parsed dict into the final schema needed by ingestion,
    specifically placing the meaningful portion into 'data'.

    'file_line' stores the entire raw line for reference,
    'function' is from 'key',
    'data' is from 'value',
    'message' is a combination,
    'event_type' is derived from log level,
    'data_hash' is an MD5 of the line + line_no (unique per line).
    """
    category = parsed.get("channel", "").strip()
    timestamp = parsed.get("timestamp") or None
    function = parsed.get("key", "").strip()
    data = parsed.get("value", "").strip()
    level = parsed.get("level", "").upper()

    # Build the "message" field
    if function and data:
        message = f"{function}: {data}"
    elif function:
        message = function
    elif data:
        message = data
    else:
        message = original_line

    # event_type by level
    if level == "E":
        event_type = "error"
    elif level == "W":
        event_type = "warning"
    elif level == "I":
        event_type = "info"
    else:
        event_type = "info"  # default to info if unknown

    # Unique hash
    unique_hash = compute_hash(f"{original_line}::{line_no}")

    return {
        "directory_file": original_file,
        "category": category,
        "timestamp": timestamp,
        "file_line": original_line,
        "function": function,
        "data": data,          # <--- critical for varnalyzer to detect freeze lines
        "message": message,
        "event_type": event_type,
        "data_hash": unique_hash,
    }

def parse_var(content: str, original_log_file_name: str = "") -> list:
    """
    Parse the content of a VAR log file line-by-line.
    Return a list of dictionaries (the final schema).
    """
    results = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        parsed = parse_line(line)
        if parsed is None:
            continue
        record = transform_log(parsed, line, original_log_file_name, line_no)
        results.append(record)
    return results


# -------------------------------------------------------------------
# Quick test if run directly
# -------------------------------------------------------------------
if __name__ == "__main__":
    import pprint

    sample_log = """\
2025-03-27 19:50:32.666  4807 31770 I KrtKl-g : aspect ratio scale factor: 1
04-04 08:26:59.936  4932  1019 I VAR - i : Codec presentationTimeUs: 77678966177
2025-03-23T21:52:57.639Z: First log message
Some unknown line that doesn't match any pattern
2025-03-27 20:00:01.123  4807 31770 I KrtKl-g : Video freeze was detected
"""
    parsed_records = parse_var(sample_log, "test4.varapp")
    pprint.pprint(parsed_records)
