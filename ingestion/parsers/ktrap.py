#!/usr/bin/env python3
"""
ingestion/parsers/ktrap.py

Parses multi-line k_trap log blocks. Each block typically starts with:
   ------------------------- TIME Received --------------------------
   ----------------- 08/30/2024 08:35:00 ---------------
   ----- Software: PTA1NHBD-N -- Model: Hopper with 16 ------------------------
   ...
   (some lines)
   ...
   [Next block starts with "------------------------- TIME Received --------------------------" again]

Returns a list of dicts, each representing one 'ktrap' message.
"""

import re
from datetime import datetime

TIME_HEADER = r"^-+\sTIME Received\s-+$"
DATESTAMP_LINE = r"^----+\s+(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s+----+$"
SOFTWARE_MODEL_LINE = r"^----- Software:\s+([^-\s]+)\s*--\s*Model:\s*(.+)\s*----+$"

def parse_ktrap_file(content: str, rx_id: str) -> list:
    """
    Splits the entire file content by blocks, extracts each block's date/time, software info,
    and the raw message lines.
    Returns a list of dicts with fields like:
       {
         "timestamp": ISO8601 string,
         "rx_id": <rx_id>,
         "category": "k_trap",
         "software": <parsed software version>,
         "model": <parsed model>,
         "data": <entire block as a single multiline string>,
       }
    """
    # Split into blocks by "------------------------- TIME Received --------------------------"
    # We'll keep the delimiter in the result so we can parse the date/time line that follows.
    blocks = re.split(r"(?:^-+ TIME Received -+$)", content, flags=re.MULTILINE)

    results = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Usually the block starts with lines:
        #   ----------------- 08/30/2024 08:35:00 ---------------
        #   ----- Software: PTA1NHBD-N -- Model: Hopper with 16 ------------------------
        #   ...
        lines = block.splitlines()
        # We expect the first or second line to have the date stamp
        # and the next line to have software info.
        # Let's scan them:
        timestamp_str = None
        software = None
        model = None

        # We store the entire block as "data"
        block_data = block

        # parse date from lines
        for i, line in enumerate(lines):
            line = line.strip()
            date_match = re.match(DATESTAMP_LINE, line)
            if date_match:
                # parse date
                date_str = date_match.group(1)  # "08/30/2024 08:35:00"
                # Convert to ISO8601
                try:
                    dt = datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S")
                    timestamp_str = dt.isoformat()
                except ValueError:
                    # fallback - just store the raw date
                    timestamp_str = date_str
                # continue looking for software line
            sw_model_match = re.match(SOFTWARE_MODEL_LINE, line)
            if sw_model_match:
                software = sw_model_match.group(1).strip()
                model = sw_model_match.group(2).strip()
                break  # we have enough info, skip the rest

        # fallback if we didn't find a timestamp
        if not timestamp_str:
            # no date found, we can default to None or store a fallback
            # but let's just store the current time
            timestamp_str = datetime.now().isoformat()

        result = {
            "timestamp": timestamp_str,
            "rx_id": rx_id,
            "category": "k_trap",
            "software": software,
            "model": model,
            "data": block_data,  # store entire block for reference
        }
        results.append(result)

    return results
