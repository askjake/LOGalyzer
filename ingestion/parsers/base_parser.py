# base_parser.py

import re
from datetime import datetime, timezone
from dateutil import parser as dateparser

class BaseParser:
    """
    Base class for log parsing. Provides standard parsing methods.
    """

    def __init__(self):
        self.last_valid_timestamp = None

    def convert_to_utc(self, timestamp_str):
        """Converts timestamps to UTC format."""
        timestamp_str = timestamp_str.strip()
        current_year = datetime.now().year
        try:
            dt = dateparser.parse(timestamp_str, fuzzy=True)
        except:
            if re.match(r'^\d{1,2}/\d{1,2}\b', timestamp_str):
                timestamp_str = f"{current_year}/{timestamp_str}"
            elif re.match(r'^[A-Za-z]{3,}\s+\d{1,2}\b', timestamp_str):
                parts = timestamp_str.split()
                if len(parts) >= 2:
                    timestamp_str = f"{parts[0]} {parts[1]} {current_year} {' '.join(parts[2:])}"
            try:
                dt = dateparser.parse(timestamp_str, fuzzy=True)
            except:
                dt = None

        if not dt:
            return self.fallback_to_previous_timestamp(timestamp_str)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.isoformat()

    def fallback_to_previous_timestamp(self, timestamp_str):
        """Falls back to the last valid timestamp if a new one is invalid."""
        if self.last_valid_timestamp:
            time_match = re.search(r'(\d{1,2}):(\d{1,2}):(\d{1,2})', timestamp_str)
            if time_match:
                hours, minutes, seconds = map(int, time_match.groups())
                fallback = self.last_valid_timestamp.replace(
                    hour=hours, minute=minutes, second=seconds, microsecond=0
                )
                return fallback.astimezone(timezone.utc).isoformat()
            else:
                return self.last_valid_timestamp.astimezone(timezone.utc).isoformat()
        return None

    def parse_log_line(self, line):
        """
        Placeholder method for parsing log lines.
        Should be implemented in child classes.
        """
        raise NotImplementedError("Subclasses should implement this method.")

