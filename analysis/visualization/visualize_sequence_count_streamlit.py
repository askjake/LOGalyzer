#!/usr/bin/env python3
"""
visualize_sequence_count_streamlit.py

Streamlit dashboard that displays the current row counts for tables as stored in
the tracking table "table_row_counts". It automatically creates the tracking table
if needed, obtains credentials from credentials.txt, and refreshes every 60 seconds.
"""

import os
import json
import psycopg2
import pandas as pd
import streamlit as st
import logging
import time
from datetime import datetime
from psycopg2.extras import DictCursor
import plotly.express as px

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

# Auto-refresh every 60 seconds
REFRESH_INTERVAL_SEC = 60
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL_SEC:
    st.session_state.last_refresh = time.time()
    st.experimental_rerun()

# Read credentials from file
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
creds_path = os.path.join(base_dir, "credentials.txt")
if not os.path.exists(creds_path):
    st.error(f"credentials.txt not found: {creds_path}")
    st.stop()

with open(creds_path, "r") as f:
    creds = json.load(f)

db_host = creds.get("DB_HOST")
db_name = creds.get("DB_NAME")
db_user = creds.get("DB_USER")
db_pass = creds.get("DB_PASSWORD")

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_pass)
    logging.info("Connected to DB.")
except Exception as e:
    st.error(f"DB connection error: {e}")
    st.stop()

# Create the tracking table if it doesn't exist
create_table_sql = """
CREATE TABLE IF NOT EXISTS table_row_counts (
    table_name TEXT PRIMARY KEY,
    row_count BIGINT,
    last_updated TIMESTAMP DEFAULT now()
);
"""
with conn.cursor() as cur:
    cur.execute(create_table_sql)
    conn.commit()
    logging.info("Ensured table_row_counts exists.")

# Query the row counts from the tracking table
query = """
SELECT table_name, row_count, last_updated
FROM table_row_counts
ORDER BY row_count DESC;
"""
try:
    df = pd.read_sql(query, conn)
except Exception as e:
    st.error(f"Error querying table_row_counts: {e}")
    conn.close()
    st.stop()

conn.close()

# Display the results in the Streamlit app
st.title("Table Row Counts")
st.write("This dashboard displays the current row counts for tables as tracked in `table_row_counts`.")

if df.empty:
    st.write("No data available. Please ensure the tracking table is populated.")
else:
    fig = px.bar(
        df,
        x="table_name",
        y="row_count",
        hover_data=["last_updated"],
        title="Row Counts per Table",
        labels={"table_name": "Table Name", "row_count": "Row Count"}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Raw Data")
    st.dataframe(df)
