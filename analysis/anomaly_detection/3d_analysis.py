#!/usr/bin/env python3
import os
import json
import psycopg2
import pandas as pd
import numpy as np
import plotly.express as px
import argparse
from datetime import datetime, timezone


def load_db_credentials():
    """Load PostgreSQL credentials from credentials.txt"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    credentials_file = os.path.join(base_dir, "credentials.txt")
    try:
        with open(credentials_file, "r") as f:
            creds = json.load(f)
            return {
                "DB_HOST": creds["DB_HOST"],
                "DB_NAME": creds["DB_NAME"],
                "DB_USER": creds["DB_USER"],
                "DB_PASSWORD": creds["DB_PASSWORD"]
            }
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading credentials: {e}")
        return None


DB_CREDENTIALS = load_db_credentials()


def pg_connect():
    """Establishes a database connection."""
    if not DB_CREDENTIALS:
        return None
    try:
        conn = psycopg2.connect(
            host=DB_CREDENTIALS["DB_HOST"],
            database=DB_CREDENTIALS["DB_NAME"],
            user=DB_CREDENTIALS["DB_USER"],
            password=DB_CREDENTIALS["DB_PASSWORD"]
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None


def fetch_anomalies(start_date=None, end_date=None) -> pd.DataFrame:
    """
    Fetch all records from the 'anomalies' table with an optional date filter.
    No filtering is applied on anomaly flags so that all records are plotted.
    """
    conn = pg_connect()
    if not conn:
        print("❌ No database connection available.")
        return pd.DataFrame()

    try:
        query = """
        SELECT "timestamp",
               COALESCE(rxid, "RxID") AS rxid,
               COALESCE(significance_score, 0) AS significance_score,
               autoenc_is_anomaly,
               lstm_is_anomaly,
               sgs_duration_ms,
               event_type,
               message
        FROM public.anomalies
        """
        # Add date filters if provided
        if start_date and end_date:
            query += f" WHERE \"timestamp\" >= '{start_date}' AND \"timestamp\" <= '{end_date}'"
        elif start_date:
            query += f" WHERE \"timestamp\" >= '{start_date}'"
        elif end_date:
            query += f" WHERE \"timestamp\" <= '{end_date}'"
        query += " ORDER BY \"timestamp\" DESC"
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"❌ Error fetching anomalies: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def normalize_timestamps(series: pd.Series) -> pd.Series:
    """
    Parses timestamps using pandas.to_datetime with errors='coerce',
    and replaces any NaT with the median of valid timestamps.
    """
    parsed = pd.to_datetime(series, errors='coerce', utc=True)
    valid_times = parsed.dropna()
    if valid_times.empty:
        return parsed
    median_time = valid_times.median()
    return parsed.fillna(median_time)


def assign_color(row):
    """
    Determine the color for each anomaly based on:
      - 'epg handle' in message or event_type => green
      - 'signal protector' in message or event_type => yellow
      - both auto & lstm => black
      - auto only => red
      - lstm only => orange
      - else => blue
    """
    msg_lower = str(row["message"]).lower()
    evt_lower = str(row["event_type"]).lower()

    if "epg handle" in msg_lower or "epg handle" in evt_lower:
        return "green"
    elif "signal protector" in msg_lower or "signal protector" in evt_lower:
        return "yellow"
    elif row["autoenc_is_anomaly"] and row["lstm_is_anomaly"]:
        return "black"
    elif row["autoenc_is_anomaly"]:
        return "red"
    elif row["lstm_is_anomaly"]:
        return "orange"
    else:
        return "blue"


def process_data(df):
    """Prepares and formats data for plotting."""
    if df.empty:
        print("⚠️ No data found.")
        return None

    # Normalize timestamps and ensure we keep the datetime type
    df["timestamp"] = normalize_timestamps(df["timestamp"])
    # Create a version of timestamp that ignores the year (e.g., set year to 2000)
    df["timestamp_no_year"] = df["timestamp"].apply(lambda dt: dt.replace(year=2000))
    # Convert timestamps to numeric (Unix timestamp in seconds) for 3D plotting
    df["timestamp_numeric"] = df["timestamp_no_year"].apply(lambda x: x.timestamp())

    # Convert significance to numeric; if null, set to 0
    df["significance_score"] = pd.to_numeric(df["significance_score"], errors='coerce').fillna(0)

    # Assign color based on anomaly-related flags or keywords
    df["color"] = df.apply(assign_color, axis=1)

    # Scale the point size by sgs_duration_ms (if available)
    df["size"] = pd.to_numeric(df["sgs_duration_ms"], errors='coerce').fillna(1)
    if df["size"].max() > 0:
        df["size"] = np.clip(df["size"] / df["size"].max() * 20, 5, 30)
    else:
        df["size"] = 5

    return df


def plot_3d_anomalies(df):
    """Generates a 3D scatter plot with smaller dot sizes."""
    if df.empty:
        print("[WARN] No valid data for plotting.")
        return

    print(f"[INFO] Plotting {len(df)} anomaly records on 3D scatter plot.")
    fig = px.scatter_3d(
        df,
        x="timestamp_numeric",
        y="rxid",
        z="significance_score",
        color="color",
        size="size",
        size_max=8,  # Adjust dot size as needed
        hover_data=["timestamp", "event_type", "message"],
        title="3D Anomaly Correlation (Time vs RxID vs Significance)"
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Time (Unix Timestamp)",
            yaxis_title="Receiver ID",
            zaxis_title="Anomaly Significance",
        ),
        legend_title="Anomaly Category",
        template="plotly_dark"
    )
    fig.show()


def plot_pivot_heatmap(df):
    """
    Aggregates the anomaly records into a pivot table with daily time buckets and rxid,
    and displays a heatmap of anomaly counts.
    """
    if df.empty:
        print("[WARN] No data available for pivot table.")
        return

    # Create a daily time bucket using the normalized timestamp
    df['date_bucket'] = df['timestamp'].dt.floor('D')
    # Aggregate anomaly count by day and rxid
    pivot_df = df.pivot_table(index='date_bucket', columns='rxid',
                              values='significance_score', aggfunc='count', fill_value=0)
    # Convert index to string for display
    pivot_df.index = pivot_df.index.strftime("%Y-%m-%d")

    print(f"[INFO] Creating heatmap pivot table with {pivot_df.shape[0]} days and {pivot_df.shape[1]} rxids.")
    fig = px.imshow(pivot_df,
                    labels=dict(x="Receiver ID", y="Date", color="Anomaly Count"),
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    title="Heatmap of Anomalies: Count per Day by Receiver")
    fig.update_layout(template="plotly_dark")
    fig.show()


def main():
    parser = argparse.ArgumentParser(description="3D Anomaly Plot and Pivot Table with Date and Receiver ID Filters")
    parser.add_argument("--start_date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("--rxids", type=str, help="Comma-separated list of receiver IDs to filter")
    args = parser.parse_args()

    rxid_list = None
    if args.rxids:
        rxid_list = [r.strip() for r in args.rxids.split(",")]

    df = fetch_anomalies(args.start_date, args.end_date)
    if rxid_list:
        # Filter DataFrame to include only specified receiver IDs
        df = df[df["rxid"].isin(rxid_list)]

    df = process_data(df)
    if df is not None and not df.empty:
        # Plot the 3D scatter graph
        plot_3d_anomalies(df)
        # Then create and display the pivot table heatmap
        plot_pivot_heatmap(df)
    else:
        print("[INFO] Nothing to plot: No records match the criteria.")


if __name__ == "__main__":
    main()
