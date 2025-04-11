
# ingestion.db.postgres_storer.py

import re
import os
import json
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
from utils import (
    sftp_list_files, read_remote_file, compute_file_hash,
    is_file_imported, mark_file_as_imported, create_imported_files_table
)


# ----------------------------------------------------------------------
# 1) Load credentials & set up a global connection pool
# ----------------------------------------------------------------------
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
if DB_CREDENTIALS:
    try:
        pg_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=20,
            host=DB_CREDENTIALS["DB_HOST"],
            database=DB_CREDENTIALS["DB_NAME"],
            user=DB_CREDENTIALS["DB_USER"],
            password=DB_CREDENTIALS["DB_PASSWORD"]
        )
        print(f"✅ PostgreSQL Connection Pool Initialized: {DB_CREDENTIALS['DB_HOST']}")
    except Exception as e:
        print(f"❌ Failed to initialize PostgreSQL pool: {e}")
        pg_pool = None
else:
    print("❌ Missing or invalid database credentials.")
    pg_pool = None


def pg_connect():
    """Get a connection from the pool instead of creating a new one each time."""
    if pg_pool:
        return pg_pool.getconn()
    print("❌ pg_pool is not initialized.")
    return None


def release_pg_conn(conn):
    """Release the connection back to the pool."""
    if pg_pool and conn:
        pg_pool.putconn(conn)


def connect(credentials):
    """Directly connect to PostgreSQL (not recommended for frequent queries)."""
    try:
        return psycopg2.connect(
            host=credentials["DB_HOST"],
            database=credentials["DB_NAME"],
            user=credentials["DB_USER"],
            password=credentials["DB_PASSWORD"]
        )
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return None


# ----------------------------------------------------------------------
# Helper: Ensure unique constraint on a given column
# ----------------------------------------------------------------------
def ensure_unique_constraint(connection, table_name, column='data_hash'):
    cursor = connection.cursor()
    # Generate a unique constraint name based on the table name.
    constraint_name = f"{table_name}_data_hash_unique"
    try:
        cursor.execute("""
            SELECT tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name
            WHERE tc.table_name = %s
              AND tc.constraint_type = 'UNIQUE'
              AND ccu.column_name = %s
        """, (table_name, column))
        result = cursor.fetchone()
        if not result:
            alter_query = f'ALTER TABLE "{table_name}" ADD CONSTRAINT {constraint_name} UNIQUE ({column})'
            cursor.execute(alter_query)
            connection.commit()
            #print(f"✅ Added unique constraint {constraint_name} on {column} in table {table_name}")
        #else:
            #print(f"[INFO] Unique constraint on {column} already exists in table {table_name}.")
    except Exception as e:
        connection.rollback()
        print(f"[WARN] Could not add unique constraint on {column} in table {table_name}: {e}")
    finally:
        cursor.close()

# ----------------------------------------------------------------------
# 2) Create or update tables
# ----------------------------------------------------------------------
def create_table_if_not_exists(connection, table_name):
    """Creates a per-log table if it does not already exist, then ensures it has the unique constraint."""
    cursor = connection.cursor()
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS "{table_name}" (
        id SERIAL PRIMARY KEY,
        directory_file TEXT NOT NULL,
        category TEXT,
        timestamp TIMESTAMP WITH TIME ZONE,
        file_line TEXT,
        function TEXT,
        data TEXT,
        data_hash TEXT,
        message TEXT,
        event_type TEXT
    )
    """
    cursor.execute(create_table_query)
    connection.commit()
    cursor.close()
    # Ensure the unique constraint exists on data_hash
    ensure_unique_constraint(connection, table_name)


def get_existing_columns(cursor, table_name):
    """Retrieves existing columns in a table."""
    cursor.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
    """, (table_name.lower(),))
    return {row[0].lower() for row in cursor.fetchall()}


def add_missing_columns(cursor, table_name, new_columns):
    """
    Dynamically adds missing columns to the table, ensuring duplicates aren't added.
    Each column is created as TEXT (you can refine types if needed).
    """
    existing_columns = get_existing_columns(cursor, table_name)
    for column in new_columns:
        col_lower = column.lower()
        if col_lower not in existing_columns and column != "id":
            try:
                alter_query = f"""ALTER TABLE "{table_name}" ADD COLUMN "{column}" TEXT"""
                cursor.execute(alter_query)
                cursor.connection.commit()
                #print(f"✅ Added column '{column}' to table '{table_name}'")
            except Exception as e:
                cursor.connection.rollback()


def create_bulk_table_if_not_exists(connection, bulk_table_name):
    """
    Creates a bulk table:
      - includes rxid, rxid2, source_table, plus standard log columns.
      - Then ensures the unique constraint on data_hash is present.
    """
    cursor = connection.cursor()
    create_query = f"""
    CREATE TABLE IF NOT EXISTS "{bulk_table_name}" (
        id SERIAL PRIMARY KEY,
        rxid TEXT,
        rxid2 TEXT,
        source_table TEXT,
        directory_file TEXT,
        category TEXT,
        timestamp TIMESTAMP WITH TIME ZONE,
        file_line TEXT,
        function TEXT,
        data TEXT,
        data_hash TEXT,
        message TEXT,
        event_type TEXT
    )
    """
    cursor.execute(create_query)
    connection.commit()
    cursor.close()
    ensure_unique_constraint(connection, bulk_table_name)


def extract_rxids(table_name: str):
    """
    Parses table_name for:
      - primary RxID => the first 'R' followed by 10 digits
      - optional secondary RxID => a second 'R' followed by 10 digits if found
    Returns (primary_rxid, secondary_rxid_or_None)
    """
    pattern = r'(R\d{10})'
    matches = re.findall(pattern, table_name)
    if not matches:
        return None, None
    primary = matches[0]
    secondary = matches[1] if len(matches) > 1 else None
    return primary, secondary


def store_in_bulk_table(connection, logs, primary_rxid, secondary_rxid, source_table):
    """
    Duplicates rows into a "bulk" table named after the primary_rxid.
    After inserting into the bulk table, this function calls the enricher.
    """
    if not primary_rxid:
        return  # no primary RxID => can't store in bulk table

    bulk_table_name = f"{primary_rxid}"
    create_bulk_table_if_not_exists(connection, bulk_table_name)

    # Add extra fields to every record
    for log in logs:
        log["rxid"] = primary_rxid
        log["rxid2"] = secondary_rxid
        log["source_table"] = source_table

    # Collect all keys from logs and ensure columns exist
    cursor = connection.cursor()
    new_columns = {key for log in logs for key in log.keys()}
    add_missing_columns(cursor, bulk_table_name, new_columns)

    columns = list(new_columns)
    values = [tuple(log.get(col, None) for col in columns) for log in logs]

    insert_query = f"""
        INSERT INTO "{bulk_table_name}" ({', '.join(f'"{col}"' for col in columns)})
        VALUES %s
        ON CONFLICT (data_hash) DO NOTHING
    """
    try:
        execute_values(cursor, insert_query, values)
        cursor.connection.commit()
        rows_attempted = len(values)
        rows_inserted = cursor.rowcount
        duplicates = rows_attempted - rows_inserted
        dup_pct = (duplicates / rows_attempted) * 100 if rows_attempted else 0
        print(f"✅ Migrating => Table: {bulk_table_name}")
    except Exception as e:
        print(f"❌ Error storing logs in bulk table '{bulk_table_name}': {e}")
        cursor.connection.rollback()
    finally:
        cursor.close()

    # Call the enricher for the bulk table
    import subprocess
    try:
        subprocess.run(["python", "ingestion/linking/enricher_bert.py", "--table_name", bulk_table_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ENRICHER CALL] Error calling enricher for bulk table '{bulk_table_name}': {e}")


# ----------------------------------------------------------------------
# 4) The main storage function, with table enrichment
# ----------------------------------------------------------------------
def store_parsed_logs(connection, table_name, logs, original_log_file_name):
    """
    1) Insert the logs into the given 'table_name'.
    2) Also merge them into a "bulk" table (named {primary_rxid}).
    3) (Previously) Drop the source table if it met certain naming patterns.
       Now, we skip dropping so that records are preserved.
    """
    cursor = connection.cursor()
    if not logs:
        print("No logs to insert.")
        cursor.close()
        return

    # Step A: Prepare logs for insertion
    for log in logs:
        if "directory_file" not in log or log["directory_file"] is None:
            log["directory_file"] = original_log_file_name
        if "message" not in log:
            log["message"] = "N/A"
        if "data_hash" not in log:
            log["data_hash"] = compute_file_hash(log.get("data", ""))
        if "event_type" not in log:
            log["event_type"] = None

    create_table_if_not_exists(connection, table_name)

    # Ensure columns in original table
    new_columns = {key for log in logs for key in log.keys()}
    add_missing_columns(cursor, table_name, new_columns)

    # Insert into the original table
    columns = list(new_columns)
    values = [tuple(log.get(col, None) for col in columns) for log in logs]
    insert_query = f"""
        INSERT INTO "{table_name}" ({', '.join(f'"{col}"' for col in columns)})
        VALUES %s
        ON CONFLICT (data_hash) DO NOTHING
    """
    try:
        execute_values(cursor, insert_query, values)
        connection.commit()
    except Exception as e:
        print(f"❌ Error inserting logs into '{table_name}': {e}")
        connection.rollback()
        cursor.close()
        return
    finally:
        cursor.close()

    rows_attempted = len(values)
    rows_inserted = cursor.rowcount if rows_attempted else 0
    duplicates = rows_attempted - rows_inserted
    dup_pct = (duplicates / rows_attempted) * 100 if rows_attempted else 0

    # Step B: Merge logs into the "bulk" table
    primary_rxid, secondary_rxid = extract_rxids(table_name)
    store_in_bulk_table(connection, logs, primary_rxid, secondary_rxid, table_name)

    # Step C: (Previously) Drop the single table based on naming rules.
    # Instead of dropping, we now keep the original table intact.
    print(f"[CLEANUP] Keeping table '{table_name}' intact (records not removed).")
