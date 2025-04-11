import os
import stat
import gzip
import hashlib
import psycopg2

def create_database_if_not_exists(admin_host, admin_user, admin_password, target_db):
    """
    Connects to the 'postgres' default DB on admin_host using admin_user/admin_password.
    Checks if 'target_db' exists; if not, creates it.
    admin_user must have CREATEDB privileges or be superuser.
    """
    try:
        # Connect to the default 'postgres' db
        admin_conn = psycopg2.connect(
            host=admin_host,
            dbname=target_db,
            user=admin_user,
            password=admin_password
        )
        admin_conn.autocommit = True
        with admin_conn.cursor() as cur:
            # check if DB exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (target_db,))
            exists = cur.fetchone()
            if not exists:
                print(f"[AUTO-DB] Database '{target_db}' doesn't exist. Creating...")
                cur.execute(f'CREATE DATABASE "{target_db}"')
            else:
                print(f"[AUTO-DB] Database '{target_db}' already exists.")
    except Exception as e:
        print(f"[AUTO-DB] Error creating DB '{target_db}': {e}")
    finally:
        if 'admin_conn' in locals():
            admin_conn.close()


def ensure_database_exists(creds, target_key, admin_keys=("DB_HOST", "IMPORT_DB_ADMIN_USER", "IMPORT_DB_ADMIN_PASS")):
    """
    Checks if the database specified by creds[target_key] exists.
    Expects that creds contains the target database name (e.g., HAPPY_PATH_DB) along with admin credentials.
    """
    if target_key not in creds:
        print(f"[ERROR] Missing '{target_key}' in credentials.")
        return
    target_db = creds[target_key]
    # Use the same host as DB_HOST; admin credentials must be provided in creds.
    admin_host = creds.get("DB_HOST")
    admin_user = creds.get("IMPORT_DB_ADMIN_USER", creds.get("DB_USER"))
    admin_pass = creds.get("IMPORT_DB_ADMIN_PASS", creds.get("DB_PASSWORD"))
    if not all([admin_host, admin_user, admin_pass]):
        print("[ERROR] Missing admin credentials for database creation.")
        return
    create_database_if_not_exists(admin_host, admin_user, admin_pass, target_db)

def connect_import_db(creds):
    """
    1) Using admin credentials, create the IMPORT_DB if it doesn't exist.
    2) Connect to IMPORT_DB using normal import user credentials.
    3) Return that connection.
    """
    admin_host = creds["IMPORT_DB_HOST"]
    admin_user = creds["IMPORT_DB_ADMIN_USER"]  # admin user that can create DB
    admin_pass = creds["IMPORT_DB_ADMIN_PASS"]
    target_db_name = creds["IMPORT_DB"]  # the DB name we want to create

    create_database_if_not_exists(admin_host, admin_user, admin_pass, target_db_name)

    import_user = creds["IMPORT_DB_USER"]
    import_pass = creds["IMPORT_DB_PASSWORD"]
    try:
        conn = psycopg2.connect(
            host=admin_host,
            dbname=target_db_name,
            user=import_user,
            password=import_pass
        )
        return conn
    except Exception as e:
        print(f"[IMPORT] Error connecting to import_db '{target_db_name}' as user '{import_user}': {e}")
        return None

def sftp_list_files(sftp, remote_path, file_filter=None):
    all_files = []
    def recursive_list(path):
        try:
            for entry in sftp.listdir_attr(path):
                filename = entry.filename if entry.filename else ""
                full_path = os.path.join(path, filename).replace("\\", "/")
                if stat.S_ISDIR(entry.st_mode):
                    recursive_list(full_path)
                elif file_filter is None or (isinstance(filename, str) and file_filter in filename):
                    all_files.append(full_path)
        except Exception as e:
            print(f"Error accessing {path}: {e}")
    recursive_list(remote_path)
    return all_files

def normalize_remote_path(path):
    return path.replace("\\", "/")

def read_remote_file(sftp, remote_file_path):
    remote_file_path = normalize_remote_path(remote_file_path)
    try:
        with sftp.open(remote_file_path, 'rb') as remote_file:
            if remote_file_path.endswith('.gz'):
                with gzip.GzipFile(fileobj=remote_file) as gz_file:
                    content = gz_file.read()
            else:
                content = remote_file.read()
        return content.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Failed to read remote file {remote_file_path}: {e}")
        return None

def compute_file_hash(file_content):
    """Compute a hash for the file content."""
    return hashlib.md5(file_content.encode('utf-8')).hexdigest()

def create_imported_files_table(connection):
    """Create 'imported_files' table in import_db if not exists."""
    if not connection:
        print("[IMPORT] No connection to import_db. Can't create table.")
        return
    cursor = connection.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS imported_files (
        id SERIAL PRIMARY KEY,
        file_name TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT imported_files_file_hash_unique UNIQUE (file_hash)
    )
    """
    cursor.execute(create_table_query)
    connection.commit()
    cursor.close()

def ensure_imported_files_schema(connection):
    """
    Ensures that the 'imported_files' table has the necessary columns and unique constraint.
    """
    if not connection:
        print("[IMPORT] No connection to import_db. Can't ensure schema.")
        return
    cursor = connection.cursor()
    # First, create the table if it does not exist
    create_imported_files_table(connection)
    # Now attempt to add missing columns (if any) and enforce a unique constraint on file_hash.
    try:
        alter_query = """
        ALTER TABLE imported_files
          ADD COLUMN IF NOT EXISTS file_name TEXT NOT NULL DEFAULT 'unknown',
          ADD COLUMN IF NOT EXISTS file_hash TEXT;
        """
        cursor.execute(alter_query)
        connection.commit()
    except Exception as e:
        print(f"[DEBUG] Issue altering table for missing columns: {e}")

    # Ensure unique constraint on file_hash. This may fail if it already exists.
    try:
        unique_query = """
        ALTER TABLE imported_files
          ADD CONSTRAINT imported_files_file_hash_unique UNIQUE (file_hash)
        """
        cursor.execute(unique_query)
        connection.commit()
    except Exception as e:
        # If the error is that the constraint already exists, we can ignore it.
        if "already exists" in str(e):
            pass
        else:
            print(f"[DEBUG] Issue enforcing unique constraint: {e}")
    cursor.close()

def is_file_imported(connection, file_hash):
    """
    Check if file_hash is in 'imported_files' table of this import DB connection.
    If the table or column is missing, we create/alter it, rollback, and retry.
    """
    if not connection:
        print("[IMPORT] No connection to import_db. Assuming not imported.")
        return False

    check_query = "SELECT COUNT(*) FROM imported_files WHERE file_hash = %s"
    try:
        cursor = connection.cursor()
        cursor.execute(check_query, (file_hash,))
        count = cursor.fetchone()[0]
        cursor.close()
        return (count > 0)

    except psycopg2.Error as e:
        error_message = str(e).lower()
        cursor.close()  # make sure the cursor is closed if there's an error

        # 1) If the table doesn't exist, rollback and create it.
        if "relation \"imported_files\" does not exist" in error_message:
            print("[DEBUG] 'imported_files' table missing. Creating table...")
            connection.rollback()  # <-- IMPORTANT: rollback first
            create_imported_files_table(connection)
            # Now that the table is created, retry once:
            try:
                cursor = connection.cursor()
                cursor.execute(check_query, (file_hash,))
                count = cursor.fetchone()[0]
                cursor.close()
                return (count > 0)
            except Exception as e2:
                print(f"[ERROR] is_file_imported failed even after creating table: {e2}")
                connection.rollback()
                return False

        # 2) If the column doesn't exist, rollback and alter the table.
        elif "column \"file_hash\" does not exist" in error_message:
            print("[DEBUG] Column 'file_hash' missing in imported_files. Altering table...")
            connection.rollback()
            try:
                cursor = connection.cursor()
                alter_query = """
                ALTER TABLE imported_files
                  ADD COLUMN IF NOT EXISTS file_hash TEXT;
                """
                cursor.execute(alter_query)
                connection.commit()
                cursor.close()
                # Retry
                cursor = connection.cursor()
                cursor.execute(check_query, (file_hash,))
                count = cursor.fetchone()[0]
                cursor.close()
                return (count > 0)
            except Exception as e2:
                print(f"[ERROR] Failed to alter imported_files table: {e2}")
                connection.rollback()
                return False

        else:
            # Some other error
            print(f"[ERROR] Error checking imported_files: {e}")
            connection.rollback()
            return False

def mark_file_as_imported(pg_conn, file_name, file_hash):
    """
    Marks a file as imported by inserting its hash into the imported_files table.
    Uses a transaction-safe INSERT ... ON CONFLICT.
    If an error occurs, logs it and rolls back so the connection can be reused.
    """
    try:
        with pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO imported_files (file_name, file_hash)
                VALUES (%s, %s)
                ON CONFLICT (file_hash) DO NOTHING;
            """, (file_name, file_hash))
        pg_conn.commit()
        #print(f"[INFO] Marked file '{file_name}' as imported.")
    except psycopg2.Error as e:
        pg_conn.rollback()  # IMPORTANT: recover connection
        print(f"[ERROR] Failed to mark file '{file_name}' as imported: {e}")

def get_all_user_tables(pg_conn):
    """
    Returns only tables where the name is 'R' followed by exactly 10 digits.
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


def connect_happy_path_db(creds):
    """
    Connects to the HAPPY_PATH database using keys:
      HAPPY_PATH_DB, HAPPY_PATH_USER, HAPPY_PATH_PASSWORD.
    Assumes the host is the same as DB_HOST.
    """
    try:
        conn = psycopg2.connect(
            host=creds["DB_HOST"],
            dbname=creds["HAPPY_PATH_DB"],
            user=creds["HAPPY_PATH_USER"],
            password=creds["HAPPY_PATH_PASSWORD"]
        )
        return conn
    except Exception as e:
        print(f"[HAPPY_PATH] Error connecting to happy path db '{creds['HAPPY_PATH_DB']}': {e}")
        return None