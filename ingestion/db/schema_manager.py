import os
import json
import psycopg2
from neo4j import GraphDatabase


class SchemaManager:
    """
    Manages schema creation and migrations for PostgreSQL and Neo4j.

    You can specify a target database by passing target_db. This allows you to
    create or update the schema in the HAPPY_PATH DB (or any alternate DB) instead
    of the default production database.
    """

    def __init__(self, pg_credentials, neo4j_credentials, target_db=None):
        if target_db:
            # Override the database field with the target database name.
            pg_credentials["database"] = target_db
        self.pg_conn = psycopg2.connect(**pg_credentials)
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_credentials["NEO4J_URI"],
            auth=(neo4j_credentials["NEO4J_USER"], neo4j_credentials["NEO4J_PASS"])
        )

    def create_postgres_tables(self):
        """Creates necessary PostgreSQL tables if they don't exist."""
        with self.pg_conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS log_entries (
                    id SERIAL PRIMARY KEY,
                    directory_file TEXT NOT NULL,
                    category TEXT,
                    timestamp TIMESTAMP WITH TIME ZONE,
                    file_line TEXT,
                    function TEXT,
                    data TEXT,
                    data_hash TEXT UNIQUE,
                    message TEXT
                );
            """)
            self.pg_conn.commit()

    def create_neo4j_constraints(self):
        """Creates necessary constraints in Neo4j."""
        with self.neo4j_driver.session() as session:
            # Unique constraint for log lines.
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (l:LogLine) REQUIRE l.unique_key IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (p:Process) REQUIRE p.name IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (s:Session) REQUIRE s.sid IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (c:Connection) REQUIRE c.cid IS UNIQUE
            """)

    def close(self):
        """Closes database connections."""
        self.pg_conn.close()
        self.neo4j_driver.close()


if __name__ == "__main__":
    # Determine the project root (assumes schema_manager.py is in <project>/db/)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    credentials_file = os.path.join(base_dir, "credentials.txt")

    # Load credentials from the JSON file.
    with open(credentials_file, "r") as f:
        credentials = json.load(f)

    # Build PostgreSQL credentials for the default (production) database.
    pg_credentials = {
        "host": credentials["DB_HOST"],
        "database": credentials["DB_NAME"],
        "user": credentials["DB_USER"],
        "password": credentials["DB_PASSWORD"]
    }

    # Use the same credentials for Neo4j.
    neo4j_credentials = credentials

    # Create or update the production schema.
    print("[INFO] Creating production schema...")
    schema_manager = SchemaManager(pg_credentials, neo4j_credentials)
    schema_manager.create_postgres_tables()
    schema_manager.create_neo4j_constraints()
    schema_manager.close()

    # If HAPPY_PATH_DB is specified, create/update its schema as well.
    if "HAPPY_PATH_DB" in credentials:
        print("[INFO] Creating schema in HAPPY_PATH DB...")
        happy_pg_credentials = {
            "host": credentials["DB_HOST"],
            "database": credentials["HAPPY_PATH_DB"],
            "user": credentials["HAPPY_PATH_USER"],
            "password": credentials["HAPPY_PATH_PASSWORD"]
        }
        happy_schema_manager = SchemaManager(happy_pg_credentials, neo4j_credentials)
        happy_schema_manager.create_postgres_tables()
        happy_schema_manager.create_neo4j_constraints()
        happy_schema_manager.close()
