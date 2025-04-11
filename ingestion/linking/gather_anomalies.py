#!/usr/bin/env python3

## ingestion/linking/gather_anomalies.py
import psycopg2
import os
import json
import sys
import argparse


def main():
    """
    Main entry point: parse args, then run gather_anomalies for either
    a specific table_name or all tables.
    """
    parser = argparse.ArgumentParser(
        description="Gather anomalies from R-tables into the 'anomalies' table."
    )
    parser.add_argument("--table_name", type=str,
                        help="If set, only gather anomalies from this single R########## table.")
    args = parser.parse_args()

    gather_anomalies(args.table_name)


def gather_anomalies(single_table=None):
    """
    Connect to PostgreSQL and run a DO block that:
      1. Creates any missing columns in the anomalies table based on each R########## table.
      2. Dynamically builds a filter condition that considers any column whose name ends with
         '_is_anomaly' (checks for TRUE) or contains 'score' (checks for > 0) plus explicit conditions
         for 'customer_marked_flag' and 'important_investigate'.
      3. Inserts records from each table that satisfy these dynamic conditions.

      If single_table is provided, only that table is processed.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        creds_path = os.path.join(base_dir, "credentials.txt")

        if not os.path.exists(creds_path):
            print(f"[ERR] credentials.txt not found: {creds_path}")
            return

        with open(creds_path, "r") as f:
            j = json.load(f)

        db_host = j.get("DB_HOST")
        db_name = j.get("DB_NAME")
        db_user = j.get("DB_USER")
        db_pass = j.get("DB_PASSWORD")

        try:
            pg_conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_pass)
        except Exception as e:
            print(f"[ERR] logs DB connect fail: {e}")
            return

        pg_conn.autocommit = True
        cur = pg_conn.cursor()

        single_table_clause = ""
        if single_table:
            single_table_clause = f"AND tablename = '{single_table}'"
            print(f"[DEBUG] Single table mode enabled for table: {single_table}")

        # The DO block now builds its filter conditions dynamically.
        sql_block = rf"""
        DO $$
        DECLARE
            tbl record;
            column_list text;
            existing_columns text[];
            new_columns text[];
            new_col text;
            col_def text;
            filter_conditions text := '';
            anomaly_count integer;
            dyn_rec record;
        BEGIN
            -- Fetch existing columns in the anomalies table.
            SELECT array_agg(c.column_name::text)
            INTO existing_columns
            FROM information_schema.columns c
            WHERE c.table_name = 'anomalies';
            RAISE NOTICE 'Existing columns in anomalies: %', existing_columns;

            -- Iterate through matching tables.
            FOR tbl IN
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname = 'public'
                  AND tablename ~ '^R[0-9]{10}'
                  {single_table_clause}
            LOOP
                RAISE NOTICE 'Processing table: %', tbl.tablename;

                -- Identify and add any missing columns from the source table to anomalies.
                SELECT array_agg(c.column_name::text)
                INTO new_columns
                FROM information_schema.columns c
                WHERE c.table_name = tbl.tablename
                  AND c.column_name NOT IN (SELECT unnest(existing_columns));
                RAISE NOTICE 'New columns from table %: %', tbl.tablename, new_columns;
                IF new_columns IS NOT NULL THEN
                    FOR new_col IN SELECT unnest(new_columns) LOOP
                        SELECT c.data_type ||
                               CASE WHEN c.character_maximum_length IS NOT NULL
                                    THEN '(' || c.character_maximum_length || ')'
                                    ELSE ''
                               END
                        INTO col_def
                        FROM information_schema.columns c
                        WHERE c.table_name = tbl.tablename
                          AND c.column_name = new_col;
                        RAISE NOTICE 'Adding column % with type % to anomalies', new_col, col_def;
                        EXECUTE format('ALTER TABLE anomalies ADD COLUMN IF NOT EXISTS %I %s', new_col, col_def);
                    END LOOP;
                END IF;

                -- Refresh the list of existing columns.
                SELECT array_agg(c.column_name::text)
                INTO existing_columns
                FROM information_schema.columns c
                WHERE c.table_name = 'anomalies';
                RAISE NOTICE 'Updated existing columns in anomalies: %', existing_columns;

                -- Generate a column list from the source table.
                SELECT string_agg(quote_ident(c.column_name), ', ')
                INTO column_list
                FROM information_schema.columns c
                WHERE c.table_name = tbl.tablename;
                RAISE NOTICE 'Column list for table %: %', tbl.tablename, column_list;

                -- Dynamically build anomaly filter conditions.
                filter_conditions := '';
                FOR dyn_rec IN
                  SELECT column_name FROM information_schema.columns
                  WHERE table_name = tbl.tablename
                    AND (
                          column_name ILIKE '%_is_anomaly'
                          OR (column_name ILIKE '%score%' AND column_name NOT ILIKE '%_is_anomaly')
                        )
                LOOP
                    IF filter_conditions <> '' THEN
                        filter_conditions := filter_conditions || ' OR ';
                    END IF;
                    IF dyn_rec.column_name ILIKE '%_is_anomaly' THEN
                        filter_conditions := filter_conditions || format('%I = TRUE', dyn_rec.column_name);
                    ELSE
                        filter_conditions := filter_conditions || format('%I > 0', dyn_rec.column_name);
                    END IF;
                END LOOP;
                -- Explicitly include customer_marked_flag and important_investigate if they exist.
                IF EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_name = tbl.tablename AND column_name = 'customer_marked_flag') THEN
                    IF filter_conditions <> '' THEN
                        filter_conditions := filter_conditions || ' OR ';
                    END IF;
                    filter_conditions := filter_conditions || 'COALESCE(NULLIF(customer_marked_flag, ''''), ''false'')::BOOLEAN = TRUE';
                END IF;
                IF EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_name = tbl.tablename AND column_name = 'important_investigate') THEN
                    IF filter_conditions <> '' THEN
                        filter_conditions := filter_conditions || ' OR ';
                    END IF;
                    filter_conditions := filter_conditions || 'COALESCE(NULLIF(important_investigate::text, ''''), ''false'')::BOOLEAN = TRUE';
                END IF;
                IF filter_conditions IS NULL OR filter_conditions = '' THEN
                    filter_conditions := 'FALSE';
                END IF;
                RAISE NOTICE 'Filter conditions for table %: %', tbl.tablename, filter_conditions;

                -- Count anomalies in the current table.
                EXECUTE format('SELECT count(*) FROM %I WHERE %s', tbl.tablename, filter_conditions)
                INTO anomaly_count;
                RAISE NOTICE 'Found % anomalies in table %', anomaly_count, tbl.tablename;

                IF anomaly_count > 0 THEN
                    EXECUTE format(
                        'INSERT INTO anomalies (table_name, %s)
                         SELECT %L, %s
                         FROM %I
                         WHERE %s
                         ORDER BY "timestamp" DESC',
                        column_list, tbl.tablename, column_list, tbl.tablename, filter_conditions
                    );
                    RAISE NOTICE 'Inserted anomalies from table %', tbl.tablename;
                ELSE
                    RAISE NOTICE 'No anomalies to insert from table %', tbl.tablename;
                END IF;
            END LOOP;
        END $$;
        """

        print("[DEBUG] Executing dynamic DO block to gather anomalies...")
        cur.execute(sql_block)
        print("[DEBUG] DO block executed successfully.")
        cur.close()
        pg_conn.close()
        print("Anomalies gathered and inserted into the 'anomalies' table successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
