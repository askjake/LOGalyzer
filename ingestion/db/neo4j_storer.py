import shutil
from neo4j import GraphDatabase
import time  # For sleep delays

class Neo4jStorer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        #print("[NEO4J] Driver initialized.")

    def close(self):
        self.driver.close()
        #print("[NEO4J] Driver closed.")

    def store_log_line(self, log_data):
        """
        Stores a single log entry in Neo4j.
        (This method is retained for backwards compatibility. Batch insertion is preferred.)
        """
        try:
            with self.driver.session() as session:
                session.execute_write(self._create_nodes_and_relationships, log_data)
            #print("\n[NEO4J] Log line stored.",end=" ")
        except Exception as e:
            print(f"[NEO4J ERROR] store_log_line: {e}")

    def store_log_lines(self, log_data_list, batch_size=1000, stagger_delay=0.015):
        """
        Stores log lines in batches to improve efficiency.
        Prints a progress update for each batch.
        """
        if not isinstance(log_data_list, list):
            raise TypeError(f"Expected list of dictionaries, but got {type(log_data_list)}")

        total = len(log_data_list)
        total_batches = (total + batch_size - 1) // batch_size
        print(f"[NEO4J] Inserting {total} log lines in {total_batches} batch(es).")

        # Pre-compute unique keys and set default category if missing.
        for log in log_data_list:
            if not isinstance(log, dict):
                raise TypeError("Expected dictionary in list")
            if not log.get("category"):
                print("[NEO4J WARN] Log entry missing category; defaulting to 'UNKNOWN'")
                log["category"] = "UNKNOWN"
            timestamp = log.get("timestamp", "")
            category = log.get("category", "UNKNOWN")
            data = log.get("data", "")
            log["unique_key"] = f"{timestamp}_{category}_{hash(data)}"

        for i in range(0, total, batch_size):
            batch = log_data_list[i:i + batch_size]
            time.sleep(stagger_delay)  # Stagger batch start
            try:
                with self.driver.session() as session:
                    session.execute_write(self._create_nodes_and_relationships_batch, batch)
                print(f"[NEO4J] Batch {i // batch_size + 1}/{total_batches} inserted ({len(batch)} logs).")
            except Exception as e:
                err_str = str(e)
                if "duplicate key value violates unique constraint" in err_str:
                    print(f"[NEO4J WARN] Batch {i // batch_size + 1} duplicate key error; skipping batch.")
                else:
                    print(f"[NEO4J ERROR] Batch {i // batch_size + 1} failed: {e}")

    @staticmethod
    def _create_nodes_and_relationships(tx, log_data):
        """
        Stores a single log entry in Neo4j.
        Real-time updates have been removed for clarity.
        """
        category = log_data.get("category", "UNKNOWN")
        timestamp = log_data.get("timestamp", "")
        data = log_data.get("data", "")
        event_type = log_data.get("event_type", "")
        sid = log_data.get("sid", None)
        cid = log_data.get("cid", None)

        unique_key = f"{timestamp}_{category}_{hash(data)}"

        query1 = """
        MERGE (log:LogLine {unique_key: $uniqueKey})
        ON CREATE SET log.timestamp = $timestamp, log.data = $data, log.event_type = $eventType
        """
        tx.run(query1, uniqueKey=unique_key, timestamp=timestamp, data=data, eventType=event_type)

        query2 = "MERGE (p:Process {name: $category})"
        tx.run(query2, category=category)

        query3 = """
        MATCH (p:Process {name: $category}), (log:LogLine {unique_key: $uniqueKey})
        MERGE (p)-[:GENERATED]->(log)
        """
        tx.run(query3, category=category, uniqueKey=unique_key)

        if sid:
            tx.run("MERGE (s:Session {sid: $sid})", sid=sid)
            tx.run("""
            MATCH (s:Session {sid: $sid}), (log:LogLine {unique_key: $uniqueKey})
            MERGE (log)-[:PART_OF_SESSION]->(s)
            """, sid=sid, uniqueKey=unique_key)
        if cid:
            tx.run("MERGE (c:Connection {cid: $cid})", cid=cid)
            tx.run("""
            MATCH (c:Connection {cid: $cid}), (log:LogLine {unique_key: $uniqueKey})
            MERGE (log)-[:ASSOCIATED_WITH]->(c)
            """, cid=cid, uniqueKey=unique_key)

    @staticmethod
    def _create_nodes_and_relationships_batch(tx, batch):
        """
        Batch version using UNWIND to process multiple log entries.
        Each log in the batch must have a pre-computed "unique_key".
        """
        print(f"[NEO4J] Inserting batch of {len(batch)} logs...", end=' ')
        query = """
        UNWIND $batch AS log
        MERGE (l:LogLine {unique_key: log.unique_key})
        ON CREATE SET l.timestamp = log.timestamp, l.data = log.data, l.event_type = log.event_type
        MERGE (p:Process {name: log.category})
        MERGE (p)-[:GENERATED]->(l)
        FOREACH (_ IN CASE WHEN log.sid IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s:Session {sid: log.sid})
            MERGE (l)-[:PART_OF_SESSION]->(s)
        )
        FOREACH (_ IN CASE WHEN log.cid IS NOT NULL THEN [1] ELSE [] END |
            MERGE (c:Connection {cid: log.cid})
            MERGE (l)-[:ASSOCIATED_WITH]->(c)
        )
        """
        tx.run(query, batch=batch)
        print("Done.")
