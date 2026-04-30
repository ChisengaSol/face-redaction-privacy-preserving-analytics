# import sqlite3
# from datetime import datetime
# import os

# class AnalyticsDB:
#     def __init__(self, db_path="analytics.db"):
#         self.db_path = db_path
#         # check_same_thread=False is useful if we query from different threads later
#         self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
#         self._create_tables()

#     def _create_tables(self):
#         cursor = self.conn.cursor()
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS analytics_log (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 timestamp DATETIME,
#                 queue_length INTEGER,
#                 foot_traffic INTEGER,
#                 unique_visitors INTEGER
#             )
#         ''')
#         self.conn.commit()

#     def log_metrics(self, queue_length, foot_traffic, unique_visitors):
#         cursor = self.conn.cursor()
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         cursor.execute('''
#             INSERT INTO analytics_log (timestamp, queue_length, foot_traffic, unique_visitors)
#             VALUES (?, ?, ?, ?)
#         ''', (timestamp, queue_length, foot_traffic, unique_visitors))
#         self.conn.commit()

#     def close(self):
#         self.conn.close()

# # database.py
# import sqlite3
# from datetime import datetime
# import json

# class AnalyticsDB:
#     def __init__(self, db_path="analytics.db"):
#         self.db_path = db_path
#         self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
#         self._create_tables()

#     def _create_tables(self):
#         cursor = self.conn.cursor()
#         # Original analytics table
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS analytics_log (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 timestamp DATETIME,
#                 queue_length INTEGER,
#                 foot_traffic INTEGER,
#                 unique_visitors INTEGER
#             )
#         ''')
#         # NEW: Table for the vector embeddings
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS embeddings_log (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 timestamp DATETIME,
#                 person_id INTEGER UNIQUE,
#                 embedding TEXT
#             )
#         ''')
#         self.conn.commit()

#     def log_metrics(self, queue_length, foot_traffic, unique_visitors):
#         cursor = self.conn.cursor()
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         cursor.execute('''
#             INSERT INTO analytics_log (timestamp, queue_length, foot_traffic, unique_visitors)
#             VALUES (?, ?, ?, ?)
#         ''', (timestamp, queue_length, foot_traffic, unique_visitors))
#         self.conn.commit()

#     # NEW: Method to log the embedding
#     def log_embedding(self, person_id, embedding_list):
#         cursor = self.conn.cursor()
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
#         # Convert the python list into a JSON string for SQLite storage
#         embedding_json = json.dumps(embedding_list)
        
#         cursor.execute('''
#             INSERT OR REPLACE INTO embeddings_log (timestamp, person_id, embedding)
#             VALUES (?, ?, ?)
#         ''', (timestamp, person_id, embedding_json))
#         self.conn.commit()

#     def close(self):
#         self.conn.close()

import sqlite3
from datetime import datetime
import json
import os

class AnalyticsDB:
    def __init__(self, db_path="analytics.db"):
        self.db_path = db_path
        # check_same_thread=False is useful if we query from different threads later
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # Original analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                queue_length INTEGER,
                foot_traffic INTEGER,
                unique_visitors INTEGER
            )
        ''')
        
        # NEW: Table for the vector embeddings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                person_id INTEGER UNIQUE,
                embedding TEXT
            )
        ''')
        
        self.conn.commit()

    def log_metrics(self, queue_length, foot_traffic, unique_visitors):
        cursor = self.conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO analytics_log (timestamp, queue_length, foot_traffic, unique_visitors)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, queue_length, foot_traffic, unique_visitors))
        self.conn.commit()

    # NEW: Method to log the embedding
    def log_embedding(self, person_id, embedding_list):
        cursor = self.conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert the python list into a JSON string for SQLite storage
        embedding_json = json.dumps(embedding_list)
        
        cursor.execute('''
            INSERT OR REPLACE INTO embeddings_log (timestamp, person_id, embedding)
            VALUES (?, ?, ?)
        ''', (timestamp, person_id, embedding_json))
        self.conn.commit()

    def load_embeddings(self):
        """Return all stored embeddings as [(person_id, embedding_list), ...] for re-loading on startup."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT person_id, embedding FROM embeddings_log ORDER BY person_id")
        rows = cursor.fetchall()
        return [(person_id, json.loads(embedding_json)) for person_id, embedding_json in rows]

    def close(self):
        self.conn.close()