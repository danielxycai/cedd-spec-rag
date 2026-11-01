import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_PATH = Path("rag_monitor.db")

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Creates a database connection with row factory enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Path = DB_PATH) -> None:
    """Initializes the monitoring tables if they do not exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                retrieved_clauses TEXT,
                latency_seconds REAL,
                model_name TEXT,
                feedback INTEGER DEFAULT 0, -- +1 for upvote, -1 for downvote, 0 for neutral
                notes TEXT
            )
        """)
        conn.commit()

def log_interaction(
    question: str,
    answer: str,
    retrieved_clauses: List[str],
    latency_seconds: float,
    model_name: str = "nvidia/nemotron-3.5-lightning:free",
    db_path: Path = DB_PATH
) -> int:
    """Logs a single query-response event and returns the record ID."""
    init_db(db_path)
    clauses_str = ", ".join(retrieved_clauses) if retrieved_clauses else "None"
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO query_logs (question, answer, retrieved_clauses, latency_seconds, model_name)
            VALUES (?, ?, ?, ?, ?)
        """, (question, answer, clauses_str, round(latency_seconds, 3), model_name))
        conn.commit()
        return cursor.lastrowid

def update_feedback(log_id: int, feedback: int, notes: str = "", db_path: Path = DB_PATH) -> None:
    """Updates user feedback (+1 or -1) for a specific log record."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE query_logs
            SET feedback = ?, notes = ?
            WHERE id = ?
        """, (feedback, notes, log_id))
        conn.commit()

def get_recent_logs(limit: int = 50, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Fetches recent query logs for the analytics dashboard."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, question, answer, retrieved_clauses, latency_seconds, feedback
            FROM query_logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

def get_analytics_summary(db_path: Path = DB_PATH) -> Dict[str, Any]:
    """Computes basic performance metrics for monitoring."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM query_logs")
        total_queries = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(latency_seconds) FROM query_logs")
        avg_latency_val = cursor.fetchone()[0]
        avg_latency = round(avg_latency_val, 2) if avg_latency_val else 0.0

        cursor.execute("SELECT COUNT(*) FROM query_logs WHERE feedback = 1")
        positive_fb = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM query_logs WHERE feedback = -1")
        negative_fb = cursor.fetchone()[0]

        return {
            "total_queries": total_queries,
            "avg_latency": avg_latency,
            "positive_feedback": positive_fb,
            "negative_feedback": negative_fb
        }

if __name__ == "__main__":
    print("Initializing test database...")
    init_db()
    
    # Simulate a logged query
    test_id = log_interaction(
        question="What is the testing frequency for reinforcement?",
        answer="Testing requirements are defined under Clause 15.30...",
        retrieved_clauses=["Clause 15.30", "Clause 15.33A"],
        latency_seconds=1.42
    )
    print(f"Logged interaction with ID: {test_id}")
    
    # Simulate user upvote
    update_feedback(test_id, feedback=1, notes="Accurate clause cited.")
    print("Updated feedback to +1.")
    
    # Display analytics
    summary = get_analytics_summary()
    print(f"Analytics Summary: {summary}")