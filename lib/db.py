import sqlite3
from .config import DB_PATH

DDL = """
CREATE TABLE IF NOT EXISTS user_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id INTEGER,
    week INTEGER,
    age INTEGER,
    gender TEXT,
    region TEXT,
    partner_tier TEXT,
    worker_type TEXT,
    earnings REAL,
    jobs_completed INTEGER,
    repeat_client_share REAL,
    avg_job_value REAL,
    hours_worked INTEGER,
    dispute_count INTEGER,
    tips_share REAL,
    customer_rating REAL,
    cancellation_rate REAL,
    peak_hour_jobs_share REAL,
    transaction_count INTEGER,
    gap_days INTEGER,
    nova_prev_score REAL,
    nova_score REAL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(worker_id, week)
);
"""

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_database():
    with get_conn() as conn:
        conn.execute(DDL)
        conn.commit()

def get_previous_score(worker_id: int) -> float:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT nova_score FROM user_scores
            WHERE worker_id = ?
            ORDER BY week DESC, prediction_date DESC
            LIMIT 1
            """,
            (worker_id,),
        )
        row = cur.fetchone()
        return row[0] if row else 60.0
