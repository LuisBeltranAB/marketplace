from pathlib import Path
import duckdb

DB_PATH = Path("data/processed/marketplace.duckdb")

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(database=str(DB_PATH), read_only=False)
    return conn

def close_conn(conn):
    try:
        conn.close()
    except Exception:
        pass
