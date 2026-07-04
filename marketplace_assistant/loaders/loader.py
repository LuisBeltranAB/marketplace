import pandas as pd
from pathlib import Path
from marketplace_assistant.db import get_conn


def load_file_to_duckdb(path: str, table_name: str, conn=None):
    """Load a CSV or Excel file into DuckDB as `table_name`.

    Supports .csv, .xls, .xlsx. Uses pandas to read the file then writes the
    DataFrame into DuckDB. Returns True on success.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)

    close_after = False
    if conn is None:
        conn = get_conn()
        close_after = True

    conn.register("_tmp_df", df)
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _tmp_df")

    if close_after:
        conn.close()
    return True


def load_csv_to_duckdb(csv_path: str, table_name: str, conn=None):
    """Backward-compatible alias that calls `load_file_to_duckdb`."""
    return load_file_to_duckdb(csv_path, table_name, conn=conn)
