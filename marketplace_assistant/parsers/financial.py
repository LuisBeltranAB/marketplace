from typing import Optional
from pathlib import Path
import pandas as pd
from marketplace_assistant.db import get_conn


def _as_long_series(df: pd.DataFrame) -> Optional[pd.Series]:
    # Heuristic: two-column files where first column are labels and second column values
    if df.shape[1] >= 2:
        first = df.columns[0]
        second = df.columns[1]
        if df[first].dtype == object and df[second].dtype in (object, float, int):
            s = pd.Series(df[second].values, index=df[first].astype(str).str.strip())
            return s
    return None


def _pick_value(series: pd.Series, candidates):
    for c in candidates:
        for key in series.index:
            if key.strip().lower() == c:
                try:
                    return pd.to_numeric(series.loc[key], errors="coerce")
                except Exception:
                    return series.loc[key]
    # fuzzy match
    for key in series.index:
        k = key.strip().lower()
        for c in candidates:
            if c in k:
                return pd.to_numeric(series.loc[key], errors="coerce")
    return None


def normalize_income_statement(df: pd.DataFrame, quarter: Optional[str] = None) -> pd.DataFrame:
    s = _as_long_series(df)
    row = {"quarter": quarter}
    if s is not None:
        row["sales"] = _pick_value(s, ["sales", "revenue", "total revenue"])
        row["gross_profit"] = _pick_value(s, ["gross profit", "gross_profit"])
        row["operating_income"] = _pick_value(s, ["operating income", "ebit", "operating_income"])
        row["net_income"] = _pick_value(s, ["net income", "net_income", "net profit"]) 
        row["eps"] = _pick_value(s, ["eps", "earnings per share"])
    else:
        # wide format - try to find columns
        cols = {c.strip().lower(): c for c in df.columns}
        def find(cols_map, candidates):
            for c in candidates:
                if c in cols_map:
                    return df.iloc[0][cols_map[c]]
            for k, orig in cols_map.items():
                for c in candidates:
                    if c in k:
                        return df.iloc[0][orig]
            return None

        row["sales"] = find(cols, ["sales", "revenue"])
        row["gross_profit"] = find(cols, ["gross profit", "gross_profit"]) 
        row["operating_income"] = find(cols, ["operating income", "ebit"]) 
        row["net_income"] = find(cols, ["net income", "net_income"]) 
        row["eps"] = find(cols, ["eps", "earnings per share"]) 

    return pd.DataFrame([row])


def normalize_balance_sheet(df: pd.DataFrame, quarter: Optional[str] = None) -> pd.DataFrame:
    s = _as_long_series(df)
    row = {"quarter": quarter}
    if s is not None:
        row["cash"] = _pick_value(s, ["cash", "cash and cash equivalents"])
        row["inventory"] = _pick_value(s, ["inventory", "inventories"])
        row["ppe"] = _pick_value(s, ["ppe", "property plant and equipment", "fixed assets"])
        row["debt"] = _pick_value(s, ["debt", "total debt", "long-term debt", "short-term debt"]) 
        row["equity"] = _pick_value(s, ["equity", "shareholders' equity", "stockholders' equity"]) 
    else:
        cols = {c.strip().lower(): c for c in df.columns}
        def find(cols_map, candidates):
            for c in candidates:
                if c in cols_map:
                    return df.iloc[0][cols_map[c]]
            for k, orig in cols_map.items():
                for c in candidates:
                    if c in k:
                        return df.iloc[0][orig]
            return None

        row["cash"] = find(cols, ["cash", "cash and cash equivalents"]) 
        row["inventory"] = find(cols, ["inventory"]) 
        row["ppe"] = find(cols, ["ppe", "property plant and equipment"]) 
        row["debt"] = find(cols, ["debt", "total debt"]) 
        row["equity"] = find(cols, ["equity"]) 

    return pd.DataFrame([row])


def normalize_cash_flow(df: pd.DataFrame, quarter: Optional[str] = None) -> pd.DataFrame:
    s = _as_long_series(df)
    row = {"quarter": quarter}
    if s is not None:
        row["operating_cash"] = _pick_value(s, ["cash from operating activities", "operating cash flow", "cash flow from operations"]) 
        row["investing_cash"] = _pick_value(s, ["cash from investing activities", "investing cash flow"]) 
        row["financing_cash"] = _pick_value(s, ["cash from financing activities", "financing cash flow"]) 
        row["net_change_cash"] = _pick_value(s, ["net change in cash", "net increase (decrease) in cash"]) 
    else:
        cols = {c.strip().lower(): c for c in df.columns}
        def find(cols_map, candidates):
            for c in candidates:
                if c in cols_map:
                    return df.iloc[0][cols_map[c]]
            for k, orig in cols_map.items():
                for c in candidates:
                    if c in k:
                        return df.iloc[0][orig]
            return None

        row["operating_cash"] = find(cols, ["operating cash flow", "cash from operating activities"]) 
        row["investing_cash"] = find(cols, ["investing cash flow"]) 
        row["financing_cash"] = find(cols, ["financing cash flow"]) 
        row["net_change_cash"] = find(cols, ["net change in cash"]) 

    return pd.DataFrame([row])


def import_and_normalize_csv(path: str, kind: str = "income_statement", quarter: Optional[str] = None, overwrite: bool = False):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)

    kind = kind.lower()
    if kind in ("income_statement", "income", "is"):
        norm = normalize_income_statement(df, quarter=quarter)
    elif kind in ("balance_sheet", "balance", "bs"):
        norm = normalize_balance_sheet(df, quarter=quarter)
    elif kind in ("cash_flow", "cashflow", "cf"):
        norm = normalize_cash_flow(df, quarter=quarter)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    conn = get_conn()
    conn.register("_tmp_df", norm)
    table_name = kind
    if overwrite:
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _tmp_df")
    else:
        # Try to append, otherwise create
        try:
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM _tmp_df")
        except Exception:
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _tmp_df")

    conn.close()
    return norm
