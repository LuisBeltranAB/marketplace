from pathlib import Path
import pandas as pd
from marketplace_assistant.db import get_conn


def _find_header_row(df: pd.DataFrame, keywords):
    for i, row in df.iterrows():
        for v in row.values:
            try:
                if isinstance(v, str) and any(k.lower() in v.lower() for k in keywords):
                    return i
            except Exception:
                continue
    return None


def normalize_fixed_capacity(path: str, quarter: str = None, overwrite: bool = False):
    p = Path(path)
    df = pd.read_excel(p)
    hdr = _find_header_row(df, ['Units/Day', 'Units/Quarter', 'Capital Investment'])
    if hdr is not None:
        df2 = df.iloc[hdr:]
        df2.columns = df2.iloc[0].ffill()
        df2 = df2.iloc[1:]
    else:
        df2 = df.copy()

    df2 = df2.rename(columns=lambda c: str(c).strip())
    cols = list(df2.columns)
    # try to pick numeric columns
    # expected columns: Selected, Units/Day, Units/Quarter, Capital Investment, Capital Costs/Unit
    if len(cols) >= 5:
        sel, units_day, units_q, cap_inv, cap_cost = cols[0:5]
        out = df2[[sel, units_day, units_q, cap_inv, cap_cost]].dropna(how='all')
        out = out.rename(columns={sel: 'selected', units_day: 'units_per_day', units_q: 'units_per_quarter', cap_inv: 'capital_investment', cap_cost: 'capital_costs_per_unit'})
        # coerce numeric
        for c in ['units_per_day', 'units_per_quarter', 'capital_investment', 'capital_costs_per_unit']:
            out[c] = pd.to_numeric(out[c], errors='coerce')
        out['quarter'] = quarter
    else:
        out = pd.DataFrame()

    conn = get_conn()
    conn.register('_tmp_df', out)
    table = 'fixed_capacity'
    if overwrite:
        conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp_df")
    else:
        try:
            conn.execute(f"INSERT INTO {table} SELECT * FROM _tmp_df")
        except Exception:
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _tmp_df")
    conn.close()
    return out


def normalize_stock_history(path: str, quarter: str = None, overwrite: bool = False):
    p = Path(path)
    df = pd.read_excel(p)
    hdr = _find_header_row(df, ['Stock Type', 'Shares', 'Price Per Share', 'Total Amount'])
    if hdr is not None:
        df2 = df.iloc[hdr:]
        df2.columns = df2.iloc[0].ffill()
        df2 = df2.iloc[1:]
    else:
        df2 = df.copy()

    df2 = df2.rename(columns=lambda c: str(c).strip())
    # try to find cols
    cols = list(df2.columns)
    # find numeric-ish columns
    possible_shares = [c for c in cols if 'share' in c.lower() and 'price' not in c.lower()]
    possible_price = [c for c in cols if 'price' in c.lower()]
    possible_total = [c for c in cols if 'total' in c.lower()]
    owner_cols = [c for c in cols if 'owner' in c.lower() or 'name' in c.lower()]
    if possible_shares and possible_price:
        shares = possible_shares[0]
        price = possible_price[0]
        total = possible_total[0] if possible_total else None
        owner = owner_cols[0] if owner_cols else None
        sel_cols = [c for c in [owner, shares, price, total, 'Quarter'] if c and c in df2.columns]
        out = df2[sel_cols].dropna(how='all')
        # coerce numeric
        if shares in out.columns:
            out[shares] = pd.to_numeric(out[shares], errors='coerce')
        if price in out.columns:
            out[price] = pd.to_numeric(out[price], errors='coerce')
        out['quarter'] = out['Quarter'] if 'Quarter' in out.columns else quarter
        out = out.rename(columns={shares: 'shares', price: 'price_per_share', total: 'total_amount', owner: 'owner'})
    else:
        out = pd.DataFrame()

    conn = get_conn()
    conn.register('_tmp_df', out)
    table = 'stock_history'
    if overwrite:
        conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp_df")
    else:
        try:
            conn.execute(f"INSERT INTO {table} SELECT * FROM _tmp_df")
        except Exception:
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _tmp_df")
    conn.close()
    return out
