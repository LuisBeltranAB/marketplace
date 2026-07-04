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


def normalize_customer_needs(path: str, quarter: str = None, overwrite: bool = False):
    p = Path(path)
    df = pd.read_excel(p)
    # find header row that contains 'Need' or 'Application'
    hdr = _find_header_row(df, ['need', 'application'])
    if hdr is None:
        # fallback: assume first row is header
        df2 = df.copy()
    else:
        df2 = df.iloc[hdr:]
        df2.columns = df2.iloc[0].ffill()
        df2 = df2.iloc[1:]

    df2 = df2.rename(columns=lambda c: str(c).strip())
    # first column is the question
    first_col = df2.columns[0]
    seg_cols = [c for c in df2.columns[1:] if c and 'unnamed' not in str(c).lower()]
    tidied = df2[[first_col] + seg_cols].melt(id_vars=first_col, value_name='score', var_name='segment')
    tidied = tidied.dropna(subset=[first_col])
    tidied['quarter'] = quarter

    # write to DuckDB
    conn = get_conn()
    conn.register('_tmp_df', tidied)
    table = 'customer_needs'
    if overwrite:
        conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp_df")
    else:
        try:
            conn.execute(f"INSERT INTO {table} SELECT * FROM _tmp_df")
        except Exception:
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _tmp_df")
    conn.close()
    return tidied


def normalize_use_pattern(path: str, quarter: str = None, overwrite: bool = False):
    # Similar structure to customer needs
    p = Path(path)
    df = pd.read_excel(p)
    hdr = _find_header_row(df, ['application'])
    if hdr is None:
        df2 = df.copy()
    else:
        df2 = df.iloc[hdr:]
        df2.columns = df2.iloc[0].ffill()
        df2 = df2.iloc[1:]

    df2 = df2.rename(columns=lambda c: str(c).strip())
    first_col = df2.columns[0]
    seg_cols = [c for c in df2.columns[1:] if c and 'unnamed' not in str(c).lower()]
    tidied = df2[[first_col] + seg_cols].melt(id_vars=first_col, value_name='score', var_name='segment')
    tidied = tidied.dropna(subset=[first_col])
    tidied['quarter'] = quarter

    conn = get_conn()
    conn.register('_tmp_df', tidied)
    table = 'use_pattern'
    if overwrite:
        conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp_df")
    else:
        try:
            conn.execute(f"INSERT INTO {table} SELECT * FROM _tmp_df")
        except Exception:
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _tmp_df")
    conn.close()
    return tidied


def normalize_component_changeover(path: str, quarter: str = None, overwrite: bool = False):
    p = Path(path)
    df = pd.read_excel(p)
    hdr = _find_header_row(df, ['component group'])
    if hdr is not None:
        df2 = df.iloc[hdr:]
        df2.columns = df2.iloc[0].ffill()
        df2 = df2.iloc[1:]
    else:
        df2 = df.copy()

    df2 = df2.rename(columns=lambda c: str(c).strip())
    # expect columns like Component Group, Time to Change Components [hours], Direct Expenses to Change Components
    cols = list(df2.columns)
    if len(cols) >= 3:
        comp_col = cols[0]
        time_col = cols[1]
        cost_col = cols[2]
        out = df2[[comp_col, time_col, cost_col]].dropna()
        out = out.rename(columns={comp_col: 'component_group', time_col: 'time_hours', cost_col: 'direct_expense'})
        out['time_hours'] = pd.to_numeric(out['time_hours'], errors='coerce')
        out['direct_expense'] = pd.to_numeric(out['direct_expense'], errors='coerce')
        out['quarter'] = quarter
    else:
        out = pd.DataFrame()

    conn = get_conn()
    conn.register('_tmp_df', out)
    table = 'component_changeover'
    if overwrite:
        conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp_df")
    else:
        try:
            conn.execute(f"INSERT INTO {table} SELECT * FROM _tmp_df")
        except Exception:
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _tmp_df")
    conn.close()
    return out


def normalize_factory_location(path: str, quarter: str = None, overwrite: bool = False):
    p = Path(path)
    df = pd.read_excel(p)
    hdr = _find_header_row(df, ['region', 'production facility location'])
    if hdr is not None:
        df2 = df.iloc[hdr:]
        df2.columns = df2.iloc[0].ffill()
        df2 = df2.iloc[1:]
    else:
        df2 = df.copy()

    df2 = df2.rename(columns=lambda c: str(c).strip())
    # Build rows: pick first string as region, next string that's not 'Blank' as city, first numeric as cost
    rows = []
    for _, r in df2.iterrows():
        vals = list(r.values)
        # filter out NaN
        vals_clean = [v for v in vals if pd.notna(v)]
        if not vals_clean:
            continue
        # stop at 'End of Worksheet'
        if any(isinstance(v, str) and 'end of worksheet' in v.lower() for v in vals_clean):
            continue
        region = None
        city = None
        cost = None
        for v in vals_clean:
            if isinstance(v, str) and v.strip().lower() != 'blank' and region is None:
                region = v.strip()
                continue
            if isinstance(v, str) and v.strip().lower() != 'blank' and region is not None and city is None:
                city = v.strip()
                continue
            # numeric
            try:
                num = float(v)
                if cost is None:
                    cost = num
            except Exception:
                pass
        if region or city or cost:
            rows.append({'region': region, 'city': city, 'material_cost_index': cost, 'quarter': quarter})
    if rows:
        out = pd.DataFrame(rows)
        # drop header-like rows
        out = out[~out['region'].astype(str).str.strip().str.lower().isin(['region', ''])]
        out = out[~out['city'].astype(str).str.strip().str.lower().isin(['city', 'end of worksheet', ''])]
    else:
        out = pd.DataFrame()

    conn = get_conn()
    conn.register('_tmp_df', out)
    table = 'factory_location'
    if overwrite:
        conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp_df")
    else:
        try:
            conn.execute(f"INSERT INTO {table} SELECT * FROM _tmp_df")
        except Exception:
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _tmp_df")
    conn.close()
    return out


def normalize_open_sales_office(path: str, quarter: str = None, overwrite: bool = False):
    p = Path(path)
    xls = pd.ExcelFile(p)
    frames = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(p, sheet_name=sheet)
        hdr = _find_header_row(df, ['city', 'open', 'close', 'current status'])
        if hdr is not None:
            df2 = df.iloc[hdr:]
            df2.columns = df2.iloc[0].ffill()
            df2 = df2.iloc[1:]
        else:
            df2 = df.copy()
        df2 = df2.rename(columns=lambda c: str(c).strip())
        cols = list(df2.columns)
        # expect city, open, close, current status, setup/close cost, quarterly lease cost
        if len(cols) >= 6:
            city_col = cols[0]
            open_col = cols[1]
            close_col = cols[2]
            status_col = cols[3]
            setup_col = cols[4]
            lease_col = cols[5]
            out = df2[[city_col, open_col, close_col, status_col, setup_col, lease_col]].dropna(how='all')
            out = out.rename(columns={city_col: 'city', open_col: 'open', close_col: 'close', status_col: 'status', setup_col: 'setup_close_cost', lease_col: 'quarterly_lease_cost'})
            # drop rows like 'End of Worksheet'
            out = out[~out['city'].astype(str).str.contains('End of Worksheet', case=False, na=False)]
            # coerce numeric columns
            out['setup_close_cost'] = pd.to_numeric(out['setup_close_cost'], errors='coerce')
            out['quarterly_lease_cost'] = pd.to_numeric(out['quarterly_lease_cost'], errors='coerce')
            out['quarter'] = quarter
            out['region_sheet'] = sheet
            frames.append(out)
    if frames:
        full = pd.concat(frames, ignore_index=True)
    else:
        full = pd.DataFrame()

    conn = get_conn()
    conn.register('_tmp_df', full)
    table = 'open_sales_offices'
    if overwrite:
        conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp_df")
    else:
        try:
            conn.execute(f"INSERT INTO {table} SELECT * FROM _tmp_df")
        except Exception:
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _tmp_df")
    conn.close()
    return full
