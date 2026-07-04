import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from marketplace_assistant.db import get_conn
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform


CITY_COORDS = {
    'los angeles': (34.0522, -118.2437),
    'chicago': (41.8781, -87.6298),
    'toronto': (43.6532, -79.3832),
    'mexico city': (19.4326, -99.1332),
    'abu dhabi': (24.4539, 54.3773),
    'accra': (5.6037, -0.1870),
    'cairo': (30.0444, 31.2357),
    'johannesburg': (-26.2041, 28.0473),
    'santo domingo': (18.4861, -69.9312),
    'santiago': (-33.4489, -70.6693),
    'sao paulo': (-23.5505, -46.6333),
    'quito': (-0.1807, -78.4678),
    'paris': (48.8566, 2.3522),
    'warsaw': (52.2297, 21.0122),
    'moscow': (55.7558, 37.6173),
    'london': (51.5074, -0.1278),
    'tokyo': (35.6762, 139.6503),
    'mumbai': (19.0760, 72.8777),
    'shanghai': (31.2304, 121.4737),
    'sydney': (-33.8688, 151.2093),
}


def get_city_coords(city: str):
    if not isinstance(city, str):
        return None
    return CITY_COORDS.get(city.strip().lower())


def plot_city_map(df: pd.DataFrame, city_col: str, region_col: str | None = None, title: str = ""):
    cleaned = df.copy()
    cleaned[city_col] = cleaned[city_col].astype(str).str.strip()
    cleaned = cleaned[~cleaned[city_col].str.contains('end of worksheet', case=False, na=False)]
    coords = cleaned[city_col].apply(get_city_coords)
    kept = coords.notna()
    cleaned = cleaned[kept].copy()
    if cleaned.empty:
        return None
    cleaned[['lat', 'lon']] = pd.DataFrame(coords[kept].tolist(), index=cleaned.index)
    color = region_col if region_col and region_col in cleaned.columns else city_col
    fig = px.scatter_geo(
        cleaned,
        lat='lat',
        lon='lon',
        color=color,
        hover_name=city_col,
        hover_data={city_col: True, 'lat': False, 'lon': False},
        projection='natural earth',
        title=title,
    )
    fig.update_traces(marker=dict(size=10))
    return fig


def load_table(conn, table_name: str) -> pd.DataFrame:
    try:
        df = conn.execute(f"SELECT * FROM {table_name} ORDER BY quarter").df()
        return df
    except Exception:
        return pd.DataFrame()


def build_segment_similarity(cust_df: pd.DataFrame):
    if cust_df.empty:
        return None, None, None

    numeric_df = cust_df.copy()
    numeric_df['score'] = pd.to_numeric(numeric_df['score'], errors='coerce')
    numeric_df = numeric_df.dropna(subset=['score'])

    try:
        pivot = numeric_df.pivot_table(index='Need', columns='segment', values='score', aggfunc='mean').fillna(0)
    except Exception:
        return None, None, None

    if pivot.empty or pivot.shape[1] < 2:
        return None, None, None

    segment_vectors = pivot.T.astype(float)
    distance_matrix = pd.DataFrame(squareform(pdist(segment_vectors, metric='euclidean')), index=segment_vectors.index, columns=segment_vectors.index)
    linkage_matrix = linkage(segment_vectors, method='average')

    ranked_pairs = []
    for i in range(len(distance_matrix.columns)):
        for j in range(i + 1, len(distance_matrix.columns)):
            a = distance_matrix.columns[i]
            b = distance_matrix.columns[j]
            ranked_pairs.append((a, b, float(distance_matrix.loc[a, b])))

    ranked_pairs.sort(key=lambda x: x[2])
    return distance_matrix, linkage_matrix, ranked_pairs


def plot_dendrogram(linkage_matrix, labels):
    den = dendrogram(linkage_matrix, labels=labels, orientation='top', no_labels=False, no_plot=True)
    fig = go.Figure()
    for x, y in zip(den['icoord'], den['dcoord']):
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='steelblue'), hoverinfo='skip'))
    fig.update_layout(
        title='Cluster Dendrogram',
        xaxis_title='Segments',
        yaxis_title='Distance',
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=80),
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def page_marketing():
    st.header("Marketing — Segment Strategy")
    st.caption("Why Workhorse + Traveler · Q2 research data")

    SEGMENTS = ["Costcutter", "Innovator", "Mercedes", "Workhorse", "Traveler"]
    CHOSEN    = ["Workhorse", "Traveler"]
    COLORS    = {
        "Costcutter": "#aec7e8", "Innovator": "#ffbb78",
        "Mercedes": "#98df8a",   "Workhorse": "#1f77b4", "Traveler": "#ff7f0e",
    }

    # ── Raw data ──────────────────────────────────────────────────────────────
    demand_raw = {
        "City":        ["Los Angeles","Chicago","Toronto","Mexico City","Abu Dhabi","Accra","Cairo","Johannesburg","Santo Domingo","Santiago","Sao Paulo","Quito","Paris","Warsaw","Moscow","London","Tokyo","Mumbai","Shanghai","Sydney"],
        "Region":      ["NORAM","NORAM","NORAM","NORAM","MEA","MEA","MEA","MEA","LATAM","LATAM","LATAM","LATAM","EUROPE","EUROPE","EUROPE","EUROPE","APAC","APAC","APAC","APAC"],
        "Costcutter":  [3957,3364,4009,2412,2719,1283,1831,2283,1804,2204,2962,1448,4140,2398,3107,3371,2717,2421,2802,2760],
        "Innovator":   [3342,3354,2060,1694,1351,375,909,1073,657,835,1393,536,2456,1527,2356,2495,3198,1525,2335,2363],
        "Mercedes":    [2844,2913,2645,1460,1689,354,824,1108,798,801,1629,648,2702,1516,2123,2824,2672,1299,1273,1906],
        "Workhorse":   [4355,3949,4139,2677,2982,1226,1625,2280,2118,2727,3252,1317,4377,2525,3482,4718,3036,2488,2206,3164],
        "Traveler":    [3205,2873,2572,1610,1740,424,899,1287,725,948,1634,536,2967,1502,2156,2776,3450,1364,2201,1509],
    }
    demand = pd.DataFrame(demand_raw)

    price_raw = {
        "City":        ["Los Angeles","Chicago","Toronto","Mexico City","Abu Dhabi","Accra","Cairo","Johannesburg","Santo Domingo","Santiago","Sao Paulo","Quito","Paris","Warsaw","Moscow","London","Tokyo","Mumbai","Shanghai","Sydney"],
        "Costcutter":  [2280.68,2183.47,2219.84,2257.45,1937.40,1930.84,2079.60,2074.01,1858.24,1968.45,1860.40,1864.84,2167.01,2168.13,2099.49,2188.95,2093.43,2092.25,2166.29,2195.55],
        "Innovator":   [3835.30,3948.04,3652.88,3627.13,3367.14,3571.67,3609.37,3660.80,3365.87,3513.79,3382.41,3556.28,3621.89,3850.31,3576.98,3571.50,3461.67,3545.03,3445.23,3739.26],
        "Mercedes":    [5089.38,5230.30,5145.75,4952.64,4485.65,4309.00,4583.87,4513.22,4327.46,4346.04,4427.52,4293.36,5004.84,5057.93,4690.35,4678.61,4892.28,4895.63,4498.03,4484.83],
        "Workhorse":   [3333.72,3112.08,3345.69,3234.20,2690.17,2928.12,2809.87,2899.39,2673.32,2721.93,2883.97,2781.27,2959.12,3233.72,3049.45,3214.86,2863.45,3078.40,2995.64,3076.12],
        "Traveler":    [3470.40,3706.02,3592.28,3763.82,3398.29,3151.87,3361.74,3194.58,3147.21,3168.05,3230.91,3342.65,3628.11,3575.03,3407.73,3470.26,3554.97,3510.43,3266.07,3492.74],
    }
    price = pd.DataFrame(price_raw)

    needs_raw = {
        "Need": [
            "Portability", "LAN connectivity", "Can use on road", "Monitor easy on eyes",
            "Reliability", "Slim design", "Courteous service", "High performance over price",
            "Multiple programs", "Easy to set up", "Large-scale tasks", "Customizable",
            "Fast processing", "Fast graphics", "Light weight", "Lowest price",
            "Email capable", "Graphical detail", "Easy to use", "Live video",
            "Financial stability", "Service everywhere", "Ultra fast", "Storage space",
            "Competent sales", "Quick response", "Safe/secure", "Available locally",
        ],
        "Costcutter": [54,103,48,102,114,69,108,98,86,125,59,99,95,69,63,123,106,87,127,50,102,99,95,100,115,86,132,100],
        "Innovator":  [100,123,76,109,115,113,108,125,123,93,117,121,124,129,105,93,113,112,101,117,102,102,122,117,83,120,114,92],
        "Mercedes":   [76,129,94,116,123,104,117,127,127,105,125,111,128,130,101,89,114,129,99,111,124,122,129,121,114,126,109,113],
        "Workhorse":  [83,115,85,120,119,107,112,111,101,114,107,107,109,109,94,120,109,92,122,116,109,115,101,112,105,104,124,118],
        "Traveler":   [123,110,126,121,122,121,112,108,99,113,95,106,110,102,114,107,126,84,104,118,103,110,98,118,98,107,119,107],
    }
    needs = pd.DataFrame(needs_raw)

    use_raw = {
        "Application": [
            "Communications","Presentations","Word Processing","Data management",
            "Engineering/design","Bookkeeping/budgeting","Manufacturing control",
            "Technical graphics","CAD/CAM","Business graphics","Statistical analysis",
            "Web design","",
        ],
        "Costcutter": [112,103,125,109,73,114,69,70,38,93,59,69,0],
        "Innovator":  [121,109,99,116,116,100,102,125,104,108,124,125,0],
        "Mercedes":   [121,111,105,113,105,84,132,120,120,102,111,88,0],
        "Workhorse":  [120,108,126,119,75,126,88,72,64,110,90,115,0],
        "Traveler":   [124,131,122,109,89,101,72,63,33,114,68,58,0],
    }
    use_df = pd.DataFrame(use_raw)
    use_df = use_df[use_df["Application"] != ""]

    # ── Derived: revenue potential ────────────────────────────────────────────
    rev_rows = []
    for seg in SEGMENTS:
        total_demand = demand[seg].sum()
        avg_price    = price[seg].mean()
        rev_rows.append(dict(Segment=seg, Demand=total_demand, Avg_Price=avg_price,
                             Revenue_Potential=total_demand * avg_price,
                             Chosen=seg in CHOSEN))
    rev_df = pd.DataFrame(rev_rows).sort_values("Revenue_Potential", ascending=False)

    # ── Section 1: Decision rationale KPIs ───────────────────────────────────
    st.subheader("Why Workhorse + Traveler?")
    c1, c2, c3, c4 = st.columns(4)
    wh = rev_df[rev_df.Segment == "Workhorse"].iloc[0]
    tr = rev_df[rev_df.Segment == "Traveler"].iloc[0]
    c1.metric("Workhorse global demand", f"{int(wh.Demand):,} units", "Largest segment worldwide")
    c2.metric("Workhorse revenue potential", f"${wh.Revenue_Potential/1e6:.1f}M", "#1 across all segments")
    c3.metric("Traveler global demand", f"{int(tr.Demand):,} units", "#3 globally")
    c4.metric("Traveler revenue potential", f"${tr.Revenue_Potential/1e6:.1f}M")

    combined_rev = wh.Revenue_Potential + tr.Revenue_Potential
    cc = rev_df[rev_df.Segment == "Costcutter"].iloc[0]
    st.info(
        f"**Combined WH + Traveler revenue potential: ${combined_rev/1e6:.1f}M** — "
        f"more than any other two-segment combo that shares design overlap. "
        f"Costcutter (${cc.Revenue_Potential/1e6:.1f}M potential) was rejected due to its "
        f"lowest price ceiling (avg ${cc.Avg_Price:,.0f}) leaving almost no gross margin."
    )

    # ── Section 2: Segment comparison — demand + price ────────────────────────
    st.markdown("---")
    st.subheader("All Segments: Demand vs Revenue Potential")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_demand = px.bar(
            rev_df.sort_values("Demand", ascending=True),
            x="Demand", y="Segment", orientation="h",
            color="Chosen",
            color_discrete_map={True: "#1f77b4", False: "#c7c7c7"},
            template="plotly_white",
            title="Global 12-Month Demand by Segment",
            labels={"Demand": "Units", "Chosen": "Chosen"},
            text_auto=",d",
        )
        fig_demand.update_layout(showlegend=False)
        st.plotly_chart(fig_demand, use_container_width=True)

    with col_b:
        fig_rev = px.bar(
            rev_df.sort_values("Revenue_Potential", ascending=True),
            x="Revenue_Potential", y="Segment", orientation="h",
            color="Chosen",
            color_discrete_map={True: "#ff7f0e", False: "#c7c7c7"},
            template="plotly_white",
            title="Revenue Potential (Demand × Avg Price Ceiling)",
            labels={"Revenue_Potential": "$", "Chosen": "Chosen"},
            text_auto="$,.0f",
        )
        fig_rev.update_layout(showlegend=False)
        st.plotly_chart(fig_rev, use_container_width=True)

    # ── Section 3: Avg price ceilings ────────────────────────────────────────
    st.markdown("---")
    st.subheader("Average Price Ceiling by Segment")
    price_summary = rev_df[["Segment", "Avg_Price", "Chosen"]].sort_values("Avg_Price", ascending=True)
    fig_price = px.bar(
        price_summary, x="Avg_Price", y="Segment", orientation="h",
        color="Chosen",
        color_discrete_map={True: "#2ca02c", False: "#c7c7c7"},
        template="plotly_white",
        title="What customers will pay on average (global)",
        labels={"Avg_Price": "Avg Price Ceiling ($)", "Chosen": "Chosen"},
        text_auto="$,.0f",
    )
    fig_price.update_layout(showlegend=False)
    st.plotly_chart(fig_price, use_container_width=True)
    st.caption(
        "Costcutter ($2,084 avg) leaves the least room for margin. "
        "Mercedes ($4,695) has high price but requires expensive specs and has the smallest demand. "
        "Workhorse and Traveler sit in the mid-market sweet spot."
    )

    # ── Section 4: Customer needs heatmap (all 5 segments) ───────────────────
    st.markdown("---")
    st.subheader("Customer Needs Heatmap — All Segments")
    needs_pivot = needs.set_index("Need")[SEGMENTS]
    fig_heat = px.imshow(
        needs_pivot,
        labels=dict(x="Segment", y="Need", color="Importance Score"),
        aspect="auto",
        color_continuous_scale="Blues",
        template="plotly_white",
        title="Customer Need Importance by Segment (higher = more important)",
    )
    fig_heat.update_xaxes(tickangle=0)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Section 5: WH vs Traveler needs overlap ───────────────────────────────
    st.markdown("---")
    st.subheader("Workhorse vs Traveler — Needs Overlap")
    st.caption("Scores above 110 are high priority for that segment.")

    wh_t = needs[["Need", "Workhorse", "Traveler"]].copy()
    wh_t["Overlap"] = wh_t[["Workhorse", "Traveler"]].min(axis=1)
    wh_t["Difference"] = abs(wh_t["Workhorse"] - wh_t["Traveler"])
    wh_t = wh_t.sort_values("Overlap", ascending=False)

    fig_overlap = go.Figure()
    fig_overlap.add_trace(go.Bar(
        x=wh_t["Need"], y=wh_t["Workhorse"], name="Workhorse",
        marker_color="#1f77b4", opacity=0.85,
    ))
    fig_overlap.add_trace(go.Bar(
        x=wh_t["Need"], y=wh_t["Traveler"], name="Traveler",
        marker_color="#ff7f0e", opacity=0.85,
    ))
    fig_overlap.add_hline(y=110, line_dash="dot", line_color="gray",
                          annotation_text="High priority threshold (110)",
                          annotation_position="right")
    fig_overlap.update_layout(
        barmode="group", template="plotly_white",
        title="Shared needs mean one brand design can serve both segments",
        xaxis_tickangle=-40, height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_overlap, use_container_width=True)

    shared = wh_t[wh_t["Overlap"] >= 110]["Need"].tolist()
    st.success(f"**{len(shared)} needs with score ≥ 110 in both segments:** {', '.join(shared)}")

    # ── Section 6: Use pattern comparison ────────────────────────────────────
    st.markdown("---")
    st.subheader("How Each Segment Uses Their PC")
    use_melt = use_df.melt(id_vars="Application", value_vars=SEGMENTS,
                           var_name="Segment", value_name="Usage Score")
    use_melt["Chosen"] = use_melt["Segment"].isin(CHOSEN)
    fig_use = px.bar(
        use_melt[use_melt["Segment"].isin(CHOSEN)],
        x="Application", y="Usage Score", color="Segment",
        barmode="group",
        color_discrete_map={"Workhorse": "#1f77b4", "Traveler": "#ff7f0e"},
        template="plotly_white",
        title="Workhorse vs Traveler — PC Application Usage",
        text_auto="d",
    )
    fig_use.update_layout(xaxis_tickangle=-35, height=400,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_use, use_container_width=True)
    st.caption(
        "Both segments prioritise Word Processing, Communications and Business Graphics. "
        "Traveler is stronger on Presentations (131) and needs portability. "
        "Workhorse leads on Data Management and Bookkeeping — typical office worker tasks."
    )

    # ── Section 7: Demand by city for chosen segments ────────────────────────
    st.markdown("---")
    st.subheader("Demand by City — Workhorse + Traveler")
    city_df = demand[["City", "Region", "Workhorse", "Traveler"]].copy()
    city_df["Combined"] = city_df["Workhorse"] + city_df["Traveler"]
    city_df = city_df.sort_values("Combined", ascending=True)

    fig_city = px.bar(
        city_df.melt(id_vars=["City", "Region"], value_vars=["Workhorse", "Traveler"],
                     var_name="Segment", value_name="Demand"),
        x="Demand", y="City", color="Segment", orientation="h",
        barmode="stack",
        color_discrete_map={"Workhorse": "#1f77b4", "Traveler": "#ff7f0e"},
        facet_col="Region", facet_col_wrap=3,
        template="plotly_white",
        title="12-Month Demand per City (stacked Workhorse + Traveler)",
    )
    fig_city.update_layout(height=600)
    st.plotly_chart(fig_city, use_container_width=True)


def page_finance():
    st.header("Finance — Q1 Actuals + Q2 Forecast")
    st.caption("Q1 data sourced from simulation exports · Q2 figures are planning inputs")

    # ── Q1 Actuals (hardcoded from BalanceSheet/IncomeStatement/CashFlow/CD Q1) ──
    Q1 = dict(
        cash=912_000,
        cd_balance=0,
        inventory=0,
        net_fixed_assets=0,
        sinking_fund=0,
        total_assets=912_000,
        conventional_loan=0,
        long_term_loan=0,
        emergency_loan=0,
        common_stock=1_000_000,
        retained_earnings=-88_000,
        total_equity=912_000,
        revenues=0,
        cogs=0,
        gross_profit=0,
        market_research_expense=88_000,
        total_expenses=88_000,
        net_income=-88_000,
        eps=-8.80,
        shares_outstanding=10_000,
        cd_rate_pct=1.5,
    )

    # ── Sidebar Q2 planning inputs ────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Q2 Financial Planning")

    q2_stock_proceeds = st.sidebar.number_input(
        "Stock proceeds Q2 ($)", 0, 5_000_000, 1_000_000, step=100_000,
        help="Stock-Q2 shows 10,000 shares @ $100 already decided = $1,000,000"
    )
    q2_revenue = st.sidebar.number_input(
        "Expected Q2 revenue ($)", 0, 20_000_000, 0, step=100_000,
        help="Set once you've decided pricing and production volume"
    )
    q2_cogs_pct = st.sidebar.slider("COGS as % of revenue", 50, 90, 75)
    q2_market_research = st.sidebar.number_input("Market research ($)", 0, 500_000, 88_000, step=1_000)
    q2_advertising = st.sidebar.number_input("Advertising ($)", 0, 2_000_000, 0, step=50_000)
    q2_rd = st.sidebar.number_input("R&D ($)", 0, 1_000_000, 0, step=50_000)
    q2_sales_force = st.sidebar.number_input("Sales force expense ($)", 0, 500_000, 0, step=10_000)

    factory_options = {
        "None — no new capacity ($0)": 0,
        "25 units/day — 1,625 units/qtr ($600K)": 600_000,
        "50 units/day — 3,250 units/qtr ($1.1M)": 1_100_000,
        "100 units/day — 6,500 units/qtr ($2.2M)": 2_200_000,
        "150 units/day — 9,750 units/qtr ($3.6M)": 3_600_000,
    }
    factory_choice = st.sidebar.selectbox("Factory capacity investment", list(factory_options.keys()))
    q2_factory_capex = factory_options[factory_choice]

    office_options = {
        "None": 0,
        "LA only ($180K setup + $80K lease)": 260_000,
        "Toronto only ($160K setup + $70K lease)": 230_000,
        "LA + Toronto ($340K setup + $150K lease)": 490_000,
        "LA + Chicago ($350K setup + $154K lease)": 504_000,
        "Paris + London ($350K setup + $151K lease)": 501_000,
        "Custom — enter below": -1,
    }
    office_choice = st.sidebar.selectbox("Office openings", list(office_options.keys()))
    if office_options[office_choice] == -1:
        q2_office_cost = st.sidebar.number_input("Custom office cost ($)", 0, 2_000_000, 0, step=10_000)
    else:
        q2_office_cost = office_options[office_choice]

    q2_cd_deposit = st.sidebar.number_input(
        "Invest in CD ($)", 0, 900_000, 0, step=50_000,
        help=f"Earns {Q1['cd_rate_pct']}%/quarter. Must leave enough operating cash."
    )
    q2_borrow_lt = st.sidebar.number_input("Borrow long-term loan ($)", 0, 5_000_000, 0, step=100_000)

    # ── Q2 Calculations ───────────────────────────────────────────────────────
    q2_gross_profit = q2_revenue * (1 - q2_cogs_pct / 100)
    q2_operating_expenses = (
        q2_market_research + q2_advertising + q2_rd + q2_sales_force + q2_office_cost
    )
    q2_ebit = q2_gross_profit - q2_operating_expenses
    q2_net_income = q2_ebit  # no taxes in early quarters (loss carry forward)
    q2_cd_interest = q2_cd_deposit * (Q1["cd_rate_pct"] / 100)

    cash_start = Q1["cash"]
    cash_end = (
        cash_start
        + q2_stock_proceeds
        + q2_revenue
        - (q2_revenue * q2_cogs_pct / 100)
        - q2_operating_expenses
        - q2_factory_capex
        - q2_cd_deposit
        + q2_borrow_lt
        + q2_cd_interest
    )

    q2_total_equity = Q1["common_stock"] + q2_stock_proceeds + Q1["retained_earnings"] + q2_net_income
    q2_total_debt = Q1["conventional_loan"] + Q1["long_term_loan"] + q2_borrow_lt
    debt_to_equity = q2_total_debt / q2_total_equity if q2_total_equity > 0 else float("inf")
    emergency_loan_risk = cash_end < 0

    # ── KPI Cards — Q1 Actuals ────────────────────────────────────────────────
    st.subheader("Q1 Actuals")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cash", f"${Q1['cash']:,.0f}")
    c2.metric("Total Equity", f"${Q1['total_equity']:,.0f}")
    c3.metric("Debt", f"${Q1['conventional_loan'] + Q1['long_term_loan']:,.0f}")
    c4.metric("Net Income", f"${Q1['net_income']:,.0f}")
    c5.metric("EPS", f"${Q1['eps']:.2f}")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Revenue", f"${Q1['revenues']:,.0f}")
    c7.metric("Market Research", f"${Q1['market_research_expense']:,.0f}")
    c8.metric("Shares Outstanding", f"{Q1['shares_outstanding']:,}")
    c9.metric("CD Balance", f"${Q1['cd_balance']:,.0f}")
    c10.metric("CD Rate", f"{Q1['cd_rate_pct']}% / qtr")

    # ── Q2 Projected Cash Waterfall ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Q2 Cash Flow Projection")

    waterfall_items = [
        ("Q1 Cash",              cash_start,       "absolute"),
        ("+ Stock proceeds",     q2_stock_proceeds,"relative"),
        ("+ Revenue",            q2_revenue,       "relative"),
        ("- COGS",               -(q2_revenue * q2_cogs_pct / 100), "relative"),
        ("- Operating expenses", -q2_operating_expenses, "relative"),
        ("- Factory CapEx",      -q2_factory_capex, "relative"),
        ("- CD deposit",         -q2_cd_deposit,   "relative"),
        ("+ Loans borrowed",     q2_borrow_lt,     "relative"),
        ("+ CD interest",        q2_cd_interest,   "relative"),
        ("Q2 Ending Cash",       cash_end,         "total"),
    ]

    wf_labels  = [r[0] for r in waterfall_items]
    wf_values  = [r[1] for r in waterfall_items]
    wf_measure = [r[2] for r in waterfall_items]
    wf_colors  = ["#d62728" if v < 0 else "#2ca02c" for v in wf_values]
    wf_colors[0]  = "#1f77b4"   # Q1 Cash — neutral
    wf_colors[-1] = "#ff7f0e" if cash_end < 0 else "#1f77b4"  # ending cash

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=wf_measure,
        x=wf_labels,
        y=wf_values,
        connector=dict(line=dict(color="gray", dash="dot")),
        increasing=dict(marker_color="#2ca02c"),
        decreasing=dict(marker_color="#d62728"),
        totals=dict(marker_color="#1f77b4"),
        texttemplate="%{y:$,.0f}",
        textposition="outside",
    ))
    fig_wf.add_hline(y=0, line_color="black", line_width=1)
    fig_wf.update_layout(
        template="plotly_white",
        title="Q2 Cash Waterfall",
        yaxis_title="$",
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # ── Q2 Ending KPIs ────────────────────────────────────────────────────────
    st.subheader("Q2 Projected Outcome")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Projected Cash", f"${cash_end:,.0f}",
              delta=f"{cash_end - cash_start:,.0f} vs Q1",
              delta_color="normal")
    d2.metric("Projected Net Income", f"${q2_net_income:,.0f}")
    d3.metric("Projected Total Equity", f"${q2_total_equity:,.0f}")
    d4.metric("Debt / Equity", f"{debt_to_equity:.2f}")
    d5.metric("CD Interest earned", f"${q2_cd_interest:,.0f}")

    if emergency_loan_risk:
        st.error(
            f"Emergency loan risk — projected cash is **${cash_end:,.0f}**. "
            "Reduce spending or borrow before submitting Q2 decisions."
        )
    elif cash_end < 200_000:
        st.warning(
            f"Low cash buffer — **${cash_end:,.0f}** projected. "
            "Consider borrowing a conventional loan or reducing CapEx."
        )
    else:
        st.success(f"Cash position looks healthy: **${cash_end:,.0f}** projected at end of Q2.")

    # ── CD Opportunity Analysis ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Certificate of Deposit Opportunity")
    st.caption(f"CD rate: {Q1['cd_rate_pct']}% per quarter (locked for 3 months)")

    cd_amounts = [0, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000]
    cd_rows = []
    for amt in cd_amounts:
        interest = amt * Q1["cd_rate_pct"] / 100
        residual_cash = cash_start + q2_stock_proceeds - q2_operating_expenses - q2_factory_capex - q2_office_cost - amt
        cd_rows.append(dict(CD_Deposit=amt, Interest_Earned=interest,
                            Residual_Cash=residual_cash, Safe=residual_cash > 0))
    cd_df = pd.DataFrame(cd_rows)

    fig_cd = px.bar(
        cd_df, x="CD_Deposit", y="Interest_Earned",
        color="Safe",
        color_discrete_map={True: "#2ca02c", False: "#d62728"},
        template="plotly_white",
        title="CD Interest Earned vs. Amount Deposited (red = cash goes negative)",
        labels={"CD_Deposit": "CD Deposit ($)", "Interest_Earned": "Interest Earned ($)", "Safe": "Cash positive?"},
        text_auto="$,.0f",
    )
    st.plotly_chart(fig_cd, use_container_width=True)

    # ── Payment to Business Partners ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Payment to Business Partners (Q1)")
    partners = pd.DataFrame([
        dict(Company="Apex Systems",        Amount=0),
        dict(Company="AM2 Computers",       Amount=0),
        dict(Company="Apex Global Systems", Amount=0),
        dict(Company="KOVA",                Amount=0),
    ])
    st.dataframe(partners, use_container_width=True, height=180)
    st.caption("All partner payments are $0 in Q1. Update when Q2 licensing decisions are made.")

    # ── Planned Q2 outflows summary ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Q2 Planned Outflows Summary")
    outflows = pd.DataFrame([
        dict(Item="Market Research",      Amount=q2_market_research,     Type="Operating"),
        dict(Item="Advertising",          Amount=q2_advertising,          Type="Operating"),
        dict(Item="R&D",                  Amount=q2_rd,                   Type="Operating"),
        dict(Item="Sales Force",          Amount=q2_sales_force,          Type="Operating"),
        dict(Item="Office Costs",         Amount=q2_office_cost,          Type="Operating"),
        dict(Item="Factory CapEx",        Amount=q2_factory_capex,        Type="Investing"),
        dict(Item="CD Deposit",           Amount=q2_cd_deposit,           Type="Financing"),
        dict(Item="COGS",                 Amount=int(q2_revenue * q2_cogs_pct / 100), Type="Operating"),
    ])
    outflows = outflows[outflows["Amount"] > 0]
    if not outflows.empty:
        fig_out = px.pie(outflows, names="Item", values="Amount", color="Type",
                         template="plotly_white", title="Q2 Spend Breakdown")
        st.plotly_chart(fig_out, use_container_width=True)
    else:
        st.info("Adjust the sidebar inputs to model Q2 spending.")


def page_market_entry():
    st.header("Market Entry — NORAM vs Europe")
    st.caption("Workhorse + Traveler segments · Q2 research data")

    # ── Raw data ──────────────────────────────────────────────────────────────
    cities = pd.DataFrame([
        dict(City="Los Angeles",  Region="NORAM",  Workhorse=4355, Traveler=3205, WH_Price=3333.72, T_Price=3470.40, Setup=180000, Lease_Q=80000,  Open=False),
        dict(City="Chicago",      Region="NORAM",  Workhorse=3949, Traveler=2873, WH_Price=3112.08, T_Price=3706.02, Setup=170000, Lease_Q=74000,  Open=False),
        dict(City="Toronto",      Region="NORAM",  Workhorse=4139, Traveler=2572, WH_Price=3345.69, T_Price=3592.28, Setup=160000, Lease_Q=70000,  Open=False),
        dict(City="Mexico City",  Region="NORAM",  Workhorse=2677, Traveler=1610, WH_Price=3234.20, T_Price=3763.82, Setup=120000, Lease_Q=56000,  Open=False),
        dict(City="Paris",        Region="EUROPE", Workhorse=4377, Traveler=2967, WH_Price=2959.12, T_Price=3628.11, Setup=180000, Lease_Q=75000,  Open=False),
        dict(City="Warsaw",       Region="EUROPE", Workhorse=2525, Traveler=1502, WH_Price=3233.72, T_Price=3575.03, Setup=110000, Lease_Q=46000,  Open=False),
        dict(City="Moscow",       Region="EUROPE", Workhorse=3482, Traveler=2156, WH_Price=3049.45, T_Price=3407.73, Setup=140000, Lease_Q=55000,  Open=False),
        dict(City="London",       Region="EUROPE", Workhorse=4718, Traveler=2776, WH_Price=3214.86, T_Price=3470.26, Setup=170000, Lease_Q=76000,  Open=False),
    ])
    factories = {"NORAM": ("Mexico City", 99), "EUROPE": ("Warsaw", 97)}

    # ── Sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Market Entry Controls")
    mkt_share = st.sidebar.slider("Market share assumption (%)", 5, 30, 10, step=1)
    avg_margin = st.sidebar.slider("Gross margin (%)", 10, 50, 25, step=5)
    brand_price_wh = st.sidebar.number_input("Your Workhorse brand price ($)", 1500, 4000, 2800, step=50)
    brand_price_t  = st.sidebar.number_input("Your Traveler brand price ($)", 1500, 4500, 3200, step=50)

    # ── Derived columns ───────────────────────────────────────────────────────
    df = cities.copy()
    df["Lease_Annual"] = df["Lease_Q"] * 4
    df["Effective_Setup"] = df.apply(lambda r: 0 if r["Open"] else r["Setup"], axis=1)
    df["Annual_Cost"] = df["Effective_Setup"] + df["Lease_Annual"]

    share = mkt_share / 100
    df["WH_Units_Captured"] = (df["Workhorse"] * share).round(0)
    df["T_Units_Captured"]  = (df["Traveler"]  * share).round(0)
    df["Revenue"] = (df["WH_Units_Captured"] * brand_price_wh + df["T_Units_Captured"] * brand_price_t)
    df["Gross_Profit"] = df["Revenue"] * (avg_margin / 100)
    df["Net_After_Costs"] = df["Gross_Profit"] - df["Annual_Cost"]
    df["Rev_Potential_100pct"] = df["Workhorse"] * df["WH_Price"] + df["Traveler"] * df["T_Price"]

    summary = (
        df.groupby("Region")
        .agg(
            WH_Demand=("Workhorse", "sum"),
            T_Demand=("Traveler", "sum"),
            Avg_WH_Price=("WH_Price", "mean"),
            Avg_T_Price=("T_Price", "mean"),
            Revenue=("Revenue", "sum"),
            Gross_Profit=("Gross_Profit", "sum"),
            Annual_Cost=("Annual_Cost", "sum"),
            Net_After_Costs=("Net_After_Costs", "sum"),
        )
        .reset_index()
    )

    noram = summary[summary["Region"] == "NORAM"].iloc[0]
    europe = summary[summary["Region"] == "EUROPE"].iloc[0]

    # ── KPI statement cards ───────────────────────────────────────────────────
    st.subheader("Decision Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🇺🇸 NORAM")
        st.metric("WH + Traveler demand", f"{int(noram['WH_Demand'] + noram['T_Demand']):,} units")
        st.metric("Avg WH price ceiling", f"${noram['Avg_WH_Price']:,.0f}")
        st.metric("Avg Traveler price ceiling", f"${noram['Avg_T_Price']:,.0f}")
        st.metric(f"Revenue @ {mkt_share}% share", f"${noram['Revenue']:,.0f}")
        st.metric(f"Gross profit @ {avg_margin}% margin", f"${noram['Gross_Profit']:,.0f}")
        st.metric("Annual office cost", f"${noram['Annual_Cost']:,.0f}")
        net_delta = noram['Net_After_Costs'] - europe['Net_After_Costs']
        st.metric("Net after office costs", f"${noram['Net_After_Costs']:,.0f}",
                  delta=f"${net_delta:,.0f} vs Europe", delta_color="normal")
        factory_city, factory_idx = factories["NORAM"]
        st.metric("Factory", f"{factory_city} — material index {factory_idx}")
        st.info("No offices open yet — all costs reflect new entry")

    with col2:
        st.markdown("#### 🇪🇺 EUROPE")
        st.metric("WH + Traveler demand", f"{int(europe['WH_Demand'] + europe['T_Demand']):,} units")
        st.metric("Avg WH price ceiling", f"${europe['Avg_WH_Price']:,.0f}")
        st.metric("Avg Traveler price ceiling", f"${europe['Avg_T_Price']:,.0f}")
        st.metric(f"Revenue @ {mkt_share}% share", f"${europe['Revenue']:,.0f}")
        st.metric(f"Gross profit @ {avg_margin}% margin", f"${europe['Gross_Profit']:,.0f}")
        st.metric("Annual office cost", f"${europe['Annual_Cost']:,.0f}")
        st.metric("Net after office costs", f"${europe['Net_After_Costs']:,.0f}")
        factory_city, factory_idx = factories["EUROPE"]
        st.metric("Factory", f"{factory_city} — material index {factory_idx}")
        st.info("No offices open yet — every city requires setup investment")

    # ── Chart 1: Revenue vs Entry Cost bubble ─────────────────────────────────
    st.markdown("---")
    st.subheader("Revenue Potential vs Entry Cost by City")
    st.caption("Bubble size = total Workhorse + Traveler demand. Dashed line = breakeven (revenue = annual cost).")

    fig_bubble = px.scatter(
        df,
        x="Annual_Cost",
        y="Revenue",
        size="WH_Units_Captured",
        color="Region",
        text="City",
        hover_data={"Workhorse": True, "Traveler": True, "WH_Price": ":.0f", "T_Price": ":.0f",
                    "Effective_Setup": ":,", "Lease_Annual": ":,", "Annual_Cost": ":,", "Revenue": ":,.0f",
                    "Net_After_Costs": ":,.0f", "Open": True},
        color_discrete_map={"NORAM": "#1f77b4", "EUROPE": "#ff7f0e"},
        template="plotly_white",
        title=f"Revenue vs Annual Entry Cost ({mkt_share}% market share, ${brand_price_wh:,} WH / ${brand_price_t:,} T)",
    )
    # breakeven line
    max_cost = df["Annual_Cost"].max() * 1.1
    fig_bubble.add_shape(type="line", x0=0, y0=0, x1=max_cost, y1=max_cost,
                         line=dict(color="gray", dash="dash"))
    fig_bubble.update_traces(textposition="top center")
    fig_bubble.update_layout(xaxis_title="Annual Office Cost ($)", yaxis_title="Estimated Revenue ($)")
    st.plotly_chart(fig_bubble, use_container_width=True)

    # ── Chart 2: Demand heatmap ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Segment Demand by City (Workhorse + Traveler)")
    heatmap_df = df[["City", "Region", "Workhorse", "Traveler"]].set_index("City")
    heatmap_df = heatmap_df.sort_values("Region")[["Workhorse", "Traveler"]]
    fig_heat = px.imshow(
        heatmap_df.T,
        labels=dict(x="City", y="Segment", color="Units of demand"),
        aspect="auto",
        color_continuous_scale="Blues",
        template="plotly_white",
        title="12-Month Potential Demand",
    )
    fig_heat.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Chart 3: Price ceiling vs your brand price ────────────────────────────
    st.markdown("---")
    st.subheader("Price Ceiling vs Your Brand Price")
    price_rows = []
    for _, r in df.iterrows():
        price_rows.append(dict(City=r["City"], Region=r["Region"], Segment="Workhorse", PriceCeiling=r["WH_Price"], YourPrice=brand_price_wh))
        price_rows.append(dict(City=r["City"], Region=r["Region"], Segment="Traveler",  PriceCeiling=r["T_Price"],  YourPrice=brand_price_t))
    price_df = pd.DataFrame(price_rows)
    price_df["Headroom"] = price_df["PriceCeiling"] - price_df["YourPrice"]

    fig_price = px.bar(
        price_df,
        x="City",
        y="PriceCeiling",
        color="Segment",
        barmode="group",
        facet_row="Segment",
        template="plotly_white",
        color_discrete_map={"Workhorse": "#1f77b4", "Traveler": "#ff7f0e"},
        title="Customer Price Ceiling by City",
    )
    for seg, price in [("Workhorse", brand_price_wh), ("Traveler", brand_price_t)]:
        row_idx = 2 if seg == "Workhorse" else 1
        fig_price.add_hline(y=price, line_dash="dot", line_color="red",
                            annotation_text=f"Your {seg} price ${price:,}",
                            annotation_position="right", row=row_idx, col=1)
    fig_price.update_layout(height=500, xaxis_tickangle=-30)
    st.plotly_chart(fig_price, use_container_width=True)

    st.caption("Red dotted line = your brand price. Bars above the line = positive headroom (customers willing to pay more than you charge).")

    # ── Chart 4: Break-even quarters ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Break-Even Analysis")
    st.caption("How many quarters until gross profit covers the full setup + one year of lease?")

    df["Quarterly_GP"] = df["Gross_Profit"] / 4
    df["Total_Entry_Cost"] = df["Effective_Setup"] + df["Lease_Annual"]
    df["Breakeven_Quarters"] = df.apply(
        lambda r: 0.0 if r["Total_Entry_Cost"] == 0 else r["Total_Entry_Cost"] / r["Quarterly_GP"]
        if r["Quarterly_GP"] > 0 else float("inf"),
        axis=1,
    )

    fig_be = px.bar(
        df.sort_values("Breakeven_Quarters"),
        x="City",
        y="Breakeven_Quarters",
        color="Region",
        color_discrete_map={"NORAM": "#1f77b4", "EUROPE": "#ff7f0e"},
        template="plotly_white",
        title=f"Quarters to Break Even ({mkt_share}% share, {avg_margin}% margin)",
        text_auto=".1f",
    )
    fig_be.add_hline(y=4, line_dash="dash", line_color="red",
                     annotation_text="1-year threshold", annotation_position="right")
    fig_be.update_layout(yaxis_title="Quarters", xaxis_tickangle=-30)
    st.plotly_chart(fig_be, use_container_width=True)

    # ── City-level detail table ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("City-Level Detail")
    display_cols = {
        "City": "City", "Region": "Region", "Open": "Already Open",
        "Workhorse": "WH Demand", "Traveler": "T Demand",
        "WH_Price": "WH Price Ceiling", "T_Price": "T Price Ceiling",
        "Effective_Setup": "Setup Cost", "Lease_Annual": "Annual Lease",
        "Revenue": "Revenue", "Gross_Profit": "Gross Profit", "Net_After_Costs": "Net (after costs)",
        "Breakeven_Quarters": "Break-even (qtrs)",
    }
    display_df = df[list(display_cols.keys())].rename(columns=display_cols)
    display_df = display_df.sort_values(["Region", "Net (after costs)"], ascending=[True, False])
    fmt_cols = {
        "WH Price Ceiling": "${:,.0f}", "T Price Ceiling": "${:,.0f}",
        "Setup Cost": "${:,.0f}", "Annual Lease": "${:,.0f}",
        "Revenue": "${:,.0f}", "Gross Profit": "${:,.0f}", "Net (after costs)": "${:,.0f}",
        "Break-even (qtrs)": "{:.1f}",
    }
    st.dataframe(
        display_df.style.format(fmt_cols),
        use_container_width=True,
        height=320,
    )


def main():
    st.set_page_config(page_title="Marketplace Assistant", layout="wide")
    st.title("Marketplace Assistant — Dashboard (MVP)")

    st.sidebar.header("Controls")
    page = st.sidebar.selectbox("Page", ["Finance", "Marketing", "Manufacturing", "Market Entry"])

    conn = get_conn()
    income_df = load_table(conn, "income_statement")
    bs_df = load_table(conn, "balance_sheet")
    cf_df = load_table(conn, "cash_flow")
    cust = load_table(conn, "customer_needs")
    sales = load_table(conn, "open_sales_offices")
    factory = load_table(conn, "factory_location")

    if page == "Finance":
        page_finance()
        conn.close()
        return

    elif page == "Marketing":
        page_marketing()
        conn.close()
        return

    elif page == "Market Entry":
        page_market_entry()
        conn.close()
        return

    else:  # Manufacturing
        st.header("Manufacturing")
        st.write("Factory location map and manufacturing data overview.")

        if not factory.empty:
            st.subheader("Factory Location Map")
            fig = plot_city_map(factory, 'city', 'region', title='Factory Locations by Region')
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No factory cities could be geocoded.')
            st.write(factory[['city', 'region', 'material_cost_index']].head(50))
        else:
            st.info("No factory location data available. Normalize FactoryLocation-Q2.xlsx.")

    conn.close()


if __name__ == "__main__":
    main()
