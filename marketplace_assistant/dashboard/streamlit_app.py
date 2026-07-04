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


def page_market_entry():
    st.header("Market Entry — NORAM vs Europe")
    st.caption("Workhorse + Traveler segments · Q2 research data")

    # ── Raw data ──────────────────────────────────────────────────────────────
    cities = pd.DataFrame([
        dict(City="Los Angeles",  Region="NORAM",  Workhorse=4355, Traveler=3205, WH_Price=3333.72, T_Price=3470.40, Setup=180000, Lease_Q=80000,  Open=True),
        dict(City="Chicago",      Region="NORAM",  Workhorse=3949, Traveler=2873, WH_Price=3112.08, T_Price=3706.02, Setup=170000, Lease_Q=74000,  Open=False),
        dict(City="Toronto",      Region="NORAM",  Workhorse=4139, Traveler=2572, WH_Price=3345.69, T_Price=3592.28, Setup=160000, Lease_Q=70000,  Open=True),
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
        st.metric("Annual office cost (new entries only)", f"${noram['Annual_Cost']:,.0f}")
        st.metric("Net after office costs", f"${noram['Net_After_Costs']:,.0f}",
                  delta=f"${noram['Net_After_Costs'] - europe['Net_After_Costs']:,.0f} vs Europe")
        factory_city, factory_idx = factories["NORAM"]
        st.metric("Factory", f"{factory_city} — material index {factory_idx}")
        st.success("LA + Toronto already open → $0 new setup cost for these cities")

    with col2:
        st.markdown("#### 🇪🇺 EUROPE")
        st.metric("WH + Traveler demand", f"{int(europe['WH_Demand'] + europe['T_Demand']):,} units")
        st.metric("Avg WH price ceiling", f"${europe['Avg_WH_Price']:,.0f}")
        st.metric("Avg Traveler price ceiling", f"${europe['Avg_T_Price']:,.0f}")
        st.metric(f"Revenue @ {mkt_share}% share", f"${europe['Revenue']:,.0f}")
        st.metric(f"Gross profit @ {avg_margin}% margin", f"${europe['Gross_Profit']:,.0f}")
        st.metric("Annual office cost (all new entries)", f"${europe['Annual_Cost']:,.0f}")
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
        st.header("Finance")
        st.write("This section shows the raw financial tables and trends.")

        if not income_df.empty:
            st.subheader("Income Statement")
            st.dataframe(income_df)
            st.subheader("Income trends")
            cols_to_plot = [c for c in ["sales", "net_income"] if c in income_df.columns]
            if cols_to_plot:
                fig = px.line(income_df, x="quarter", y=cols_to_plot, markers=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No income statement data available. Run `normalize-csv` on your exports.")

        if not bs_df.empty:
            st.subheader("Balance Sheet")
            st.dataframe(bs_df)
        else:
            st.info("No balance sheet data available.")

        if not cf_df.empty:
            st.subheader("Cash Flow")
            st.dataframe(cf_df)
        else:
            st.info("No cash flow data available.")

    elif page == "Marketing":
        st.header("Marketing")
        st.write("Customer needs heatmap and sales office location map.")

        if not cust.empty:
            st.subheader("Customer Needs Heatmap")
            try:
                pivot = cust.pivot_table(index='Need', columns='segment', values='score', aggfunc='mean')
                fig = px.imshow(pivot.fillna(0), labels=dict(x='Segment', y='Need', color='Score'), aspect='auto', title='Customer Needs by Segment')
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.dataframe(cust)
        else:
            st.info("No customer needs data available. Normalize CustomerNeeds-Q2.xlsx.")

        st.markdown("---")
        st.subheader("Segment Similarity")
        distance_matrix, linkage_matrix, ranked_pairs = build_segment_similarity(cust)
        if distance_matrix is not None and linkage_matrix is not None and ranked_pairs is not None:
            pivot = cust.pivot_table(index='Need', columns='segment', values='score', aggfunc='mean').fillna(0)
            similarity_df = distance_matrix.round(2)
            fig = px.imshow(
                similarity_df,
                labels=dict(x='Segment', y='Segment', color='Distance'),
                aspect='auto',
                title='Segment Distance Matrix'
            )
            st.plotly_chart(fig, use_container_width=True)

            dendro_fig = plot_dendrogram(linkage_matrix, labels=pivot.T.index.tolist())
            st.plotly_chart(dendro_fig, use_container_width=True)

            st.write("Top similar segment pairs")
            top_pairs = pd.DataFrame(ranked_pairs[:10], columns=['segment_a', 'segment_b', 'distance'])
            st.dataframe(top_pairs)
        else:
            st.info("Not enough segment data to compute similarity.")

        st.markdown("---")
        st.subheader("Sales Office Map")
        if not sales.empty:
            fig = plot_city_map(sales, 'city', 'region_sheet', title='Sales Offices by Region')
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No sales office cities could be geocoded.')
            st.write(sales[['city', 'region_sheet', 'status', 'setup_close_cost', 'quarterly_lease_cost']].head(50))
        else:
            st.info("No sales office data available. Normalize OpenSalesOffice-Q2.xlsx.")

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
