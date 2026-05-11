import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airbnb Analytics Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #f0f4f8 0%, #e8edf2 100%);
        border: 1px solid #FF5A5F33;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FF5A5F;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .kpi-delta {
        font-size: 0.85rem;
        color: #00a699;
        margin-top: 4px;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FF5A5F;
        border-left: 3px solid #FF5A5F;
        padding-left: 10px;
        margin-bottom: 12px;
    }

    /* Header subtitle */
    .header-subtitle { color: #888888; font-size: 0.85rem; margin: 0; }

    /* Plotly — light text */
    .js-plotly-plot text,
    .js-plotly-plot .legendtext { fill: #333333 !important; }
    .js-plotly-plot .gtitle     { fill: #222222 !important; }
    .js-plotly-plot .gridlayer path     { stroke: rgba(0,0,0,0.12) !important; }
    .js-plotly-plot .zerolinelayer path { stroke: rgba(0,0,0,0.2)  !important; }
</style>
""", unsafe_allow_html=True)

# ── MongoDB Connection ────────────────────────────────────────────────────────
@st.cache_resource
def get_mongo_client():
    # ⚠️  แก้ CONNECTION_STRING ให้ตรงกับ Atlas cluster ของคุณ
    MONGO_URI = st.secrets.get("MONGO_URI", "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return client

@st.cache_data(ttl=600, show_spinner="⏳ กำลังโหลดข้อมูลจาก MongoDB…")
def load_data():
    client = get_mongo_client()
    db = client["sample_airbnb"]
    col = db["listingsAndReviews"]

    projection = {
        "name": 1, "property_type": 1, "room_type": 1,
        "bedrooms": 1, "bathrooms": 1, "beds": 1,
        "price": 1, "cleaning_fee": 1, "security_deposit": 1,
        "minimum_nights": 1, "number_of_reviews": 1,
        "review_scores.review_scores_rating": 1,
        "review_scores.review_scores_cleanliness": 1,
        "review_scores.review_scores_location": 1,
        "review_scores.review_scores_value": 1,
        "host.host_response_rate": 1,
        "host.host_is_superhost": 1,
        "address.market": 1,
        "address.country": 1,
        "amenities": 1,
        "last_scraped": 1,
    }

    docs = list(col.find({}, projection).limit(5000))
    records = []
    for d in docs:
        addr   = d.get("address", {})
        rs     = d.get("review_scores", {})
        host   = d.get("host", {})
        amenities = d.get("amenities", [])

        # helper: convert Decimal128 / string to float
        def to_float(v):
            if v is None:
                return np.nan
            try:
                return float(str(v))
            except Exception:
                return np.nan

        records.append({
            "name":            d.get("name", ""),
            "property_type":   d.get("property_type", "Other"),
            "room_type":       d.get("room_type", "Other"),
            "bedrooms":        to_float(d.get("bedrooms")),
            "bathrooms":       to_float(d.get("bathrooms")),
            "beds":            to_float(d.get("beds")),
            "price":           to_float(d.get("price")),
            "cleaning_fee":    to_float(d.get("cleaning_fee")),
            "security_deposit":to_float(d.get("security_deposit")),
            "minimum_nights":  to_float(d.get("minimum_nights")),
            "num_reviews":     int(d.get("number_of_reviews", 0)),
            "rating":          to_float(rs.get("review_scores_rating")),
            "score_clean":     to_float(rs.get("review_scores_cleanliness")),
            "score_location":  to_float(rs.get("review_scores_location")),
            "score_value":     to_float(rs.get("review_scores_value")),
            "response_rate":   host.get("host_response_rate", ""),
            "is_superhost":    host.get("host_is_superhost", False),
            "market":          addr.get("market", "Unknown"),
            "country":         addr.get("country", "Unknown"),
            "num_amenities":   len(amenities),
            "has_wifi":        "Wifi" in amenities,
            "has_pool":        any("pool" in a.lower() for a in amenities),
            "has_kitchen":     "Kitchen" in amenities,
            "has_parking":     any("parking" in a.lower() for a in amenities),
        })

    df = pd.DataFrame(records)

    # clean extremes
    df = df[df["price"].between(10, 2000, inclusive="both")]
    df = df[df["price"].notna()]
    return df


# ── Sidebar Filters ───────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_Bélo.svg",
        width=120,
    )
    st.sidebar.markdown("## 🔍 Filters")

    countries = sorted(df["country"].dropna().unique().tolist())
    sel_country = st.sidebar.multiselect("🌏 Country", countries, default=countries[:5] if len(countries) > 5 else countries)

    markets_available = sorted(df[df["country"].isin(sel_country)]["market"].dropna().unique().tolist()) if sel_country else []
    sel_market = st.sidebar.multiselect("📍 Market", markets_available, default=markets_available)

    room_types = sorted(df["room_type"].dropna().unique().tolist())
    sel_room = st.sidebar.multiselect("🛏 Room Type", room_types, default=room_types)

    prop_types = sorted(df["property_type"].value_counts().head(15).index.tolist())
    sel_prop = st.sidebar.multiselect("🏘 Property Type (Top 15)", prop_types, default=prop_types)

    price_min, price_max = int(df["price"].min()), int(df["price"].max())
    sel_price = st.sidebar.slider("💰 Price per Night ($)", price_min, price_max, (price_min, min(500, price_max)))

    sel_superhost = st.sidebar.checkbox("⭐ Superhost Only", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption("Data: MongoDB Sample Airbnb · `sample_airbnb.listingsAndReviews`")

    # Apply filters
    mask = (
        df["room_type"].isin(sel_room) &
        df["property_type"].isin(sel_prop) &
        df["price"].between(sel_price[0], sel_price[1])
    )
    if sel_country:
        mask &= df["country"].isin(sel_country)
    if sel_market:
        mask &= df["market"].isin(sel_market)
    if sel_superhost:
        mask &= df["is_superhost"] == True

    return df[mask].copy()


# ── KPI Row ───────────────────────────────────────────────────────────────────
def render_kpis(df: pd.DataFrame):
    total      = len(df)
    avg_price  = df["price"].mean()
    avg_rating = df["rating"].mean()
    superhosts = df["is_superhost"].sum()
    avg_reviews= df["num_reviews"].mean()

    cols = st.columns(5)
    kpis = [
        ("🏠 Total Listings",      f"{total:,}",           f"avg {df['num_amenities'].mean():.1f} amenities"),
        ("💰 Avg Price / Night",   f"${avg_price:.0f}",    f"range ${df['price'].min():.0f}–${df['price'].max():.0f}"),
        ("⭐ Avg Rating",          f"{avg_rating:.1f}/100", f"{df['rating'].notna().sum():,} rated"),
        ("🏆 Superhosts",          f"{superhosts:,}",       f"{superhosts/total*100:.1f}% of total"),
        ("📝 Avg Reviews",         f"{avg_reviews:.0f}",   f"total {df['num_reviews'].sum():,}"),
    ]
    for col, (label, val, delta) in zip(cols, kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{val}</p>
            <p class="kpi-label">{label}</p>
            <p class="kpi-delta">{delta}</p>
        </div>""", unsafe_allow_html=True)


# ── Charts ────────────────────────────────────────────────────────────────────
AIRBNB_COLORS = ["#FF5A5F", "#FC642D", "#FFB400", "#00A699", "#484848",
                 "#767676", "#00d4aa", "#a78bfa", "#f472b6", "#34d399"]

def chart_price_by_room(df):
    fig = px.box(
        df.dropna(subset=["price", "room_type"]),
        x="room_type", y="price", color="room_type",
        title="Price Distribution by Room Type",
        color_discrete_sequence=AIRBNB_COLORS,
        points="outliers",
    )
    fig.update_layout(**chart_layout(), showlegend=False, xaxis_title="", yaxis_title="Price ($)")
    return fig

def chart_top_markets(df):
    top = (df.groupby("market")
             .agg(listings=("price","count"), avg_price=("price","mean"))
             .nlargest(12,"listings").reset_index())
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=top["market"], y=top["listings"], name="Listings",
                         marker_color="#FF5A5F", opacity=0.85), secondary_y=False)
    fig.add_trace(go.Scatter(x=top["market"], y=top["avg_price"], name="Avg Price",
                             mode="lines+markers", line=dict(color="#FFB400", width=2),
                             marker=dict(size=7)), secondary_y=True)
    fig.update_layout(title="Top Markets — Listings vs Avg Price", **chart_layout())
    fig.update_yaxes(title_text="Listings",    secondary_y=False)
    fig.update_yaxes(title_text="Avg Price ($)", secondary_y=True)
    return fig

def chart_property_pie(df):
    top = df["property_type"].value_counts().head(8)
    others = df["property_type"].value_counts().iloc[8:].sum()
    labels = list(top.index) + (["Others"] if others else [])
    values = list(top.values) + ([others] if others else [])
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker_colors=AIRBNB_COLORS,
        textinfo="percent+label",
        hovertemplate="%{label}: %{value:,} listings<extra></extra>"
    ))
    fig.update_layout(title="Property Type Breakdown", **chart_layout(), showlegend=False)
    return fig

def chart_rating_dist(df):
    d = df["rating"].dropna()
    fig = go.Figure(go.Histogram(
        x=d, nbinsx=30,
        marker_color="#FF5A5F", opacity=0.85,
        hovertemplate="Rating %{x}: %{y} listings<extra></extra>"
    ))
    fig.add_vline(x=d.mean(), line_dash="dash", line_color="#FFB400",
                  annotation_text=f"Mean {d.mean():.1f}", annotation_position="top right")
    fig.update_layout(title="Review Score Rating Distribution", xaxis_title="Rating (0–100)",
                      yaxis_title="Count", **chart_layout())
    return fig

def chart_price_heatmap(df):
    pivot = (df.groupby(["room_type","property_type"])["price"]
               .mean()
               .unstack(fill_value=0))
    pivot = pivot[[c for c in pivot.columns if pivot[c].sum() > 0]]
    # keep top property types
    top_props = df["property_type"].value_counts().head(8).index
    pivot = pivot[[c for c in pivot.columns if c in top_props]]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0,"#1a1a2e"],[0.5,"#FF5A5F"],[1,"#FFB400"]],
        hoverongaps=False,
        hovertemplate="Room: %{y}<br>Property: %{x}<br>Avg Price: $%{z:.0f}<extra></extra>"
    ))
    fig.update_layout(title="Avg Price Heatmap — Room × Property Type", **chart_layout())
    return fig

def chart_amenity_impact(df):
    amenity_cols = {"WiFi": "has_wifi", "Pool": "has_pool",
                    "Kitchen": "has_kitchen", "Parking": "has_parking"}
    rows = []
    for name, col in amenity_cols.items():
        with_a = df[df[col]]["price"].mean()
        without = df[~df[col]]["price"].mean()
        rows.append({"Amenity": name, "With": with_a, "Without": without})
    comp = pd.DataFrame(rows).melt(id_vars="Amenity", var_name="Status", value_name="Avg Price")
    fig = px.bar(comp, x="Amenity", y="Avg Price", color="Status", barmode="group",
                 title="Amenity Impact on Avg Price",
                 color_discrete_map={"With": "#FF5A5F", "Without": "#484848"})
    fig.update_layout(**chart_layout())
    return fig

def chart_reviews_vs_price(df):
    clean = df.dropna(subset=["price","num_reviews","rating"])
    sample = clean.sample(min(800, len(clean)), random_state=42)
    fig = px.scatter(
        sample, x="price", y="num_reviews",
        color="room_type", size="rating",
        size_max=15,
        hover_data=["name","market","bedrooms"],
        title="Price vs Number of Reviews (size = Rating)",
        color_discrete_sequence=AIRBNB_COLORS,
        opacity=0.7,
    )
    fig.update_layout(**chart_layout(), xaxis_title="Price ($)", yaxis_title="# Reviews")
    return fig

def chart_score_radar(df):
    cats = ["Cleanliness","Location","Value","Rating (÷10)"]
    vals_all  = [df["score_clean"].mean(), df["score_location"].mean(),
                 df["score_value"].mean(), df["rating"].mean()/10]
    vals_sup  = [df[df["is_superhost"]]["score_clean"].mean(),
                 df[df["is_superhost"]]["score_location"].mean(),
                 df[df["is_superhost"]]["score_value"].mean(),
                 df[df["is_superhost"]]["rating"].mean()/10]

    fig = go.Figure()
    for name, vals, color in [("All Hosts", vals_all, "#484848"),
                               ("Superhosts", vals_sup, "#FF5A5F")]:
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", name=name,
            line_color=color, fillcolor=color, opacity=0.4
        ))
    fig.update_layout(
        title="Score Radar — Superhosts vs All",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10],
                            tickfont=dict(color="#aaa"),
                            gridcolor="#333"),
            angularaxis=dict(tickfont=dict(color="#ccc"), gridcolor="#333"),
            bgcolor="#1a1a2e",
        ),
        **chart_layout()
    )
    return fig

def chart_bedrooms_price(df):
    d = df[df["bedrooms"].between(0,6)].dropna(subset=["price","bedrooms"])
    d["bedrooms"] = d["bedrooms"].astype(int).astype(str) + " BR"
    fig = px.violin(d, x="bedrooms", y="price", color="bedrooms",
                    box=True, points=False,
                    title="Price Distribution by Bedrooms",
                    color_discrete_sequence=AIRBNB_COLORS)
    fig.update_layout(**chart_layout(), showlegend=False,
                      xaxis_title="Bedrooms", yaxis_title="Price ($)")
    return fig

def chart_layout():
    _axis = dict(
        gridcolor="rgba(128,128,128,0.15)",
        linecolor="rgba(128,128,128,0.3)",
        zerolinecolor="rgba(128,128,128,0.2)",
    )
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        margin=dict(t=45, b=30, l=40, r=20),
        height=360,
        title_font=dict(size=14),
        xaxis=_axis,
        yaxis=_axis,
    )


# ── Data Table ────────────────────────────────────────────────────────────────
def render_table(df):
    show_cols = ["name","property_type","room_type","market","country",
                 "bedrooms","price","rating","num_reviews","is_superhost"]
    display = df[show_cols].dropna(subset=["price"]).head(200).copy()
    display.columns = ["Name","Type","Room","Market","Country","BR","Price $","Rating","Reviews","Superhost"]
    display["Price $"]   = display["Price $"].map("${:.0f}".format)
    display["Rating"]    = display["Rating"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "–")
    display["Superhost"] = display["Superhost"].map({True:"✅", False:"❌"})
    st.dataframe(display, use_container_width=True, height=320)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">
        <span style="font-size:2.4rem">🏠</span>
        <div>
            <h1 style="margin:0;font-size:1.8rem;color:#FF5A5F">Airbnb Analytics Dashboard</h1>
            <p class="header-subtitle">MongoDB · sample_airbnb.listingsAndReviews</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    try:
        df_raw = load_data()
    except Exception as e:
        st.error(f"❌ ไม่สามารถเชื่อมต่อ MongoDB ได้: {e}")
        st.info("กรุณาตั้งค่า `MONGO_URI` ใน `.streamlit/secrets.toml`\n\n```toml\nMONGO_URI = 'mongodb+srv://...' \n```")
        st.stop()

    df = render_sidebar(df_raw)

    if df.empty:
        st.warning("⚠️ ไม่มีข้อมูลที่ตรงกับ Filter ที่เลือก — กรุณาปรับเงื่อนไข")
        st.stop()

    # KPIs
    render_kpis(df)
    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1
    st.markdown('<p class="section-header">📊 Market & Listing Overview</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(chart_top_markets(df), use_container_width=True)
    c2.plotly_chart(chart_property_pie(df), use_container_width=True)

    # Row 2
    st.markdown('<p class="section-header">💰 Price Analysis</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(chart_price_by_room(df), use_container_width=True)
    # c2.plotly_chart(chart_bedrooms_price(df), use_container_width=True)

    # Row 3
    c1, c2 = st.columns(2)
    c1.plotly_chart(chart_price_heatmap(df), use_container_width=True)
    c2.plotly_chart(chart_amenity_impact(df), use_container_width=True)

    # Row 4
    st.markdown('<p class="section-header">⭐ Review & Quality Scores</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(chart_rating_dist(df), use_container_width=True)
    # c2.plotly_chart(chart_score_radar(df), use_container_width=True)

    # Row 5 — full width scatter
    st.markdown('<p class="section-header">🔍 Listing Explorer</p>', unsafe_allow_html=True)
    fig_scatter = chart_reviews_vs_price(df)
    fig_scatter.update_layout(height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Table
    st.markdown('<p class="section-header">📋 Listing Table (top 200)</p>', unsafe_allow_html=True)
    render_table(df)

    st.caption(f"Showing **{len(df):,}** of **{len(df_raw):,}** listings · Filtered by sidebar controls")


if __name__ == "__main__":
    main()
