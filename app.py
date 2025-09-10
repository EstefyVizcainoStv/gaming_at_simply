# app.py
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==============================
# Streamlit setup
# ==============================
st.set_page_config(page_title="Client Support — Closer Dashboard", layout="wide")
st.markdown(
    """
    <style>
      .metric-small div[data-testid="stMetricValue"]{font-size:28px;}
      .card {border-radius:18px; background:#fff; box-shadow:0 10px 20px rgba(0,0,0,.08); overflow:hidden;}
      .card-h {padding:14px 18px; color:#111; font-weight:800; font-size:20px;}
      .card-b {padding:18px 22px;}
      .big  {font-size:36px; font-weight:800; margin:2px 0 4px}
      .meta {font-size:16px; opacity:.9}
      .pow  {font-size:22px; font-weight:800; margin-top:8px}
      .sub  {opacity:.8; font-size:14px}
      .stats {font-size:16px; margin-top:10px; line-height:1.6}
      .note {opacity:.75; font-size:13px}
      table {border-collapse: separate; border-spacing: 0 4px;}
      thead tr th {background:#eef2ff !important; color:#111827 !important; font-weight:700 !important;}
      tbody tr td {border:0 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Configuration / column names
# ==============================
COLS = {
    "status": "Status",
    "client": "ClientName",
    "tam": "TAM",
    "closer": "Closer",
    "ttc": "Time to Close",          # numeric hours
    "created": "CreatedAt",
    "closed": "ClosedAt",
    "reopened": "ReopenedAt",
}

POSSIBLE_PATHS = ["GAMING 2.csv", "/mnt/data/GAMING 2.csv", "data/GAMING 2.csv"]
EXCLUDE_STATUSES_FOR_SPEED = {"pending", "processing"}
DROP_NAMES = {"not found", "nan", "none", ""}

# ==============================
# Helpers
# ==============================
def load_data(upload):
    """Load CSV from upload or common local paths; show message if none found."""
    if upload is not None:
        df = pd.read_csv(upload)
    else:
        path = next((p for p in POSSIBLE_PATHS if os.path.exists(p)), None)
        if not path:
            st.title("Client Support — Closer Dashboard")
            st.info(
                "👋 Drop your CSV in the **sidebar** (or save it next to `app.py` as `GAMING 2.csv`).\n\n"
                "**Required columns (case-insensitive):** `Closer`, `ClientName`, `Time to Close`.\n"
                "Optional: `Status`, `ReopenedAt`, `ClosedAt`, `CreatedAt`."
            )
            st.stop()
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def safe_col(df, *opts, default=None):
    for c in opts:
        if c in df.columns:
            return c
    return default

def clean_people(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return s

def percent_rank(series: pd.Series) -> pd.Series:
    """0..1 percentile rank; if all equal, return 0.5."""
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True, method="average")

def compute_reopen(df, closer_col, reopened_col):
    """Reopen counts, rate, and penalty (0..15) per closer."""
    tmp = df.copy()
    if reopened_col in tmp.columns:
        tmp[reopened_col] = pd.to_datetime(tmp[reopened_col], errors="coerce", dayfirst=True)
    else:
        tmp[reopened_col] = pd.NaT
    tmp["_reopened_flag"] = tmp[reopened_col].notna()

    quality = (
        tmp.groupby(closer_col)
           .agg(tickets_closed=("_reopened_flag", "size"),
                reopens=("_reopened_flag", "sum"))
           .reset_index()
    )
    quality["reopen_rate"] = quality["reopens"] / quality["tickets_closed"].replace(0, np.nan)
    quality["reopen_rate"] = quality["reopen_rate"].fillna(0.0)

    MAX_PEN = 15.0
    quality["penalty"] = (percent_rank(quality["reopen_rate"]) * MAX_PEN).round(1)
    return quality.rename(columns={closer_col: "Closer"})

def score_closer_table(df, closer_col, client_col, ttc_col, reopened_col, status_col):
    """Build leaderboard for Closers with normalized pillars + reopen penalty."""
    d = df.copy()

    # sanitize closer
    d[closer_col] = clean_people(d[closer_col])
    d = d[d[closer_col].notna()]
    d = d[~d[closer_col].str.lower().isin(DROP_NAMES)]

    # numeric TTC
    if not np.issubdtype(d[ttc_col].dtype, np.number):
        d[ttc_col] = pd.to_numeric(d[ttc_col], errors="coerce")

    # volume
    vol = d.groupby(closer_col).size().reset_index(name="Tickets Closed")

    # per-client throughput (tickets per closer across clients)
    per_client = (
        d.groupby([closer_col, client_col]).size()
         .reset_index(name="cnt")
         .groupby(closer_col)["cnt"].sum()
         .reset_index(name="Closed per Client")
    )

    # speed base — exclude pending/processing
    speed_base = d.copy()
    if status_col in speed_base.columns:
        mask = ~speed_base[status_col].astype(str).str.lower().isin(EXCLUDE_STATUSES_FOR_SPEED)
        speed_base = speed_base[mask]
    speed = (
        speed_base.groupby(closer_col)[ttc_col].mean().reset_index()
                  .rename(columns={ttc_col: "Avg Close (h)"})
    )

    # merge pillars
    lb = vol.merge(per_client, on=closer_col, how="left").merge(speed, on=closer_col, how="left")

    # normalized scores (0..100)
    lb["Volume Score (0–100)"]     = (percent_rank(lb["Tickets Closed"]) * 100).round(1)
    lb["Throughput Score (0–100)"] = (percent_rank(lb["Closed per Client"]) * 100).round(1)
    lb["Speed Score (0–100)"]      = ((1 - percent_rank(lb["Avg Close (h)"])) * 100).round(1)  # lower hours = faster

    # base power (weights)
    w_speed, w_through, w_vol = 0.30, 0.40, 0.30
    lb["POWER SCORE (0–100)"] = (
        lb["Speed Score (0–100)"] * w_speed
      + lb["Throughput Score (0–100)"] * w_through
      + lb["Volume Score (0–100)"] * w_vol
    ).round(1)

    # re-open penalty
    quality = compute_reopen(d, closer_col, reopened_col)
    lb = lb.merge(quality[["Closer","reopen_rate","penalty","reopens","tickets_closed"]],
                  left_on=closer_col, right_on="Closer", how="left")
    lb = lb.fillna({"reopen_rate":0.0, "penalty":0.0, "reopens":0, "tickets_closed":0})

    # adjusted power
    lb["Adj POWER SCORE (0–100)"] = (lb["POWER SCORE (0–100)"] - lb["penalty"]).clip(lower=0)

    # flavor title + tiers
    lb["Title"] = np.where(lb["Speed Score (0–100)"] >= 70, "Lightning Closer", "Steady Operator")
    def tier_from_score(s):
        if s >= 80: return "Mythic 🟣"
        if s >= 70: return "Legendary 🟡"
        if s >= 55: return "Epic 🔵"
        return "Rare 🟢"
    lb["Tier"] = lb["Adj POWER SCORE (0–100)"].apply(tier_from_score)

    # rank & order
    lb = lb.rename(columns={closer_col:"Closer"})
    lb = lb.sort_values("Adj POWER SCORE (0–100)", ascending=False).reset_index(drop=True)
    lb.insert(0, "Rank", lb.index + 1)

    return lb[
        ["Rank","Closer","Title","Tier",
         "Adj POWER SCORE (0–100)","penalty",
         "Speed Score (0–100)","Throughput Score (0–100)","Volume Score (0–100)",
         "Avg Close (h)","Tickets Closed","Closed per Client",
         "reopens","reopen_rate"]
    ].rename(columns={"penalty":"Re-open Penalty (0–15)","reopen_rate":"Re-open Rate",
                      "reopens":"Reopens"})

def styled_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    s = (df.style
         # ↓ new API
         .hide(axis="index")
         .format({
             "Adj POWER SCORE (0–100)":"{:.1f}",
             "Re-open Penalty (0–15)":"{:.1f}",
             "Speed Score (0–100)":"{:.1f}",
             "Throughput Score (0–100)":"{:.1f}",
             "Volume Score (0–100)":"{:.1f}",
             "Avg Close (h)":"{:.2f}",
             "Closed per Client":"{:.2f}",
             "Re-open Rate":"{:.2%}",
         })
         .background_gradient(cmap="YlOrRd", subset=["Adj POWER SCORE (0–100)"])
         .bar(subset=["Re-open Penalty (0–15)"], color="#ef4444")
         .background_gradient(cmap="PuBuGn", subset=["Speed Score (0–100)"])
         .background_gradient(cmap="BuGn",   subset=["Throughput Score (0–100)"])
         .background_gradient(cmap="Greys",  subset=["Volume Score (0–100)"])
         .background_gradient(cmap="RdYlGn_r", subset=["Avg Close (h)"])
         .bar(subset=["Tickets Closed"], color="#6c5ce7")
         .bar(subset=["Closed per Client"], color="#10b981")
    )
    return s


def achievement_card(row, place):
    medal = {1:"🥇", 2:"🥈", 3:"🥉"}[place]
    grad  = {1:"linear-gradient(90deg,#f59e0b,#b45309)",
             2:"linear-gradient(90deg,#9ca3af,#6b7280)",
             3:"linear-gradient(90deg,#b45309,#92400e)"}[place]
    name  = row["Closer"]
    title = row["Title"]
    tier  = row["Tier"]
    adj   = f'{row["Adj POWER SCORE (0–100)"]:.1f}'
    pen   = f'{row["Re-open Penalty (0–15)"]:.1f}'
    rate  = f'{row["Re-open Rate"]:.2%}'
    avg   = f'{row["Avg Close (h)"]:.2f}'
    per   = f'{row["Closed per Client"]:.2f}'
    vol   = f'{row["Tickets Closed"]:.0f}'

    st.markdown(
        f"""
        <div class="card">
          <div class="card-h" style="background:{grad}">{medal} ACHIEVEMENT UNLOCKED</div>
          <div class="card-b">
            <div class="big">CLOSER: {name}</div>
            <div class="meta">Title: “{title}”</div>
            <div class="meta">Tier: {tier}</div>
            <div class="pow">Adjusted Power: {adj}</div>
            <div class="sub">(Penalty: {pen} · Re-open rate: {rate})</div>
            <div class="stats">
              <div>Avg Close (h): <b>{avg}</b></div>
              <div>Closed per Client: <b>{per}</b></div>
              <div>Tickets Closed: <b>{vol}</b></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================
# Sidebar — data load & options
# ==============================
st.sidebar.header("Load data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
df = load_data(uploaded)

# Resolve columns dynamically
status_col   = safe_col(df, COLS["status"], "status")
client_col   = safe_col(df, COLS["client"], "Client")
closer_col   = safe_col(df, COLS["closer"], "Closer")
ttc_col      = safe_col(df, COLS["ttc"], "Time to Close", "time_to_close", "TTC")
reopened_col = safe_col(df, COLS["reopened"], "ReopenedAt", "reopenedat")
closed_col   = safe_col(df, COLS["closed"], "ClosedAt", "closedat")
created_col  = safe_col(df, COLS["created"], "CreatedAt", "createdat")

required = [client_col, closer_col, ttc_col]
if any(c is None for c in required):
    st.error("Your CSV is missing required columns (Closer, ClientName, Time to Close).")
    st.stop()

# Filters
st.sidebar.header("Filters")
exclude_names = st.sidebar.multiselect(
    "Exclude Closers",
    sorted(df[closer_col].dropna().astype(str).unique()),
    default=["Bas","Estefy","Shub"]
)
show_top_n = st.sidebar.slider("Top N for charts", 5, 20, 15)

# Apply basic cleaning & exclusions
df[closer_col] = clean_people(df[closer_col])
df = df[~df[closer_col].str.lower().isin({n.lower() for n in exclude_names})]

# ==============================
# KPIs
# ==============================
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    total_tickets = len(df)
    unique_clients = df[client_col].nunique()
    unique_closers = df[closer_col].nunique()
    avg_ttc = pd.to_numeric(df[ttc_col], errors="coerce").mean()
    c1.metric("Tickets", f"{total_tickets:,}")
    c2.metric("Unique Clients", f"{unique_clients:,}")
    c3.metric("Closers", f"{unique_closers:,}")
    c4.metric("Avg Close (h)", f"{avg_ttc:.2f}" if pd.notna(avg_ttc) else "—")

st.markdown("---")

# ==============================
# Tabs
# ==============================
tab1, tab2, tab3 = st.tabs(["📊 Charts", "🏆 Leaderboard", "🎖 Achievements"])

# ---------- Charts ----------
with tab1:
    colA, colB = st.columns(2)

    # Top clients by tickets
    clients_df = (
        df.groupby(client_col).size().reset_index(name="tickets_count")
          .sort_values("tickets_count", ascending=False).head(show_top_n)
    )
    fig_clients = px.bar(
        clients_df, x="tickets_count", y=client_col, orientation="h", color="tickets_count",
        title="👑 Which client generated the most tickets?",
        labels={"tickets_count":"Tickets", client_col:"Client"},
        template="simple_white",
    )
    fig_clients.update_yaxes(categoryorder="total ascending")
    colA.plotly_chart(fig_clients, use_container_width=True)

    # Tickets closed by closer (exclude pending/processing)
    df_closed = df.copy()
    if status_col in df_closed.columns:
        df_closed = df_closed[~df_closed[status_col].astype(str).str.lower().isin(EXCLUDE_STATUSES_FOR_SPEED)]

    closer_closed = (
        df_closed.groupby(closer_col).size().reset_index(name="tickets_closed")
                 .sort_values("tickets_closed", ascending=False).head(show_top_n)
    )
    fig_closer = px.bar(
        closer_closed, x=closer_col, y="tickets_closed", color="tickets_closed",
        title="🔥 Which Closer closed the most tickets? (pending/processing removed)",
        labels={"tickets_closed":"Tickets", closer_col:"Closer"},
        template="simple_white",
    )
    colB.plotly_chart(fig_closer, use_container_width=True)

    # Fastest closers (avg hours) — exclude pending/processing
    avg_df = (
        df_closed.groupby(closer_col)[ttc_col].mean().reset_index()
                 .rename(columns={ttc_col:"avg_hours"})
                 .sort_values("avg_hours", ascending=True).head(show_top_n)
    )
    fig_speed = px.bar(
        avg_df, x=closer_col, y="avg_hours", color="avg_hours",
        title="⚡ Fastest Closers (Avg Time to Close — lower is better)",
        labels={"avg_hours":"Avg hours", closer_col:"Closer"},
        template="simple_white",
    )
    st.plotly_chart(fig_speed, use_container_width=True)

    # Re-open champion (latest week with data)
    if reopened_col:
        tmp = df.copy()
        tmp[reopened_col] = pd.to_datetime(tmp[reopened_col], errors="coerce", dayfirst=True)
        tmp = tmp[tmp[reopened_col].notna()]
        if len(tmp):
            iso = tmp[reopened_col].dt.isocalendar()
            tmp["re_week"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
            latest_week = tmp["re_week"].sort_values().iloc[-1]
            r_week = (
                tmp[tmp["re_week"] == latest_week]
                .groupby(closer_col).size().reset_index(name="tickets_reopened")
                .sort_values("tickets_reopened", ascending=False)
            )
            fig_re = px.bar(
                r_week, x="tickets_reopened", y=closer_col, orientation="h", color="tickets_reopened",
                title=f"🔄 Re-open Champion (latest week: {latest_week})",
                labels={"tickets_reopened":"Tickets Reopened", closer_col:"Closer"},
                template="simple_white",
            )
            fig_re.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_re, use_container_width=True)

# ---------- Leaderboard ----------
with tab2:
    lb = score_closer_table(
        df, closer_col=closer_col, client_col=client_col,
        ttc_col=ttc_col, reopened_col=reopened_col, status_col=status_col
    )

    st.subheader("🏆 Power Leaderboard (Closer) — with Re-open Penalty & Normalized Scores")
    st.caption("Adjusted Power = 0.3·Speed + 0.4·Per-Client + 0.3·Volume − Penalty (0–15).")

    st.markdown(styled_table(lb).to_html(), unsafe_allow_html=True)

    with st.expander("Show raw table"):
        st.dataframe(lb, use_container_width=True)

# ---------- Achievements ----------
with tab3:
    lb = score_closer_table(
        df, closer_col=closer_col, client_col=client_col,
        ttc_col=ttc_col, reopened_col=reopened_col, status_col=status_col
    )
    top3 = lb.head(3).reset_index(drop=True)
    a,b,c = st.columns(3)
    if len(top3) > 0:
        with a: achievement_card(top3.iloc[0], 1)
    if len(top3) > 1:
        with b: achievement_card(top3.iloc[1], 2)
    if len(top3) > 2:
        with c: achievement_card(top3.iloc[2], 3)

    st.markdown('<div class="note">Tip: use the sidebar to exclude names (e.g., Bas/Estefy/Shub) or to change Top-N on charts.</div>', unsafe_allow_html=True)
