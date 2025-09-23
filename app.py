# app.py
import io
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
# Required / canonical headers
# ==============================
REQUIRED_HEADERS = [
    "ID","ClientID","Status","Title","Field","Service","Category",
    "CreatedAt","UpdatedAt","AcknowledgedAt","ProcessingAt","ClosedAt",
    "ReopenedAt","LastSimplyReplyAt","LastClientReplyAt","ClientName",
    "Time to Close","TAM","Closer"
]

# Synonyms / normalization (case-insensitive; spaces/underscores ignored)
SYNONYMS = {
    "ID": ["id"],
    "ClientID": ["clientid","client_id","client id"],
    "Status": ["status"],
    "Title": ["title"],
    "Field": ["field"],
    "Service": ["service"],
    "Category": ["category"],
    "CreatedAt": ["createdat","created_at","created at","openedat","opened_at","opened at"],
    "UpdatedAt": ["updatedat","updated_at","updated at","lastupdatedat","last updated at"],
    "AcknowledgedAt": ["acknowledgedat","ack_at","acknowledged at","acknowledged"],
    "ProcessingAt": ["processingat","processing_at","processing at"],
    "ClosedAt": ["closedat","closed_at","closed at","resolvedat","resolved_at","resolved at"],
    "ReopenedAt": ["reopenedat","reopened_at","reopened at"],
    "LastSimplyReplyAt": ["lastsimplyreplyat","last simply reply at","lastsimplyreply","last simply reply"],
    "LastClientReplyAt": ["lastclientreplyat","last client reply at","lastclientreply","last client reply"],
    "ClientName": ["clientname","client","customer","customername"],
    "Time to Close": ["time_to_close","timetoclose","ttc","resolutiontime","resolution time"],
    "TAM": ["tam","account manager","technical account manager"],
    "Closer": ["closer","agent","assignee","owner","handler"]
}

# Columns actually used by the app logic
USED = {
    "status":   "Status",
    "client":   "ClientName",
    "tam":      "TAM",
    "closer":   "Closer",
    "ttc":      "Time to Close",
    "created":  "CreatedAt",
    "closed":   "ClosedAt",
    "reopened": "ReopenedAt",
}

EXCLUDE_STATUSES_FOR_SPEED = {"pending", "processing"}
DROP_NAMES = {"not found", "nan", "none", ""}

# --- Re-open penalty configuration (Bayesian smoothing) ---
MAX_PENALTY = 15.0
# Prior of ~5% re-open rate -> (alpha / (alpha+beta)) ≈ 0.05
REOPEN_PRIOR_ALPHA = 1.0
REOPEN_PRIOR_BETA  = 19.0

# ==============================
# Helpers
# ==============================
def _norm(s: str) -> str:
    """Normalize a header: lowercase, strip, collapse spaces/underscores."""
    return "".join(str(s).strip().lower().replace("_", " ").split())

def resolve_schema(df: pd.DataFrame):
    """Return colmap (canonical→actual) and missing canonical headers."""
    actual_norm = {_norm(c): c for c in df.columns}
    colmap, missing = {}, []
    for canonical in REQUIRED_HEADERS:
        candidates = [_norm(canonical)] + [_norm(a) for a in SYNONYMS.get(canonical, [])]
        found = next((actual_norm[c] for c in candidates if c in actual_norm), None)
        if found: colmap[canonical] = found
        else:     missing.append(canonical)
    return colmap, missing

def load_any(uploaded):
    """Load ONLY from the uploader (CSV/TSV/XLS/XLSX). No local fallbacks."""
    if uploaded is None:
        return None

    name = (getattr(uploaded, "name", "") or "").lower()

    # Reset pointer to the start
    try:
        uploaded.seek(0)
    except Exception:
        pass

    if name.endswith((".xls", ".xlsx")):
        try:
            # Let pandas pick engine; if openpyxl is missing, show a friendly error.
            df = pd.read_excel(uploaded)
        except ImportError:
            st.error("Reading Excel requires the `openpyxl` package. Add it to `requirements.txt`.")
            return None
    else:
        # Read bytes once, then try common delimiters
        raw = uploaded.read()
        df = None
        for sep in [",", ";", "\t", "|"]:
            try:
                trial = pd.read_csv(io.BytesIO(raw), sep=sep)
                # If everything collapsed into one column, try next separator
                if trial.shape[1] == 1 and sep != ",":
                    continue
                df = trial
                break
            except Exception:
                continue
        if df is None:
            df = pd.read_csv(io.BytesIO(raw))  # last resort

    df.columns = [str(c).strip() for c in df.columns]
    return df

def clean_people(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)

def percent_rank(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True, method="average")

def ttc_to_hours(s: pd.Series) -> pd.Series:
    """Coerce 'Time to Close' to hours."""
    s = s.astype("string").str.strip()
    h_num = pd.to_numeric(s, errors="coerce")
    mask = h_num.isna()
    td = pd.to_timedelta(s[mask], errors="coerce")
    h_td = (td / np.timedelta64(1, "h")).astype("float")
    hours = h_num.copy()
    hours[mask] = h_td
    hours = hours.where(hours >= 0)
    if hours.notna().sum() > 5:
        p99 = np.nanpercentile(hours, 99)
        hours = hours.where(hours <= p99)
    return hours

def compute_reopen(df, closer_col, reopened_col):
    """
    Compute reopen metrics per closer on the *entire dataset*.

    - A ticket is 'reopened' if ReopenedAt is non-null.
    - 'reopen_rate_raw' = reopens / tickets_closed.
    - 'reopen_rate' = Bayesian-smoothed rate using Beta(alpha, beta) prior to
      reduce volatility for small sample sizes.
    - 'penalty' = percentile(reopen_rate) * MAX_PENALTY  (0..15).
    """
    tmp = df.copy()
    if reopened_col in tmp.columns:
        tmp[reopened_col] = pd.to_datetime(tmp[reopened_col], errors="coerce", dayfirst=True)
        tmp["_reopened_flag"] = tmp[reopened_col].notna()
    else:
        tmp["_reopened_flag"] = False

    quality = (
        tmp.groupby(closer_col)
           .agg(tickets_closed=("_reopened_flag", "size"),
                reopens=("_reopened_flag", "sum"))
           .reset_index()
    )

    # Raw rate
    quality["reopen_rate_raw"] = (
        quality["reopens"] / quality["tickets_closed"].replace(0, np.nan)
    ).fillna(0.0)

    # Smoothed rate (Beta prior)
    a, b = REOPEN_PRIOR_ALPHA, REOPEN_PRIOR_BETA
    quality["reopen_rate"] = (quality["reopens"] + a) / (quality["tickets_closed"] + a + b)

    # Penalty 0..15 by percentile of smoothed rate
    quality["penalty"] = (percent_rank(quality["reopen_rate"]) * MAX_PENALTY).round(1)

    return quality

def score_closer_table(
    df, closer_col, client_col, ttc_col, reopened_col, status_col, hours_col=None
):
    d = df.copy()
    d[closer_col] = clean_people(d[closer_col])
    d = d[d[closer_col].notna()]
    d = d[~d[closer_col].str.lower().isin(DROP_NAMES)]

    if hours_col is None:
        if not np.issubdtype(d[ttc_col].dtype, np.number):
            d[ttc_col] = pd.to_numeric(d[ttc_col], errors="coerce")
        hours_col = ttc_col

    # --- Volume pillars
    vol = d.groupby(closer_col).size().reset_index(name="Tickets Closed")

    # unique clients & per-client throughput
    uniq_clients = d.groupby(closer_col)[client_col].nunique().reset_index(name="Unique Clients")
    per_client = vol.merge(uniq_clients, on=closer_col, how="left")
    per_client["Closed per Client"] = (
        per_client["Tickets Closed"] / per_client["Unique Clients"].replace(0, np.nan)
    )

    # --- Speed (exclude pending/processing)
    speed_base = d.copy()
    if status_col in speed_base.columns and status_col is not None:
        mask = ~speed_base[status_col].astype(str).str.lower().isin(EXCLUDE_STATUSES_FOR_SPEED)
        speed_base = speed_base[mask]

    speed = (
        speed_base.groupby(closer_col)[hours_col].mean().reset_index()
                  .rename(columns={hours_col: "Avg Close (h)"})
    )

    # Merge pillars
    lb = (
        vol.merge(per_client[[closer_col, "Closed per Client", "Unique Clients"]], on=closer_col, how="left")
           .merge(speed, on=closer_col, how="left")
    )

    # --- Scores via percentile ranks (robust)
    lb["Volume Score (0–100)"]     = (percent_rank(lb["Tickets Closed"]) * 100).round(1)
    lb["Throughput Score (0–100)"] = (percent_rank(lb["Closed per Client"]) * 100).round(1)
    lb["Speed Score (0–100)"]      = ((1 - percent_rank(lb["Avg Close (h)"])) * 100).round(1)

    # Base power (weights)
    w_speed, w_through, w_vol = 0.30, 0.40, 0.30
    lb["POWER SCORE (0–100)"] = (
        lb["Speed Score (0–100)"] * w_speed
      + lb["Throughput Score (0–100)"] * w_through
      + lb["Volume Score (0–100)"] * w_vol
    ).round(1)

    # Re-open penalty & adjusted power (using SMOOTHED rate)
    quality = compute_reopen(d, closer_col, reopened_col)
    lb = lb.merge(
        quality[[closer_col, "reopen_rate", "reopen_rate_raw", "penalty", "reopens", "tickets_closed"]],
        on=closer_col, how="left"
    ).fillna({"reopen_rate":0.0, "reopen_rate_raw":0.0, "penalty":0.0, "reopens":0, "tickets_closed":0})

    lb["Adj POWER SCORE (0–100)"] = (lb["POWER SCORE (0–100)"] - lb["penalty"]).clip(lower=0)

    # Titles + tiers
    lb["Title"] = np.where(lb["Speed Score (0–100)"] >= 70, "Lightning Closer", "Steady Operator")
    def tier_from_score(s):
        if s >= 80: return "Mythic 🟣"
        if s >= 70: return "Legendary 🟡"
        if s >= 55: return "Epic 🔵"
        return "Rare 🟢"
    lb["Tier"] = lb["Adj POWER SCORE (0–100)"].apply(tier_from_score)

    lb = lb.rename(columns={closer_col: "Closer"})
    lb = lb.sort_values("Adj POWER SCORE (0–100)", ascending=False).reset_index(drop=True)
    lb.insert(0, "Rank", lb.index + 1)

    cols = [
        "Rank","Closer","Title","Tier",
        "Adj POWER SCORE (0–100)","penalty",
        "Speed Score (0–100)","Throughput Score (0–100)","Volume Score (0–100)",
        "Avg Close (h)","Tickets Closed","Unique Clients","Closed per Client",
        "reopens","reopen_rate"
    ]
    lb = lb[cols].rename(columns={
        "penalty":"Re-open Penalty (0–15)",
        "reopen_rate":"Re-open Rate",
        "reopens":"Reopens"
    })
    return lb

def styled_table(df: pd.DataFrame) -> str:
    s = (
        df.style
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
    html = s.to_html()
    html = html.replace("<th></th>", "")
    return html

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

def weekly_winner(df, closer_col, client_col, hrs_col, reopened_col, status_col, closed_col, created_col):
    """Return (top_row, latest_week_label) — we do not display the label anymore."""
    date_col = closed_col if closed_col in df.columns and closed_col else created_col
    if not date_col or date_col not in df.columns:
        return None, None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce", dayfirst=True)
    d = d[d[date_col].notna()]
    if d.empty:
        return None, None

    iso = d[date_col].dt.isocalendar()
    d["_week"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    latest_week = d["_week"].sort_values().iloc[-1]
    dweek = d[d["_week"] == latest_week].copy()
    if dweek.empty:
        return None, None

    lbw = score_closer_table(
        dweek, closer_col=closer_col, client_col=client_col,
        ttc_col=hrs_col, reopened_col=reopened_col, status_col=status_col,
        hours_col=hrs_col
    )
    if lbw.empty:
        return None, None
    return lbw.iloc[0], latest_week

# ==============================
# Sidebar — data load & schema
# ==============================
st.sidebar.header("Load data")
uploaded = st.sidebar.file_uploader("Upload CSV/TSV/XLSX", type=["csv","tsv","txt","xls","xlsx"])
df = load_any(uploaded)

st.title("Client Support — Closer Dashboard")

if df is None or df.empty:
    st.info(
        "👋 Please upload a file with the required headers to start.\n\n"
        "`ID, ClientID, Status, Title, Field, Service, Category, CreatedAt, UpdatedAt, "
        "AcknowledgedAt, ProcessingAt, ClosedAt, ReopenedAt, LastSimplyReplyAt, "
        "LastClientReplyAt, ClientName, Time to Close, TAM, Closer`"
    )
    st.stop()

# Validate / normalize schema
colmap, missing = resolve_schema(df)

with st.expander("Schema check", expanded=bool(missing)):
    ok = "✅"
    bad = "❌"
    st.write("\n".join(f"{ok if h in colmap else bad} {h} → {colmap.get(h,'(missing)')}"
                       for h in REQUIRED_HEADERS))
if missing:
    st.error(f"Your file is missing required columns: {', '.join(missing)}")
    st.stop()

# Resolve columns used by the app
status_col   = colmap[USED["status"]]
client_col   = colmap[USED["client"]]
tam_col      = colmap[USED["tam"]]
closer_col   = colmap[USED["closer"]]
ttc_col      = colmap[USED["ttc"]]
reopened_col = colmap[USED["reopened"]]
closed_col   = colmap[USED["closed"]]
created_col  = colmap[USED["created"]]

# Filters
st.sidebar.header("Filters")
exclude_names = st.sidebar.multiselect(
    "Exclude Closers",
    sorted(df[closer_col].dropna().astype(str).unique()),
    default=[]
)
show_top_n = st.sidebar.slider("Top N for charts", 5, 20, 15)

# Basic cleaning / exclusions  (FIXED bracket)
df[closer_col] = clean_people(df[closer_col])
df = df[~df[closer_col].str.lower().isin({n.lower() for n in exclude_names})]

# Robust “hours” for all speed/avg metrics
hrs_col = "_ttc_hours"
df[hrs_col] = ttc_to_hours(df[ttc_col])

# ==============================
# KPIs
# ==============================
with st.container():
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tickets", f"{len(df):,}")
    k2.metric("Unique Clients", f"{df[client_col].nunique():,}")
    k3.metric("Closers", f"{df[closer_col].nunique():,}")
    avg_ttc = df[hrs_col].mean()
    k4.metric("Avg Close (h)", f"{avg_ttc:.2f}" if pd.notna(avg_ttc) else "—")

st.markdown("---")

# Winner banner (no week label)
topw, _ = weekly_winner(df, closer_col, client_col, hrs_col,
                        reopened_col, status_col, closed_col, created_col)
if topw is not None:
    st.markdown(
        f"""
        <div style="margin:8px 0 18px; padding:14px 18px; background:#0ea5e9; color:white;
                    border-radius:14px; box-shadow:0 8px 18px rgba(0,0,0,.10);">
          <div style="font-weight:800; font-size:18px;">🏁 Winner</div>
          <div style="display:flex; gap:22px; align-items:baseline; flex-wrap:wrap;">
            <div style="font-size:30px; font-weight:800;">{topw['Closer']}</div>
            <div>Adj Power: <b>{topw['Adj POWER SCORE (0–100)']:.1f}</b></div>
            <div>Tickets: <b>{int(topw['Tickets Closed'])}</b></div>
            <div>Avg Close (h): <b>{topw['Avg Close (h)']:.2f}</b></div>
            <div>Closed/Client: <b>{topw['Closed per Client']:.2f}</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================
# Tabs
# ==============================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "🏆 Leaderboard", "🎖 Achievements", "🔄 Re-opens"])

# ---------- Charts ----------
with tab1:
    c1, c2 = st.columns(2)

    # Top clients by tickets
    clients_df = (
        df.groupby(client_col).size().reset_index(name="tickets_count")
          .sort_values("tickets_count", ascending=False).head(show_top_n)
    )
    fig_clients = px.bar(
        clients_df, x="tickets_count", y=client_col, orientation="h", color="tickets_count",
        title="👑 Which client generated the most tickets?", labels={"tickets_count":"Tickets", client_col:"Client"},
        template="simple_white",
    )
    fig_clients.update_yaxes(categoryorder="total ascending")
    c1.plotly_chart(fig_clients, use_container_width=True)

    # Closers by tickets closed (exclude pending/processing)
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
    c2.plotly_chart(fig_closer, use_container_width=True)

    # Fastest closers (avg hours) — exclude pending/processing
    avg_df = (
        df_closed.groupby(closer_col)[hrs_col].mean().reset_index()
                 .rename(columns={hrs_col:"avg_hours"})
                 .sort_values("avg_hours", ascending=True).head(show_top_n)
    )
    fig_speed = px.bar(
        avg_df, x=closer_col, y="avg_hours", color="avg_hours",
        title="⚡ Fastest Closers (Avg Time to Close — lower is better)",
        labels={"avg_hours":"Avg hours", closer_col:"Closer"},
        template="simple_white",
    )
    st.plotly_chart(fig_speed, use_container_width=True)

# ---------- Leaderboard ----------
with tab2:
    lb = score_closer_table(
        df, closer_col=closer_col, client_col=client_col,
        ttc_col=ttc_col, reopened_col=reopened_col, status_col=status_col,
        hours_col=hrs_col
    )
    st.subheader("🏆 Power Leaderboard (Closer) — with Re-open Penalty & Normalized Scores")
    st.caption("Adjusted Power = 0.3·Speed + 0.4·Per-Client + 0.3·Volume − Penalty (0–15).")
    st.markdown(styled_table(lb), unsafe_allow_html=True)

    with st.expander("Show raw table"):
        st.dataframe(lb, use_container_width=True)

# ---------- Achievements (robust dynamic layout) ----------
with tab3:
    lb = score_closer_table(
        df, closer_col=closer_col, client_col=client_col,
        ttc_col=ttc_col, reopened_col=reopened_col, status_col=status_col,
        hours_col=hrs_col
    )
    if lb.empty:
        st.info("No data to compute achievements yet.")
    else:
        top3 = lb.head(3).reset_index(drop=True)
        cols = st.columns(len(top3))
        for i in range(len(top3)):
            with cols[i]:
                achievement_card(top3.iloc[i], i + 1)

    st.markdown(
        '<div class="note">Tip: use the sidebar to exclude names or change Top-N on charts.</div>',
        unsafe_allow_html=True
    )

# ---------- Re-opens (consistent with penalty) ----------
with tab4:
    quality = compute_reopen(df, closer_col, reopened_col)

    # Overall *smoothed* re-open rate (matches the leaderboard penalty)
    rate_df = (quality[["tickets_closed", "reopens", "reopen_rate", closer_col]]
               .rename(columns={closer_col:"Closer"})
               .sort_values("reopen_rate", ascending=False)
               .head(show_top_n))

    fig_rate = px.bar(
        rate_df, x="reopen_rate", y="Closer", orientation="h", color="reopen_rate",
        title="🔄 Re-open RATE (entire dataset, smoothed)",
        labels={"reopen_rate":"Re-open rate"},
        template="simple_white",
    )
    fig_rate.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig_rate, use_container_width=True)

    # Optional: latest-week *counts*
    tmp = df.copy()
    if reopened_col:
        tmp[reopened_col] = pd.to_datetime(tmp[reopened_col], errors="coerce", dayfirst=True)
        tmp = tmp[tmp[reopened_col].notna()]
        if not tmp.empty:
            iso = tmp[reopened_col].dt.isocalendar()
            tmp["_week"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
            latest_week = tmp["_week"].max()
            r_week = (tmp[tmp["_week"] == latest_week]
                      .groupby(closer_col).size()
                      .reset_index(name="tickets_reopened")
                      .rename(columns={closer_col:"Closer"})
                      .sort_values("tickets_reopened", ascending=False))
            if not r_week.empty:
                fig_week = px.bar(
                    r_week, x="tickets_reopened", y="Closer", orientation="h", color="tickets_reopened",
                    title=f"🗓️ Re-open COUNT — latest week: {latest_week}",
                    labels={"tickets_reopened":"Tickets Reopened"},
                    template="simple_white",
                )
                fig_week.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig_week, use_container_width=True)

    st.caption(
        "Notes: A ticket counts as reopened if **ReopenedAt** is non-empty. "
        "Leaderboard penalty uses the **smoothed re-open rate across the entire dataset** "
        "(Beta prior ~5%) to avoid over-penalizing tiny sample sizes."
    )

