import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st

from data_loader import (
    load_data,
    train_test_split,
    possible_date_cols,
)
from models import (
    run_autoarima,
    run_prophet,
    run_prophet_regressors,
    run_random_forest,
    run_naive,
    forecast_future_arima,
    forecast_future_prophet,
    calc_metrics,
)
from indicators import apply_indicators, win_size
from charts import build_main_chart, build_metrics_bar_chart, build_residuals_chart


# PAGE CONFIG

st.set_page_config(
    page_title="BTC Forecasting Portal",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# GLOBAL STYLES

st.markdown("""
<style>
@import url('https://fonts.cdnfonts.com/css/jetbrains-mono');

/* ── Base ────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
}
.stApp {
    background-color: #0a0b12;
    background-image:
        radial-gradient(ellipse 80% 60% at 15% 40%, rgba(246,166,35,0.04) 0%, transparent 70%),
        radial-gradient(ellipse 60% 50% at 85% 15%, rgba(56,178,172,0.03) 0%, transparent 70%);
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0d1017 !important;
    border-right: 1px solid #1c2333 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* ── Sidebar section label ───────────────────────────────── */
.sb-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3b4a65;
    padding: 0.6rem 0 0.25rem;
    border-top: 1px solid #1c2333;
    margin-top: 0.4rem;
}
.sb-section:first-child { border-top: none; margin-top: 0; }

/* ── Page title ──────────────────────────────────────────── */
.portal-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    background: linear-gradient(125deg, #f6a623 0%, #fcd34d 45%, #f6a623 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.portal-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 400;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3b4a65;
    margin-top: 0.35rem;
}

/* ── Stat cards ──────────────────────────────────────────── */
.stat-card {
    background: #0d1117;
    border: 1px solid #1c2333;
    border-radius: 10px;
    padding: 1.1rem 1.3rem 0.9rem;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    inset: 0 0 auto 0;
    height: 2px;
    background: linear-gradient(90deg, #f6a623, #fcd34d 60%, transparent);
    border-radius: 10px 10px 0 0;
}
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3b4a65;
    margin-bottom: 0.45rem;
}
.stat-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #dce4f0;
    line-height: 1.2;
}
.stat-note {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    color: #3b4a65;
    margin-top: 0.3rem;
}
.stat-good { color: #34d399; }
.stat-bad  { color: #f87171; }

/* ── Section header ──────────────────────────────────────── */
.section-hdr {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin: 0.3rem 0 0.9rem;
}
.section-hdr-line {
    flex: 1;
    height: 1px;
    background: #1c2333;
}
.section-hdr-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.64rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #f6a623;
    white-space: nowrap;
}

/* ── Alert boxes ─────────────────────────────────────────── */
.alert {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.90rem;
    line-height: 1.55;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
}
.alert-info  { background:#0d1a2d; border-left:3px solid #3b82f6; color:#7ca3d8; }
.alert-warn  { background:#1a1200; border-left:3px solid #f6a623; color:#c89040; }
.alert-error { background:#1a0a0a; border-left:3px solid #f87171; color:#d97070; }

/* ── Feature importance bars ─────────────────────────────── */
.feat-row { margin: 0.45rem 0; }
.feat-head {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.69rem;
    color: #8b95a9;
    margin-bottom: 4px;
}
.feat-track {
    background: #1c2333;
    border-radius: 3px;
    height: 5px;
    width: 100%;
    overflow: hidden;
}
.feat-fill {
    height: 5px;
    border-radius: 3px;
    background: linear-gradient(90deg, #f6a623, #fcd34d);
}

/* ── Metric-comparison table row ─────────────────────────── */
.model-row {
    background: #0d1117;
    border: 1px solid #1c2333;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.5rem;
}
.model-row-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    color: #d4dbe8;
    margin-bottom: 0.6rem;
}

/* ── Upload drop zone (center page) ─────────────────────── */
.upload-zone-wrap {
    display: flex;
    justify-content: center;
    padding: 3rem 0 2rem;
}
.upload-zone {
    background: #0d1117;
    border: 1.5px dashed #2a3a5c;
    border-radius: 16px;
    padding: 2.8rem 3.5rem 2.5rem;
    text-align: center;
    width: 100%;
    max-width: 560px;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #f6a623; }
.upload-zone-icon {
    font-size: 3rem;
    line-height: 1;
    margin-bottom: 1rem;
    opacity: 0.35;
}
.upload-zone-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #2a3a5c;
    margin-bottom: 0.4rem;
}
.upload-zone-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #1c2a42;
    letter-spacing: 0.07em;
    margin-bottom: 1.6rem;
}

/* ── Streamlit widget overrides ──────────────────────────── */
div[data-testid="stFileUploader"] {
    border: 1px dashed #2a3a5c !important;
    border-radius: 8px !important;
    background: #0d1117 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #f6a623, #e8930f) !important;
    color: #0a0b12 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 0 0 0 rgba(246,166,35,0) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(246,166,35,0.28) !important;
}
.stButton > button:disabled {
    background: #1c2333 !important;
    color: #3b4a65 !important;
    box-shadow: none !important;
    transform: none !important;
}
.stSelectbox label, .stMultiSelect label, .stSlider label,
.stCheckbox label, .stRadio label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.79rem !important;
    color: #8b95a9 !important;
}
.stSlider [data-testid="stThumbValue"] {
    background: #f6a623 !important;
}
hr { border-color: #1c2333 !important; margin: 0.75rem 0 !important; }
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1c2333;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.69rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3b4a65;
    border-radius: 0;
    padding: 0.65rem 1.3rem;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #f6a623 !important;
    border-bottom: 2px solid #f6a623 !important;
    background: transparent !important;
}
.stDataFrame { border: 1px solid #1c2333 !important; border-radius: 8px !important; }
.stProgress > div > div {
    background: linear-gradient(90deg, #f6a623, #fcd34d) !important;
}
</style>
""", unsafe_allow_html=True)


# UI HELPERS

def section_header(text: str):
    st.markdown(f"""
    <div class="section-hdr">
        <div class="section-hdr-text">{text}</div>
        <div class="section-hdr-line"></div>
    </div>""", unsafe_allow_html=True)


def stat_card(label: str, value: str, note: str = "",
              note_class: str = ""):
    note_html = f'<div class="stat-note {note_class}">{note}</div>' if note else ""
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {note_html}
    </div>""", unsafe_allow_html=True)


def alert(text: str, kind: str = "info"):
    cls = {"info": "alert-info", "warn": "alert-warn", "error": "alert-error"}.get(kind, "alert-info")
    st.markdown(f'<div class="alert {cls}">{text}</div>', unsafe_allow_html=True)


def feat_bar(label: str, pct: float):
    st.markdown(f"""
    <div class="feat-row">
        <div class="feat-head"><span>{label}</span><span>{pct:.1f}%</span></div>
        <div class="feat-track"><div class="feat-fill" style="width:{pct:.1f}%"></div></div>
    </div>""", unsafe_allow_html=True)


def metric_row(model_name: str, m: dict):
    mase_class = "stat-good" if m["MASE"] < 1 else "stat-bad"
    mase_label = "↓ beats naive" if m["MASE"] < 1 else "↑ worse than naive"
    st.markdown(f"""
    <div class="model-row">
        <div class="model-row-name">{model_name}</div>
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:0.6rem;">
            <div>
                <div class="stat-label">MAE</div>
                <div class="stat-value" style="font-size:1.1rem">${m['MAE']:,.0f}</div>
            </div>
            <div>
                <div class="stat-label">RMSE</div>
                <div class="stat-value" style="font-size:1.1rem">${m['RMSE']:,.0f}</div>
            </div>
            <div>
                <div class="stat-label">MAPE</div>
                <div class="stat-value" style="font-size:1.1rem">{m['MAPE']:.2f}%</div>
            </div>
            <div>
                <div class="stat-label">MASE</div>
                <div class="stat-value {mase_class}" style="font-size:1.1rem">{m['MASE']:.4f}</div>
                <div class="stat-note {mase_class}">{mase_label}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# SESSION STATE INIT

for key, default in [
    ("df", None),
    ("train", None),
    ("test", None),
    ("results", None),
    ("importances", {}),
    ("future_preds", None),
    ("load_warn", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# SIDEBAR

with st.sidebar:

    # Logo
    st.markdown("""
    <div style="padding:1rem 0 3.2rem; text-align:center;">
        <div style="font-family: 'Plus Jakarta Sans'; font-size:2.6rem; line-height:1; margin-bottom:0.4rem; color:#f6a623">₿</div>
        <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1rem;
                    font-weight:800; letter-spacing:-0.02em; color:#f6a623">
            BTC Portal
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem;
                    letter-spacing:0.15em; text-transform:uppercase;
                    color:#3b4a65; margin-top:3px;">
            Forecasting Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    price_col = st.selectbox(
            "Price column",
            options=["Close", "Open", "Price", "High", "Low"],
            index=0,
            help="Which OHLC column to treat as the price series.",
        )

    # Models
    st.markdown('<div class="sb-section">Models</div>', unsafe_allow_html=True)

    model_options = ["AutoARIMA", "Prophet", "Prophet+Regressors",
                    "RandomForest", "Naive"]

    st.markdown(
        '<div style="font-size:0.75rem; color:#8b95a9; margin-bottom:0.5rem;">Select models</div>',
        unsafe_allow_html=True
    )

    selected_models = []

    # Default selections
    default_models = ["AutoARIMA", "Prophet"]

    for model in model_options:
        checked = st.checkbox(
            model,
            value=(model in default_models),
            key=f"model_{model}"
        )
        if checked:
            selected_models.append(model)
        

    # Forecast Parameters
    st.markdown('<div class="sb-section">Forecast Parameters</div>', unsafe_allow_html=True)

    test_days = st.slider(
        "Backtest window (days)",
        min_value=30, max_value=180, value=90, step=10,
        help="The last N days of the dataset held out to measure prediction accuracy.",
    )

    horizon = st.slider(
        "Future horizon (days)",
        min_value=7, max_value=180, value=30, step=7,
        help="How many calendar days beyond the dataset to project forward.",
    )

    ci_pct = st.select_slider(
        "Confidence interval",
        options=[80, 90, 95],
        value=95,
        help="Width of the uncertainty band shown around the forecast.",
    )
    interval_width = ci_pct / 100.0

    # Technical Indicators
    st.markdown('<div class="sb-section">Technical Indicators</div>', unsafe_allow_html=True)

    selected_indicators = st.multiselect(
        "Moving averages (SMA)",
        options=list(win_size.keys()),
        default=[],
        help="Simple Moving Averages overlaid on the price chart",
    )

    # Run
    st.markdown('<div class="sb-section">Run</div>', unsafe_allow_html=True)

    # uploaded_file is defined in the main area below; use session state to track it
    _file_ready = st.session_state.get("df") is not None or False
    disabled = (not _file_ready) or len(selected_models) == 0
    run_btn = st.button("Generate Forecast", disabled=disabled)

    if not _file_ready:
        st.markdown('<div class="alert alert-warn" style="margin-top:0.6rem;">Upload a CSV first !</div>',
                    unsafe_allow_html=True)
    elif not selected_models:
        st.markdown('<div class="alert alert-warn" style="margin-top:0.6rem;">Select at least one model</div>',
                    unsafe_allow_html=True)


# HEADER

col_h, col_sub = st.columns([2, 1])
with col_h:
    st.markdown(
        '<div class="portal-title">₿itcoin Price Forecasting App</div>'
        '<div class="portal-sub">Time-Series Analysis Project</div>',
        unsafe_allow_html=True,
    )
# with col_sub:
#     st.markdown("<br>", unsafe_allow_html=True)
#     alert(
#         "Upload your BTC CSV below to start.",
#         kind="info",
#     )

st.divider()


# CENTER-PAGE FILE UPLOADER  (always visible until a file is loaded)

if st.session_state.df is None:
    # Decorative wrapper
    st.markdown("""
    <div class="upload-zone-wrap">
      <div class="upload-zone">
        <div class="upload-zone-icon">₿</div>
        <div class="upload-zone-title">Upload your BTC dataset</div>
        <div class="upload-zone-sub">Accepts any Kaggle-style CSV with Date + OHLCV columns</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Real Streamlit uploader — centred with columns trick
    _left, _mid, _right = st.columns([1, 2, 1])
    with _mid:
        uploaded_file = st.file_uploader(
            "Upload BTC CSV",
            type=["csv"],
            help="Any Kaggle-style BTC historical CSV — needs a date column and at least one of Open / High / Low / Close.",
            label_visibility="collapsed",
        )

        # price_col = st.selectbox(
        #     "Price column",
        #     options=["Close", "Open", "High", "Low"],
        #     index=0,
        #     help="Which OHLC column to treat as the price series.",
        # )
else:
    # Once data is loaded, keep these variables available for re-runs
    # but don't show the uploader again (show a small swap option instead)
    uploaded_file = None
    price_col     = "Close"          # default; dataset is already cached

    _left, _mid, _right = st.columns([1, 2, 1])

    with _mid:

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("↩ Load a different CSV", key="reload_csv"):
                for _k in ["df", "train", "test", "results",
                            "importances", "future_preds", "load_warn"]:
                    st.session_state[_k] = None if _k not in ["importances", "load_warn"] else ({} if _k == "importances" else [])
                st.rerun()


# DATA LOADING

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        df, load_warnings = load_data(file_bytes, price_col)
        st.session_state.df        = df
        st.session_state.load_warn = load_warnings
        st.rerun()   # re-run so the uploader collapses and sidebar enables

    except ValueError as exc:
        alert(f"✕  {exc}", kind="error")
        st.stop()

# From here on, work exclusively from session state
if st.session_state.df is not None:

    # Data warnings
    for w in st.session_state.load_warn:
        alert(f"⚠  {w}", kind="warn")

    # Dataset stats
    df = st.session_state.df
    section_header("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Rows", f"{len(df):,}", f"{df['ds'].min().date()} → {df['ds'].max().date()}")
    with c2:
        stat_card("Latest Price", f"${df['y'].iloc[-1]:,.0f}")
    with c3:
        stat_card("All-time High", f"${df['y'].max():,.0f}")
    with c4:
        stat_card("Volume Data",
                  "Available" if "volume" in df.columns else "Not found",
                  note_class="stat-good" if "volume" in df.columns else "stat-bad")


# RUN FORECAST

if run_btn and st.session_state.df is not None and len(selected_models) > 0:

    df = st.session_state.df

    try:
        train, test = train_test_split(df, test_days)
    except ValueError as exc:
        alert(f"✕  Split error: {exc}", kind="error")
        st.stop()

    st.session_state.train = train
    st.session_state.test  = test

    backtest_results: dict = {}
    importances: dict = {}

    prog = st.progress(0, text="Initialising…")
    n = len(selected_models)

    for i, model_name in enumerate(selected_models):
        prog.progress(int(i / n * 85), text=f"Training {model_name}…")
        try:
            if model_name == "AutoARIMA":
                y_true, y_pred, conf_lo, conf_hi, dates = run_autoarima(train, test)
                backtest_results[model_name] = dict(
                    y_true=y_true, y_pred=y_pred,
                    conf_lo=conf_lo, conf_hi=conf_hi, dates=dates,
                    metrics=calc_metrics(y_true, y_pred, train["y"].values),
                )

            elif model_name == "Prophet":
                y_true, y_pred, conf_lo, conf_hi, dates = run_prophet(
                    train, test, interval_width
                )
                backtest_results[model_name] = dict(
                    y_true=y_true, y_pred=y_pred,
                    conf_lo=conf_lo, conf_hi=conf_hi, dates=dates,
                    metrics=calc_metrics(y_true, y_pred, train["y"].values),
                )

            elif model_name == "Prophet+Regressors":
                y_true, y_pred, conf_lo, conf_hi, dates = run_prophet_regressors(
                    train, test, df, interval_width
                )
                backtest_results[model_name] = dict(
                    y_true=y_true, y_pred=y_pred,
                    conf_lo=conf_lo, conf_hi=conf_hi, dates=dates,
                    metrics=calc_metrics(y_true, y_pred, train["y"].values),
                )

            elif model_name == "RandomForest":
                y_true, y_pred, dates, feat_imp = run_random_forest(train, test, df)
                backtest_results[model_name] = dict(
                    y_true=y_true, y_pred=y_pred,
                    conf_lo=None, conf_hi=None, dates=dates,
                    metrics=calc_metrics(y_true, y_pred, train["y"].values),
                )
                importances = feat_imp

            elif model_name == "Naive":
                y_true, y_pred, dates = run_naive(train, test)
                backtest_results[model_name] = dict(
                    y_true=y_true, y_pred=y_pred,
                    conf_lo=None, conf_hi=None, dates=dates,
                    metrics=calc_metrics(y_true, y_pred, train["y"].values),
                )

        except Exception as exc:
            alert(f"⚠  {model_name} failed: {exc}", kind="warn")

    # Future forecast
    future_preds = None
    if backtest_results:
        best_model = min(backtest_results, key=lambda m: backtest_results[m]["metrics"]["MAE"])
        prog.progress(90, text=f"Generating {horizon}-day future forecast…")
        try:
            if best_model == "AutoARIMA":
                f_pred, f_lo, f_hi, f_dates = forecast_future_arima(df, horizon)
            else:
                f_pred, f_lo, f_hi, f_dates = forecast_future_prophet(df, horizon, interval_width)
            future_preds = (f_dates, f_pred, f_lo, f_hi)
        except Exception as exc:
            alert(f"⚠  Future forecast failed: {exc}", kind="warn")

    prog.progress(100, text="Done!")
    prog.empty()

    st.session_state.results = backtest_results
    st.session_state.importances = importances
    st.session_state.future_preds = future_preds


# RESULTS DISPLAY

if st.session_state.results is not None and st.session_state.df is not None:

    results = st.session_state.results
    df = st.session_state.df
    train = st.session_state.train
    test = st.session_state.test
    future_preds = st.session_state.future_preds

    # Apply SMA indicators to the full dataset for chart overlay
    indicator_df = apply_indicators(df.copy(), selected_indicators)

    st.divider()

    # Tabs
    tab_chart, tab_metrics, tab_data = st.tabs([
        "📈   Forecast Chart",
        "📊   Model Metrics",
        "🗃   Raw Data",
    ])

    # TAB 1 — Forecast Chart
    with tab_chart:
        section_header("Interactive Price Forecast")

        fig = build_main_chart(
            df=df,
            train=train,
            test=test,
            backtest_results=results,
            indicator_df=indicator_df,
            selected_models=list(results.keys()),
            future_preds=future_preds,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Future forecast summary cards
        if future_preds is not None:
            f_dates, f_pred, f_lo, f_hi = future_preds
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Future Forecast Summary")
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                stat_card("Forecast Horizon", f"{len(f_dates)} days")
            with sc2:
                stat_card("End-of-Horizon Price",
                          f"${f_pred[-1]:,.0f}",
                          str(pd.Timestamp(f_dates[-1]).date()))
            with sc3:
                stat_card(f"Lower {ci_pct}% CI", f"${f_lo[-1]:,.0f}")
            with sc4:
                stat_card(f"Upper {ci_pct}% CI", f"${f_hi[-1]:,.0f}")

        # Residuals
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Residuals — Actual minus Predicted")
        fig_res = build_residuals_chart(results, list(results.keys()))
        st.plotly_chart(fig_res, use_container_width=True)

    # TAB 2 — Metrics
    with tab_metrics:
        section_header("Backtest Accuracy Metrics")

        alert(
            "<b>MAE</b> — Mean Absolute Error in USD &nbsp;|&nbsp; "
            "<b>RMSE</b> — Root Mean Squared Error (large errors penalised more) &nbsp;|&nbsp; "
            "<b>MAPE</b> — Mean Absolute Percentage Error &nbsp;|&nbsp; "
            "<b>MASE</b> — Scaled Error vs. naïve baseline  (< 1 = better than naïve)",
            kind="info",
        )
        st.markdown("<br>", unsafe_allow_html=True)

        for model_name, result in results.items():
            metric_row(model_name, result["metrics"])

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Error Comparison Chart")
        st.plotly_chart(build_metrics_bar_chart(results), use_container_width=True)

        # Feature importances
        if st.session_state.importances:
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Random Forest — Feature Importances")
            imp = st.session_state.importances
            total = sum(imp.values())
            for feat, val in sorted(imp.items(), key=lambda x: -x[1]):
                feat_bar(feat, val / total * 100)

    # TAB 3 — Raw Data
    with tab_data:
        section_header("Full Dataset")

        display_df = df.copy()
        display_df["ds"] = display_df["ds"].dt.date
        display_df.columns = [c.upper() for c in display_df.columns]
        st.dataframe(
            display_df.sort_values("DS", ascending=False),
            use_container_width=True,
            height=420,
        )