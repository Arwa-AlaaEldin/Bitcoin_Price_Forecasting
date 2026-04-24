import numpy as np
import pandas as pd
import plotly.graph_objects as go

from indicators import get_indicator_columns

# colour palette
PAL = {
    "bg":"#0d1117",
    "paper":"#0a0b12",
    "grid": "#1c2333",
    "text":"#8b95a9",
    "train":"#1e2d4d",
    "actual":"#f6a623",
    "future":"#38b2ac",
    "ci_future": "rgba(56,178,172,0.10)",

    # model colors
    "AutoARIMA": "#4ade80",
    "Prophet": "#818cf8",
    "Prophet+Regressors": "#c084fc",
    "RandomForest": "#fb923c",
    "Naive": "#f87171",

    # SMA line colors
    "SMA_20": "#fbbf24",
    "SMA_50": "#34d399",
    "SMA_200": "#60a5fa",
}

# Dash patterns for each model to be distinguishable even in greyscale
MODEL_DASH = {
    "AutoARIMA": "dot",
    "Prophet": "dot",
    "Prophet+Regressors": "dot",
    "RandomForest": "dot",
    "Naive": "dot",
}

# Base layout shared by every chart
_BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=PAL["paper"],
    plot_bgcolor=PAL["bg"],
    font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif",
              color=PAL["text"], size=11),
    margin=dict(l=12, r=12, t=52, b=12),
    hoverlabel=dict(
        bgcolor="#16213e",
        bordercolor="#2a3a5c",
        font_family="Plus Jakarta Sans, Segoe UI, sans-serif",
        font_size=12,
    ),
)


# return a filled CI band trace (used for both backtest and future CI)
def _ci_band(dates, lo, hi, fill_color, name=""):
    x = np.concatenate([dates, dates[::-1]])
    y = np.concatenate([hi, lo[::-1]])
    return go.Scatter(
        x=x, y=y, fill="toself",
        fillcolor=fill_color,
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
        name=name,
    )


# convert a hex color rgba
def _model_ci_color(hex_color, alpha=0.10):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# build the forecast chart
def build_main_chart(df, train, test, backtest_results, indicator_df, selected_models, future_preds=None):
   
    fig = go.Figure()

    # 1. Training region background
    fig.add_vrect(
        x0=str(train["ds"].min()),
        x1=str(train["ds"].max()),
        fillcolor="rgba(30,45,77,0.12)",
        line_width=0,
        annotation_text="TRAIN",
        annotation_position="top left",
        annotation=dict(font=dict(size=9, color="#2a3a5c",
                      family="Plus Jakarta Sans, Segoe UI, sans-serif"),),
    )

    # 2. Historical price
    fig.add_trace(go.Scatter(
        x=train["ds"], y=train["y"],
        name="Historical Price",
        mode="lines",
        line=dict(color=PAL["train"], width=1.8),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>$%{y:,.0f}<extra>Historical</extra>",
    ))

    # 3. SMA
    sma_cols = get_indicator_columns(indicator_df)
    for col in sma_cols:
        window   = int(col.split("_")[1])
        sma_vals = indicator_df.set_index("ds")[col].reindex(df["ds"]).values
        color   = PAL.get(col, "#94a3b8")
        fig.add_trace(go.Scatter(
            x=df["ds"], y=sma_vals,
            name=f"SMA {window}d",
            mode="lines",
            line=dict(color=color, width=1.4, dash="dot"),
            opacity=0.75,
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>SMA{window}: $%{{y:,.0f}}<extra></extra>",
        ))

    # 4. Actual test prices
    fig.add_trace(go.Scatter(
        x=test["ds"], y=test["y"],
        name="Actual Price",
        mode="lines",
        line=dict(color=PAL["actual"], width=2.8),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>$%{y:,.0f}<extra>Actual</extra>",
    ))

    # 5. Model CI bands + prediction lines
    for model_name, result in backtest_results.items():
        if model_name not in selected_models:
            continue

        color = PAL.get(model_name, "#94a3b8")
        dash = MODEL_DASH.get(model_name, "dot")
        dates = result["dates"]
        y_pred = result["y_pred"]
        conf_lo = result.get("conf_lo")
        conf_hi = result.get("conf_hi")

        if conf_lo is not None and conf_hi is not None:
            fig.add_trace(_ci_band(
                dates, conf_lo, conf_hi,
                fill_color=_model_ci_color(color, 0.08),
                name=f"{model_name} CI",
            ))

        fig.add_trace(go.Scatter(
            x=dates, y=y_pred,
            name=model_name,
            mode="lines",
            line=dict(color=color, width=2.0, dash=dash),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>${{y:,.0f}}<extra>{model_name}</extra>",
        ))

    # 7. Future forecast
    if future_preds is not None:
        f_dates, f_pred, f_lo, f_hi = future_preds

        fig.add_trace(_ci_band(
            f_dates, f_lo, f_hi,
            fill_color=PAL["ci_future"],
            name="Future CI",
        ))

        fig.add_trace(go.Scatter(
            x=f_dates, y=f_pred,
            name="Future Forecast",
            mode="lines",
            line=dict(color=PAL["future"], width=2.5),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>$%{y:,.0f}<extra>Future Forecast</extra>",
        ))


        forecast_start = pd.to_datetime(f_dates[0])

        fig.add_shape(
            type="line",
            x0=forecast_start,
            x1=forecast_start,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                color=PAL["future"],
                width=1.5,
                dash="dash",
            ),
        )

        fig.add_annotation(
            x=forecast_start,
            y=1,
            xref="x",
            yref="paper",
            text="FORECAST START",
            showarrow=False,
            yshift=10,
            font=dict(size=9, color=PAL["future"]),
        )

    # Layout
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(
            text="₿  BTC / USD — Price History & Forecast",
            font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif",
                      size=15, color="#d4dbe8"),
            x=0.01, y=0.98,
        ),
        xaxis=dict(
            gridcolor=PAL["grid"], zeroline=False,
            showspikes=True, spikecolor="#2a3a5c", spikethickness=1,
            rangeslider=dict(
                visible=True,
                bgcolor="#0d1117",
                bordercolor="#1c2333",
                thickness=0.04,
            ),
        ),
        yaxis=dict(
            gridcolor=PAL["grid"], zeroline=False,
            tickprefix="$", tickformat=",.0f",
            showspikes=True, spikecolor="#2a3a5c", spikethickness=1,
        ),
        legend=dict(
            bgcolor="#111827",
            bordercolor="#1c2333",
            borderwidth=1,
            font=dict(size=10),
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="left",   x=0.0,
        ),
        hovermode="x unified",
        height=520,
    )
    return fig



# Grouped bar chart comparing MAE and RMSE (in USD) across all models
def build_metrics_bar_chart(results):

    models = list(results.keys())
    maes   = [results[m]["metrics"]["MAE"]  for m in models]
    rmses  = [results[m]["metrics"]["RMSE"] for m in models]

    fig = go.Figure([
        go.Bar(
            name="MAE",
            x=models, y=maes,
            marker_color="#f6a623",
            marker_line_color=PAL["bg"], marker_line_width=1,
            text=[f"${v:,.0f}" for v in maes],
            textposition="outside",
            textfont=dict(size=10, color="#d4dbe8"),
        ),
        go.Bar(
            name="RMSE",
            x=models, y=rmses,
            marker_color="#38b2ac",
            marker_line_color=PAL["bg"], marker_line_width=1,
            text=[f"${v:,.0f}" for v in rmses],
            textposition="outside",
            textfont=dict(size=10, color="#d4dbe8"),
        ),
    ])

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(
            text="Model Error Comparison — MAE & RMSE (USD)",
            font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif",
                      size=13, color="#d4dbe8"),
            x=0.01,
        ),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
        xaxis=dict(gridcolor=PAL["grid"]),
        yaxis=dict(
            gridcolor=PAL["grid"],
            tickprefix="$", tickformat=",.0f",
            title=dict(text="Error (USD)", font=dict(size=11)),
        ),
        legend=dict(
            bgcolor="#111827", bordercolor="#1c2333", borderwidth=1,
        ),
        height=340,
    )
    return fig



# Line chart of residuals (actual − predicted) for each model
def build_residuals_chart(backtest_results, selected_models):
    
    fig = go.Figure()

    for model_name, result in backtest_results.items():
        if model_name not in selected_models:
            continue

        residuals = result["y_true"] - result["y_pred"]
        color = PAL.get(model_name, "#94a3b8")
        dash = MODEL_DASH.get(model_name, "dot")

        fig.add_trace(go.Scatter(
            x=result["dates"], y=residuals,
            name=model_name,
            mode="lines",
            line=dict(color=color, width=1.8, dash=dash),
            hovertemplate=(
                f"<b>%{{x|%d %b %Y}}</b><br>"
                f"Residual: $%{{y:,.0f}}<extra>{model_name}</extra>"
            ),
        ))

    # Zero line — ideal residual
    fig.add_hline(
        y=0,
        line_color="rgba(255,255,255,0.15)",
        line_width=1.5,
        annotation_text="zero error",
        annotation=dict(font=dict(size=9, color="rgba(255,255,255,0.25)")),
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(
            text="Residuals — Actual minus Predicted (USD)",
            font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif",
                      size=13, color="#d4dbe8"),
            x=0.01,
        ),
        xaxis=dict(gridcolor=PAL["grid"], zeroline=False),
        yaxis=dict(
            gridcolor=PAL["grid"], zeroline=False,
            tickprefix="$", tickformat=",.0f",
        ),
        legend=dict(
            bgcolor="#111827", bordercolor="#1c2333", borderwidth=1,
        ),
        hovermode="x unified",
        height=300,
    )
    return fig