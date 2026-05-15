from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Space+Grotesk:wght@400;500;700&display=swap');
            :root {
                --bg-1: #04101a;
                --bg-2: #0c1f2e;
                --accent-1: #39d0ff;
                --accent-2: #8cf79a;
                --text-1: #eaf6ff;
                --stroke: rgba(57, 208, 255, 0.28);
                --card: rgba(12, 31, 46, 0.7);
            }
            .stApp {
                background:
                    radial-gradient(circle at 12% 8%, rgba(57, 208, 255, 0.18), transparent 30%),
                    radial-gradient(circle at 92% 20%, rgba(140, 247, 154, 0.15), transparent 32%),
                    linear-gradient(130deg, var(--bg-1), var(--bg-2));
                color: var(--text-1);
                font-family: 'Space Grotesk', sans-serif;
            }
            h1, h2, h3 {
                font-family: 'Orbitron', sans-serif;
                letter-spacing: 0.03em;
            }
            .hero {
                border: 1px solid var(--stroke);
                border-radius: 16px;
                background: linear-gradient(145deg, rgba(57, 208, 255, 0.12), rgba(140, 247, 154, 0.08));
                padding: 1rem 1.2rem;
                margin-bottom: 0.8rem;
            }
            .credit-strip {
                width: 100%;
                display: flex;
                justify-content: flex-start;
                margin: 0.2rem 0 0.6rem 0;
            }
            .credit-pill {
                font-size: 0.82rem;
                color: var(--text-1);
                background: rgba(5, 7, 13, 0.62);
                border: 1px solid var(--stroke);
                border-radius: 999px;
                padding: 0.35rem 0.75rem;
                backdrop-filter: blur(5px);
            }
            .metric-card {
                border: 1px solid var(--stroke);
                border-radius: 14px;
                padding: 0.85rem 1rem;
                background: var(--card);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_excel_bytes(uploaded_file, default_path: str = "Data_file.xlsx") -> tuple[bytes, str]:
    if uploaded_file is not None:
        return uploaded_file.getvalue(), "Uploaded file"

    with open(default_path, "rb") as handle:
        return handle.read(), f"{default_path} (workspace)"


def load_excel_sheet(excel_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(excel_bytes), sheet_name=sheet_name)


def line_traces_by_thickness(df_long: pd.DataFrame, x_col: str, y_col: str, title: str, x_title: str, y_title: str, group_col: str = "Thickness", label_suffix: str = "nm") -> go.Figure:
    fig = go.Figure()
    for group_val in sorted(df_long[group_col].unique()):
        subset = df_long[df_long[group_col] == group_val].sort_values(x_col)
        fig.add_trace(
            go.Scatter(
                x=subset[x_col],
                y=subset[y_col],
                mode="lines",
                name=f"{group_val} {label_suffix}",
            )
        )
    fig.update_layout(template="plotly_dark", title=title, xaxis_title=x_title, yaxis_title=y_title, height=440)
    return fig


def scatter_validation(y_true: pd.Series, y_pred, title: str, x_label: str, y_label: str) -> go.Figure:
    min_v = float(min(y_true.min(), y_pred.min()))
    max_v = float(max(y_true.max(), y_pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="Points", marker={"size": 7}))
    fig.add_trace(
        go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Ideal", line={"dash": "dash"})
    )
    fig.update_layout(template="plotly_dark", title=title, xaxis_title=x_label, yaxis_title=y_label, height=420)
    return fig


def compare_curve(actual_x: pd.Series, actual_y: pd.Series, pred_x, pred_y, title: str, x_label: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_x, y=actual_y, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=pred_x, y=pred_y, mode="lines", name="Predicted", line={"dash": "dash"}))
    fig.update_layout(template="plotly_dark", title=title, xaxis_title=x_label, yaxis_title=y_label, height=430)
    return fig