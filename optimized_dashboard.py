from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from dashboard_shared import compare_curve, inject_theme, load_excel_bytes, load_excel_sheet, scatter_validation


@dataclass
class OptimizedBundle:
    df: pd.DataFrame
    jv_df: pd.DataFrame
    qe_df: pd.DataFrame
    r_df: pd.DataFrame
    d_df: pd.DataFrame
    model_jv: RandomForestRegressor
    model_qe: RandomForestRegressor
    model_r: RandomForestRegressor
    model_d: RandomForestRegressor
    y_test_jv: pd.Series
    y_pred_jv: np.ndarray
    y_test_qe: pd.Series
    y_pred_qe: np.ndarray
    y_test_r: pd.Series
    y_pred_r: np.ndarray
    y_test_d: pd.Series
    y_pred_d: np.ndarray
    r2_jv: float
    mae_jv: float
    r2_qe: float
    mae_qe: float
    r2_r: float
    mae_r: float
    r2_d: float
    mae_d: float


@st.cache_data(show_spinner=False)
def load_optimized_source(excel_bytes: bytes) -> pd.DataFrame:
    return load_excel_sheet(excel_bytes, "Optimized")


@st.cache_resource(show_spinner=False)
def train_optimized_models(excel_bytes: bytes, test_size: float, n_estimators: int, random_state: int) -> OptimizedBundle:
    df = load_optimized_source(excel_bytes)

    # J-V Data
    jv_df = df[[' v(V)', 'jtot(mA/cm2)']].copy()
    jv_df.rename(columns={' v(V)': 'Voltage', 'jtot(mA/cm2)': 'Current'}, inplace=True)
    jv_df = jv_df.dropna()
    jv_df['Voltage'] = pd.to_numeric(jv_df['Voltage'])
    jv_df['Current'] = pd.to_numeric(jv_df['Current'])
    jv_df = jv_df.reset_index(drop=True)

    x_jv = jv_df[['Voltage']]
    y_jv = jv_df['Current']
    x_train_jv, x_test_jv, y_train_jv, y_test_jv = train_test_split(
        x_jv, y_jv, test_size=test_size, random_state=random_state
    )
    model_jv = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model_jv.fit(x_train_jv, y_train_jv)
    y_pred_jv = model_jv.predict(x_test_jv)

    # QE Data
    qe_df = df[['lambda(nm)', '      QE(%)']].copy()
    qe_df.rename(columns={'lambda(nm)': 'Wavelength', '      QE(%)': 'QE'}, inplace=True)
    qe_df = qe_df.dropna()
    qe_df['Wavelength'] = pd.to_numeric(qe_df['Wavelength'])
    qe_df['QE'] = pd.to_numeric(qe_df['QE'])
    qe_df = qe_df.reset_index(drop=True)

    x_qe = qe_df[['Wavelength']]
    y_qe = qe_df['QE']
    x_train_qe, x_test_qe, y_train_qe, y_test_qe = train_test_split(
        x_qe, y_qe, test_size=test_size, random_state=random_state
    )
    model_qe = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model_qe.fit(x_train_qe, y_train_qe)
    y_pred_qe = model_qe.predict(x_test_qe)

    # Responsivity Data
    r_df = df[['lambda(nm).1', 'R[A/W]']].copy()
    r_df.rename(columns={'lambda(nm).1': 'Wavelength', 'R[A/W]': 'Responsivity'}, inplace=True)
    r_df = r_df.dropna()
    r_df['Wavelength'] = pd.to_numeric(r_df['Wavelength'])
    r_df['Responsivity'] = pd.to_numeric(r_df['Responsivity'])
    r_df = r_df.reset_index(drop=True)

    x_r = r_df[['Wavelength']]
    y_r = r_df['Responsivity']
    x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(
        x_r, y_r, test_size=test_size, random_state=random_state
    )
    model_r = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model_r.fit(x_train_r, y_train_r)
    y_pred_r = model_r.predict(x_test_r)

    # Detectivity Data
    d_df = df[['V(V)', 'D*']].copy()
    d_df.rename(columns={'V(V)': 'Voltage', 'D*': 'Detectivity'}, inplace=True)
    d_df = d_df.dropna()
    d_df['Voltage'] = pd.to_numeric(d_df['Voltage'])
    d_df['Detectivity'] = pd.to_numeric(d_df['Detectivity'])
    d_df = d_df.reset_index(drop=True)

    x_d = d_df[['Voltage']]
    y_d = d_df['Detectivity']
    x_train_d, x_test_d, y_train_d, y_test_d = train_test_split(
        x_d, y_d, test_size=test_size, random_state=random_state
    )
    model_d = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model_d.fit(x_train_d, y_train_d)
    y_pred_d = model_d.predict(x_test_d)

    return OptimizedBundle(
        df=df,
        jv_df=jv_df,
        qe_df=qe_df,
        r_df=r_df,
        d_df=d_df,
        model_jv=model_jv,
        model_qe=model_qe,
        model_r=model_r,
        model_d=model_d,
        y_test_jv=y_test_jv,
        y_pred_jv=y_pred_jv,
        y_test_qe=y_test_qe,
        y_pred_qe=y_pred_qe,
        y_test_r=y_test_r,
        y_pred_r=y_pred_r,
        y_test_d=y_test_d,
        y_pred_d=y_pred_d,
        r2_jv=r2_score(y_test_jv, y_pred_jv),
        mae_jv=mean_absolute_error(y_test_jv, y_pred_jv),
        r2_qe=r2_score(y_test_qe, y_pred_qe),
        mae_qe=mean_absolute_error(y_test_qe, y_pred_qe),
        r2_r=r2_score(y_test_r, y_pred_r),
        mae_r=mean_absolute_error(y_test_r, y_pred_r),
        r2_d=r2_score(y_test_d, y_pred_d),
        mae_d=mean_absolute_error(y_test_d, y_pred_d),
    )


def render() -> None:
    inject_theme()

    st.markdown(
        '<div class="credit-strip"><div class="credit-pill">Under the guidance of Dr. Satvik Vats, Assistant Professor, Department of CSE, in collaboration with Physics Lab</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>Optimized Device Notebook App</h1>
            <p>Welcome to the Optimized Sheet based ML Pipeline.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        excel_bytes, data_source = load_excel_bytes(None)
    except FileNotFoundError:
        st.error("Data_file.xlsx not found. Upload an xlsx file.")
        return
    except Exception as exc:
        st.error(f"Data load failed: {exc}")
        return

    defaults = {
        "test_size": 0.2,
        "n_estimators": 200,
        "random_state": 42,
        "jv_voltage": 0.5,
        "qe_wavelength": 500,
        "r_wavelength": 500,
        "d_voltage": 0.5,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.markdown("### Controls")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    with c1:
        test_size = st.number_input("Test Size", min_value=0.05, max_value=0.5, value=float(st.session_state["test_size"]), step=0.05)
    with c2:
        n_estimators = st.slider("RF Trees", min_value=50, max_value=600, value=int(st.session_state["n_estimators"]), step=50)
    with c3:
        random_state = st.number_input("Random Seed", min_value=0, value=int(st.session_state["random_state"]), step=1)
    with c4:
        train_clicked = st.button("Train Model", type="primary", use_container_width=True)

    if train_clicked:
        try:
            bundle = train_optimized_models(excel_bytes, float(test_size), int(n_estimators), int(random_state))
            st.session_state["optimized_bundle"] = bundle
            st.session_state["optimized_train_cfg"] = {
                "test_size": float(test_size),
                "n_estimators": int(n_estimators),
                "random_state": int(random_state),
            }
            st.success("Optimized models trained successfully.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return

    if "optimized_bundle" not in st.session_state:
        st.info("Click Train Model once. After that, all inputs and graphs update live without retraining.")
        return

    bundle: OptimizedBundle = st.session_state["optimized_bundle"]
    cfg = st.session_state.get("optimized_train_cfg", {})

    st.caption(f"Data source: {data_source}")
    st.caption(f"Last trained: test_size={cfg.get('test_size', '-')}, trees={cfg.get('n_estimators', '-')}, seed={cfg.get('random_state', '-')}")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("J-V R2", f"{bundle.r2_jv:.6f}")
        st.metric("J-V MAE", f"{bundle.mae_jv:.6f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("QE R2", f"{bundle.r2_qe:.6f}")
        st.metric("QE MAE", f"{bundle.mae_qe:.6f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Responsivity R2", f"{bundle.r2_r:.6f}")
        st.metric("Responsivity MAE", f"{bundle.mae_r:.6f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Detectivity R2", f"{bundle.r2_d:.6f}")
        st.metric("Detectivity MAE", f"{bundle.mae_d:.6f}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Raw Curves From Dataset")
    
    import plotly.graph_objects as go
    
    # J-V Curve
    fig_jv = go.Figure()
    fig_jv.add_trace(go.Scatter(x=bundle.jv_df["Voltage"], y=bundle.jv_df["Current"], mode='lines+markers', name='J-V'))
    fig_jv.update_layout(title="Optimized J-V Curve", xaxis_title="Voltage (V)", yaxis_title="Current Density (mA/cm²)")
    st.plotly_chart(fig_jv, use_container_width=True)
    
    # QE Curve
    fig_qe = go.Figure()
    fig_qe.add_trace(go.Scatter(x=bundle.qe_df["Wavelength"], y=bundle.qe_df["QE"], mode='lines+markers', name='QE'))
    fig_qe.update_layout(title="Optimized QE Curve", xaxis_title="Wavelength (nm)", yaxis_title="QE (%)")
    st.plotly_chart(fig_qe, use_container_width=True)
    
    # Responsivity Curve
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=bundle.r_df["Wavelength"], y=bundle.r_df["Responsivity"], mode='lines+markers', name='Responsivity'))
    fig_r.update_layout(title="Optimized Responsivity Curve", xaxis_title="Wavelength (nm)", yaxis_title="Responsivity (A/W)")
    st.plotly_chart(fig_r, use_container_width=True)
    
    # Detectivity Curve
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=bundle.d_df["Voltage"], y=bundle.d_df["Detectivity"], mode='lines+markers', name='Detectivity'))
    fig_d.update_layout(title="Optimized Detectivity Curve", xaxis_title="Voltage (V)", yaxis_title="Detectivity")
    st.plotly_chart(fig_d, use_container_width=True)

    st.subheader("Validation Graphs")
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.plotly_chart(
            scatter_validation(bundle.y_test_jv, bundle.y_pred_jv, "J-V Validation", "Actual Current", "Predicted Current"),
            use_container_width=True,
        )
    with v2:
        st.plotly_chart(
            scatter_validation(bundle.y_test_qe, bundle.y_pred_qe, "QE Validation", "Actual QE", "Predicted QE"),
            use_container_width=True,
        )
    with v3:
        st.plotly_chart(
            scatter_validation(bundle.y_test_r, bundle.y_pred_r, "Responsivity Validation", "Actual R", "Predicted R"),
            use_container_width=True,
        )
    with v4:
        st.plotly_chart(
            scatter_validation(bundle.y_test_d, bundle.y_pred_d, "Detectivity Validation", "Actual D*", "Predicted D*"),
            use_container_width=True,
        )

    st.subheader("Point Predictions")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        jv_voltage = st.number_input("J-V Voltage (V)", value=float(st.session_state["jv_voltage"]), format="%.3f")
    with p2:
        qe_wavelength = st.number_input("QE Wavelength (nm)", min_value=1.0, value=float(st.session_state["qe_wavelength"]), step=10.0)
    with p3:
        r_wavelength = st.number_input("R Wavelength (nm)", min_value=1.0, value=float(st.session_state["r_wavelength"]), step=10.0)
    with p4:
        d_voltage = st.number_input("D* Voltage (V)", value=float(st.session_state["d_voltage"]), format="%.3f")

    pred_jv = float(bundle.model_jv.predict(pd.DataFrame({"Voltage": [jv_voltage]}))[0])
    pred_qe = float(bundle.model_qe.predict(pd.DataFrame({"Wavelength": [qe_wavelength]}))[0])
    pred_r = float(bundle.model_r.predict(pd.DataFrame({"Wavelength": [r_wavelength]}))[0])
    pred_d = float(bundle.model_d.predict(pd.DataFrame({"Voltage": [d_voltage]}))[0])

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Predicted Current", f"{pred_jv:.6f}")
    r2.metric("Predicted QE", f"{pred_qe:.6f}")
    r3.metric("Predicted Responsivity", f"{pred_r:.6f}")
    r4.metric("Predicted Detectivity", f"{pred_d:.6f}")

    st.subheader("Comparison Curves (Actual vs Predicted Full Range)")
    
    # J-V Comparison
    jv_x = bundle.jv_df["Voltage"].to_numpy()
    jv_pred = bundle.model_jv.predict(pd.DataFrame({"Voltage": jv_x}))
    st.plotly_chart(
        compare_curve(bundle.jv_df["Voltage"], bundle.jv_df["Current"], jv_x, jv_pred, "J-V: Actual vs Predicted", "Voltage (V)", "Current"),
        use_container_width=True,
    )
    
    # QE Comparison
    qe_x = bundle.qe_df["Wavelength"].to_numpy()
    qe_pred = bundle.model_qe.predict(pd.DataFrame({"Wavelength": qe_x}))
    st.plotly_chart(
        compare_curve(bundle.qe_df["Wavelength"], bundle.qe_df["QE"], qe_x, qe_pred, "QE: Actual vs Predicted", "Wavelength (nm)", "QE"),
        use_container_width=True,
    )
    
    # Responsivity Comparison
    r_x = bundle.r_df["Wavelength"].to_numpy()
    r_pred = bundle.model_r.predict(pd.DataFrame({"Wavelength": r_x}))
    st.plotly_chart(
        compare_curve(bundle.r_df["Wavelength"], bundle.r_df["Responsivity"], r_x, r_pred, "Responsivity: Actual vs Predicted", "Wavelength (nm)", "Responsivity"),
        use_container_width=True,
    )
    
    # Detectivity Comparison
    d_x = bundle.d_df["Voltage"].to_numpy()
    d_pred = bundle.model_d.predict(pd.DataFrame({"Voltage": d_x}))
    st.plotly_chart(
        compare_curve(bundle.d_df["Voltage"], bundle.d_df["Detectivity"], d_x, d_pred, "Detectivity: Actual vs Predicted", "Voltage (V)", "Detectivity"),
        use_container_width=True,
    )
