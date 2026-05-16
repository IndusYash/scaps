from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from dashboard_shared import compare_curve, inject_theme, line_traces_by_thickness, load_excel_bytes, load_excel_sheet, scatter_validation


@dataclass
class TemperatureBundle:
    df: pd.DataFrame
    jv_long: pd.DataFrame
    qe_long: pd.DataFrame
    r_long: pd.DataFrame
    model_jv: RandomForestRegressor
    model_qe: RandomForestRegressor
    model_r: RandomForestRegressor
    y_test_jv: pd.Series
    y_pred_jv: np.ndarray
    y_test_qe: pd.Series
    y_pred_qe: np.ndarray
    y_test_r: pd.Series
    y_pred_r: np.ndarray
    r2_jv: float
    mae_jv: float
    r2_qe: float
    mae_qe: float
    r2_r: float
    mae_r: float


@st.cache_data(show_spinner=False)
def load_temperature_source(excel_bytes: bytes) -> pd.DataFrame:
    return load_excel_sheet(excel_bytes, "Temperature")


@st.cache_resource(show_spinner=False)
def train_temperature_models(excel_bytes: bytes, test_size: float, n_estimators: int, random_state: int) -> TemperatureBundle:
    df = load_temperature_source(excel_bytes)

    jv_cols = ["v(V)"] + [
        col for col in df.columns
        if str(col).startswith("T=") and ".1" not in str(col) and ".2" not in str(col)
    ]
    jv_df = df[jv_cols]
    jv_long = jv_df.melt(id_vars="v(V)", var_name="Temperature", value_name="Current")
    jv_long["Temperature"] = jv_long["Temperature"].str.extract(r"T=(\d+)").astype(float)
    jv_long = jv_long.dropna(inplace=False)
    jv_long = jv_long.rename(columns={"v(V)": "Voltage"})

    qe_cols = ["lambda(nm)"] + [
        col for col in df.columns
        if str(col).startswith("T=") and ".1" in str(col)
    ]
    qe_df = df[qe_cols]
    qe_long = qe_df.melt(id_vars="lambda(nm)", var_name="Temperature", value_name="QE")
    qe_long["Temperature"] = qe_long["Temperature"].str.extract(r"T=(\d+)").astype(float)
    qe_long = qe_long.dropna(inplace=False)
    qe_long = qe_long.rename(columns={"lambda(nm)": "Wavelength"})

    r_cols = ["lambda(nm).1"] + [
        col for col in df.columns
        if str(col).startswith("T=") and ".2" in str(col)
    ]
    r_df = df[r_cols]
    r_long = r_df.melt(id_vars="lambda(nm).1", var_name="Temperature", value_name="Responsivity")
    r_long["Temperature"] = r_long["Temperature"].str.extract(r"T=(\d+)").astype(float)
    r_long = r_long.dropna(inplace=False)
    r_long = r_long.rename(columns={"lambda(nm).1": "Wavelength"})

    jv_long["Temperature"] = jv_long["Temperature"].astype(int)
    qe_long["Temperature"] = qe_long["Temperature"].astype(int)
    r_long["Temperature"] = r_long["Temperature"].astype(int)

    x_jv = jv_long[["Voltage", "Temperature"]]
    y_jv = jv_long["Current"]
    x_train_jv, x_test_jv, y_train_jv, y_test_jv = train_test_split(
        x_jv, y_jv, test_size=test_size, random_state=random_state
    )
    model_jv = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model_jv.fit(x_train_jv, y_train_jv)
    y_pred_jv = model_jv.predict(x_test_jv)

    x_qe = qe_long[["Wavelength", "Temperature"]]
    y_qe = qe_long["QE"]
    x_train_qe, x_test_qe, y_train_qe, y_test_qe = train_test_split(
        x_qe, y_qe, test_size=test_size, random_state=random_state
    )
    model_qe = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model_qe.fit(x_train_qe, y_train_qe)
    y_pred_qe = model_qe.predict(x_test_qe)

    x_r = r_long[["Wavelength", "Temperature"]]
    y_r = r_long["Responsivity"]
    x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(
        x_r, y_r, test_size=test_size, random_state=random_state
    )
    model_r = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model_r.fit(x_train_r, y_train_r)
    y_pred_r = model_r.predict(x_test_r)

    return TemperatureBundle(
        df=df,
        jv_long=jv_long,
        qe_long=qe_long,
        r_long=r_long,
        model_jv=model_jv,
        model_qe=model_qe,
        model_r=model_r,
        y_test_jv=y_test_jv,
        y_pred_jv=y_pred_jv,
        y_test_qe=y_test_qe,
        y_pred_qe=y_pred_qe,
        y_test_r=y_test_r,
        y_pred_r=y_pred_r,
        r2_jv=r2_score(y_test_jv, y_pred_jv),
        mae_jv=mean_absolute_error(y_test_jv, y_pred_jv),
        r2_qe=r2_score(y_test_qe, y_pred_qe),
        mae_qe=mean_absolute_error(y_test_qe, y_pred_qe),
        r2_r=r2_score(y_test_r, y_pred_r),
        mae_r=mean_absolute_error(y_test_r, y_pred_r),
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
            <h1>Temperature Notebook App</h1>
            <p> Welcome to the Temperature Sheet based ML Pipeline.</p>
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
        "jv_temperature": 300,
        "qe_wavelength": 500,
        "qe_temperature": 300,
        "r_wavelength": 500,
        "r_temperature": 300,
        "compare_temperature": 300,
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
            bundle = train_temperature_models(excel_bytes, float(test_size), int(n_estimators), int(random_state))
            st.session_state["temperature_bundle"] = bundle
            st.session_state["temperature_train_cfg"] = {
                "test_size": float(test_size),
                "n_estimators": int(n_estimators),
                "random_state": int(random_state),
            }
            st.success("Temperature models trained successfully.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return

    if "temperature_bundle" not in st.session_state:
        st.info("Click Train Model once. After that, all inputs and graphs update live without retraining.")
        return

    bundle: TemperatureBundle = st.session_state["temperature_bundle"]
    cfg = st.session_state.get("temperature_train_cfg", {})

    st.caption(f"Data source: {data_source}")
    st.caption(f"Last trained: test_size={cfg.get('test_size', '-')}, trees={cfg.get('n_estimators', '-')}, seed={cfg.get('random_state', '-')}")

    m1, m2, m3 = st.columns(3)
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

    st.subheader("Raw Curves From Dataset")
    st.plotly_chart(
        line_traces_by_thickness(bundle.jv_long, "Voltage", "Current", "Temperature J-V Curves", "Voltage", "Current Density", group_col="Temperature", label_suffix="K"),
        width="stretch",
    )
    st.plotly_chart(
        line_traces_by_thickness(bundle.qe_long, "Wavelength", "QE", "Temperature QE vs Wavelength", "Wavelength (nm)", "QE", group_col="Temperature", label_suffix="K"),
        width="stretch",
    )
    st.plotly_chart(
        line_traces_by_thickness(bundle.r_long, "Wavelength", "Responsivity", "Temperature Responsivity vs Wavelength", "Wavelength (nm)", "Responsivity", group_col="Temperature", label_suffix="K"),
        width="stretch",
    )

    st.subheader("Validation Graphs")
    v1, v2, v3 = st.columns(3)
    with v1:
        st.plotly_chart(
            scatter_validation(bundle.y_test_jv, bundle.y_pred_jv, "Temperature Model Validation (J-V)", "Actual Current", "Predicted Current"),
            width="stretch",
        )
    with v2:
        st.plotly_chart(
            scatter_validation(bundle.y_test_qe, bundle.y_pred_qe, "Temperature QE Validation", "Actual QE", "Predicted QE"),
            width="stretch",
        )
    with v3:
        st.plotly_chart(
            scatter_validation(bundle.y_test_r, bundle.y_pred_r, "Temperature Responsivity Validation", "Actual R", "Predicted R"),
            width="stretch",
        )

    st.subheader("Point Predictions")
    p1, p2, p3 = st.columns(3)
    with p1:
        jv_voltage = st.number_input("J-V Voltage", value=float(st.session_state["jv_voltage"]), format="%.3f")
        jv_temperature = st.number_input("J-V Temperature (K)", min_value=1, value=int(st.session_state["jv_temperature"]), step=50)
    with p2:
        qe_wavelength = st.number_input("QE Wavelength", min_value=1.0, value=float(st.session_state["qe_wavelength"]), step=10.0)
        qe_temperature = st.number_input("QE Temperature (K)", min_value=1, value=int(st.session_state["qe_temperature"]), step=50)
    with p3:
        r_wavelength = st.number_input("R Wavelength", min_value=1.0, value=float(st.session_state["r_wavelength"]), step=10.0)
        r_temperature = st.number_input("R Temperature (K)", min_value=1, value=int(st.session_state["r_temperature"]), step=50)

    pred_jv = float(bundle.model_jv.predict(pd.DataFrame({"Voltage": [jv_voltage], "Temperature": [int(jv_temperature)]}))[0])
    pred_qe = float(bundle.model_qe.predict(pd.DataFrame({"Wavelength": [qe_wavelength], "Temperature": [int(qe_temperature)]}))[0])
    pred_r = float(bundle.model_r.predict(pd.DataFrame({"Wavelength": [r_wavelength], "Temperature": [int(r_temperature)]}))[0])

    r1, r2c, r3 = st.columns(3)
    r1.metric("Predicted Current", f"{pred_jv:.6f}")
    r2c.metric("Predicted QE", f"{pred_qe:.6f}")
    r3.metric("Predicted Responsivity", f"{pred_r:.6f}")

    st.subheader("Comparison Curves (Actual vs Predicted)")
    compare_temperature = st.number_input(
        "Comparison Temperature (K)",
        min_value=1,
        value=int(st.session_state["compare_temperature"]),
        step=50,
    )
    compare_temperature = int(compare_temperature)

    jv_actual = bundle.jv_long[bundle.jv_long["Temperature"] == compare_temperature].sort_values("Voltage")
    if not jv_actual.empty:
        jv_x = jv_actual["Voltage"].to_numpy()
        jv_pred = bundle.model_jv.predict(pd.DataFrame({"Voltage": jv_x, "Temperature": [compare_temperature] * len(jv_x)}))
        st.plotly_chart(
            compare_curve(jv_actual["Voltage"], jv_actual["Current"], jv_x, jv_pred, f"J-V Comparison ({compare_temperature} K)", "Voltage", "Current"),
            width="stretch",
        )
    else:
        st.info("No actual J-V data for this temperature.")

    qe_actual = bundle.qe_long[bundle.qe_long["Temperature"] == compare_temperature].sort_values("Wavelength")
    if not qe_actual.empty:
        qe_x = qe_actual["Wavelength"].to_numpy()
        qe_pred = bundle.model_qe.predict(pd.DataFrame({"Wavelength": qe_x, "Temperature": [compare_temperature] * len(qe_x)}))
        st.plotly_chart(
            compare_curve(qe_actual["Wavelength"], qe_actual["QE"], qe_x, qe_pred, f"QE Comparison ({compare_temperature} K)", "Wavelength (nm)", "QE"),
            width="stretch",
        )
    else:
        st.info("No actual QE data for this temperature.")

    r_actual = bundle.r_long[bundle.r_long["Temperature"] == compare_temperature].sort_values("Wavelength")
    if not r_actual.empty:
        r_x = r_actual["Wavelength"].to_numpy()
        r_pred = bundle.model_r.predict(pd.DataFrame({"Wavelength": r_x, "Temperature": [compare_temperature] * len(r_x)}))
        st.plotly_chart(
            compare_curve(r_actual["Wavelength"], r_actual["Responsivity"], r_x, r_pred, f"Responsivity Comparison ({compare_temperature} K)", "Wavelength (nm)", "Responsivity"),
            width="stretch",
        )
    else:
        st.info("No actual Responsivity data for this temperature.")

    st.subheader("Detailed Physical Interpretation - Temperature")
    with st.expander("Show Temperature Theory", expanded=False):
        st.markdown(
            """
1. Temperature is an external operating parameter affecting device behavior, unlike structural parameters like thickness.
2. Higher temperature increases carrier energy and can influence bandgap, carrier mobility, and recombination.
3. Temperature effects dominate in realistic operating conditions (environment, weather, thermal loading).
4. J-V, QE, and Responsivity all shift with temperature because thermal effects modulate carrier transport and generation.
5. The temperature dashboard isolates thermal analysis from structural (PVK/ETL) workflows, enabling independent optimization.

Conclusion: Temperature is a critical operational variable determining real-world device performance under varying environmental conditions.
            """
        )
