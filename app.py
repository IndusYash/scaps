from __future__ import annotations

import streamlit as st

from etl_dashboard import render as render_etl_dashboard
from donor_dashboard import render as render_donor_dashboard
from defect_dashboard import render as render_defect_dashboard
from interface_dashboard import render as render_interface_dashboard
from htl_dashboard import render as render_htl_dashboard
from pvk_dashboard import render as render_pvk_dashboard
from temperature_dashboard import render as render_temperature_dashboard


def main() -> None:
    st.set_page_config(page_title="Solar Project Dashboards", layout="wide")

    st.sidebar.title("Solar Project")
    st.sidebar.caption("Choose which notebook-backed dashboard to open.")
    selected_dashboard = st.sidebar.radio("Dashboard", ["ETL", "PVK", "HTL", "Temperature", "Interface Density", "Defect Density", "Donor Con"], index=0)

    if selected_dashboard == "ETL":
        render_etl_dashboard()
    elif selected_dashboard == "PVK":
        render_pvk_dashboard()
    elif selected_dashboard == "HTL":
        render_htl_dashboard()
    elif selected_dashboard == "Temperature":
        render_temperature_dashboard()
    elif selected_dashboard == "Interface Density":
        render_interface_dashboard()
    elif selected_dashboard == "Defect Density":
        render_defect_dashboard()
    else:
        render_donor_dashboard()


if __name__ == "__main__":
    main()
