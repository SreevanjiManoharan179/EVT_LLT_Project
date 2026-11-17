"""Group 14 — EV Battery Digital-Twin Virtual Laboratory (SOC/SOH).
Launch:  streamlit run app.py      (or ./run.sh / run.bat)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Battery Twin Lab", page_icon="🔋", layout="wide",
                   initial_sidebar_state="expanded")

from g14lab import ui  # noqa: E402

P = ROOT / "pages"
pages = {
    "Start": [st.Page(str(P / "overview.py"), title="Overview", icon=":material/home:", default=True)],
    "Data": [st.Page(str(P / "data_explorer.py"), title="1  Data hub & explorer", icon=":material/database:")],
    "Digital twin": [
        st.Page(str(P / "calibrate.py"), title="2  Calibrate the twin", icon=":material/tune:"),
        st.Page(str(P / "bench.py"), title="3  Experiment bench", icon=":material/science:"),
        st.Page(str(P / "thermal.py"), title="4  Thermal & fast charging", icon=":material/device_thermostat:"),
    ],
    "Estimation": [
        st.Page(str(P / "soc.py"), title="5  SOC estimation", icon=":material/battery_charging_full:"),
        st.Page(str(P / "soh.py"), title="6  SOH & ageing", icon=":material/monitor_heart:"),
    ],
    "Vehicle": [st.Page(str(P / "vehicle.py"), title="7  Tesla telemetry replay", icon=":material/electric_car:")],
    "Results": [st.Page(str(P / "validation.py"), title="8  Validation & export", icon=":material/fact_check:")],
}
pg = st.navigation(pages, position="sidebar")
st.markdown(ui.CSS, unsafe_allow_html=True)
ui.passport()
pg.run()
