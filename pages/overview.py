import plotly.graph_objects as go
import streamlit as st

from g14lab import ui

ui.page_header("EV Battery Digital-Twin Virtual Lab",
               "A virtual laboratory for lithium-ion state of charge and state of health. A cell twin is calibrated "
               "on NASA ageing measurements, experiments run on it under changing temperature, C-rate and ageing, "
               "and every result is checked against measured cells and two real Tesla Model Y packs.")

dc = ui.discharge_cycles()
cells = ui.cells()
ss = ui.bev_sessions() if ui.bev_available() else None
p, th, t = ui.twin()

left, right = st.columns([1.35, 1], gap="large")
with left:
    fig = go.Figure()
    for i, (cid, g) in enumerate(dc[dc.cell_id.isin(["B0005", "B0006", "B0007", "B0018"])].groupby("cell_id")):
        fig.add_trace(go.Scatter(x=g.discharge_index, y=g.soh_pct, mode="lines", name=cid,
                                 line=dict(width=2, color=ui.SERIES[i])))
    fig.add_hline(y=70, line=dict(color=ui.ALERT, dash="dash", width=1.2),
                  annotation_text="End of life (70 %)", annotation_position="bottom left",
                  annotation_font=dict(color=ui.ALERT, size=12))
    fig.update_layout(xaxis_title="Discharge cycle", yaxis_title="SOH (% of 2.0 Ah rated)", hovermode="x unified")
    ui.chart(fig, "Capacity fade of the four NASA reference cells (24 °C, 2 A)", ("Measured",), height=372)
with right:
    st.markdown('<div class="g14-ch"><span class="g14-ch-t">What the lab is built on</span></div>', unsafe_allow_html=True)
    rows = [
        ("Lab cells (NASA PCoE)", f"{len(cells)} cells", "18650, 2 Ah rated; 4 / 24 / 43 °C; 1, 2, 4 A and pulsed loads"),
        ("Discharge cycles", f"{len(dc):,}", "V, I, T at 9–19 s; capacity per cycle"),
        ("Impedance tests", f"{len(ui.impedance()):,}", "EIS 0.1 Hz – 5 kHz with fitted Re, Rct"),
        ("Charge curves", f"{len(ui.charge_cycles()):,}", "CC-CV at 1.5 A; used in twin calibration and ICA"),
    ]
    if ss is not None:
        rows += [("Vehicle sessions (KU Leuven)", f"{len(ss):,}",
                  f"{(ss.kind == 'drive').sum():,} drives, {(ss.kind == 'fast_charge').sum()} DC and "
                  f"{(ss.kind == 'slow_charge').sum()} AC charges"),
                 ("Vehicle telemetry", "3.16 M rows", "1 s pack V, I, T, BMS SOC, ambient, speed")]
    ui.sheet(rows, header=("Source", "Size", "Content"))
    st.caption(f"Active twin: {t['label']}. Fit {p.fit_rmse_mV:.1f} mV RMSE on measured voltage; "
               f"Ea {p.Ea_R0/1000:.1f} kJ/mol from {p.extra.get('arrhenius_n', '–')} cycles.")

ui.section("How the lab fits together", "Calibrated parameters and model outputs flow from left to right; "
           "every page reads the same active twin.")
dot = """
digraph G {
  rankdir=LR; bgcolor="transparent"; pad=0.1; nodesep=0.18; ranksep=0.3;
  node [shape=box, style="rounded,filled", fontname="Barlow", fontsize=13, color="#C9D1D9", fillcolor="#FFFFFF",
        fontcolor="#18212B", margin="0.1,0.05", height=0.4];
  edge [color="#7A8794", arrowsize=0.6];
  nasa [label="NASA PCoE\nageing data", fillcolor="#E9EDF2"];
  bev  [label="KU Leuven\nTesla telemetry", fillcolor="#E9EDF2"];
  cal  [label="Calibrate\nOCV, R0, R1, τ1"];
  shadow [label="Digital shadow\nR tracking"];
  twin [label="Digital twin\nECM + thermal", color="#2456C9", penwidth=2.2];
  bench[label="Experiment\nbench"];
  th   [label="Thermal &\nfast charging"];
  soc  [label="SOC\nestimation"];
  soh  [label="SOH &\nageing"];
  veh  [label="Vehicle\nreplay"];
  val  [label="Validation\n& export", color="#C46A12", penwidth=2.2];
  {rank=same; nasa; bev}
  {rank=same; cal; shadow; soh}
  {rank=same; bench; th; soc; veh}
  nasa -> cal -> twin; nasa -> shadow -> twin; twin -> bench; twin -> th; twin -> soc;
  nasa -> soh; soh -> soc; bev -> veh; bev -> th;
  bench -> val; soc -> val; soh -> val; th -> val; veh -> val;
}"""
st.graphviz_chart(dot, width="stretch")

ui.section("Demonstration path")
ui.sheet([
    ("1", "Data hub & explorer", "What was measured, at which conditions, and how clean is it?"),
    ("2", "Calibrate the twin", "Can a 1-RC model reproduce measured voltage and temperature, and how do its parameters age?"),
    ("3", "Experiment bench", "What happens to voltage, SOC and temperature when C-rate, ambient or ageing change?"),
    ("4", "Thermal & fast charging", "Where does the cell run hot, and how do real Tesla packs behave on DC chargers?"),
    ("5", "SOC estimation", "Which estimator recovers from a wrong start and a biased current sensor?"),
    ("6", "SOH & ageing", "Which health indicators and models estimate capacity on a cell never seen in training?"),
    ("7", "Tesla telemetry replay", "Does real-world data agree, and how does temperature change real energy use?"),
    ("8", "Validation & export", "How large are the errors, where are the limits, and what was run?"),
], header=("Step", "Page", "Question it answers"))
ui.note("Colour always means provenance: <b>Measured</b> data from the datasets, <b>Simulated</b> twin output, "
        "<b>Estimated</b> SOC from an estimator, <b>Predicted</b> ML or forecast output, and <b>Reference</b> for a "
        "derived quantity such as Ah-integrated or BMS-reported SOC. The twin is calibrated offline and replays recorded "
        "telemetry; it is not live-synchronised with a physical battery.")
ui.footer()
