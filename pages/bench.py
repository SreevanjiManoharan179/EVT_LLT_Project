import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from g14lab import simulator as sim
from g14lab import ui
from g14lab.sensitivity import OUTPUTS, PARAMS, tornado

ui.page_header("Experiment bench",
               "Run controlled experiments on the calibrated twin. Set the initial SOC, C-rate, ambient temperature and "
               "ageing state, sweep one variable, or save runs as scenarios and compare them.")
p, th, t = ui.twin()
soh_twin = ui.twin_soh_pct(p)
EXPS = ["CC discharge", "CC-CV charge", "Pulse discharge (square wave)", "Charge, rest, discharge"]

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    exp = c1.selectbox("Experiment", EXPS, key="b_exp")
    crate = c2.slider("C-rate", 0.2, 3.0, 1.0, 0.1, key="b_c", help="1C = the twin's capacity per hour. Pulse experiments "
                      "use twice this current at 50 % duty so the average equals the selected C-rate.")
    discharge_first = exp in ("CC discharge", "Pulse discharge (square wave)")
    soc0 = c3.slider("Initial SOC (%)", 0, 100, 100 if discharge_first else 10, key="b_soc")
    amb = c4.slider("Ambient temperature (°C)", -10, 50, 25, key="b_amb")
    c5, c6, c7, c8 = st.columns(4, gap="medium")
    src = c5.selectbox("Ageing state from", ["SOH slider (fitted growth law)", "A measured NASA cycle (digital shadow)"],
                       key="b_age_src")
    if src.startswith("SOH"):
        soh_pct = c6.slider("SOH (% of 2.0 Ah rated)", 60, 100, int(round(soh_twin)), key="b_soh",
                            help="Capacity scales with SOH; R0 and R1 follow the resistance-growth law fitted on the "
                                 "reference cells' tracked parameters.")
        rgrow = ui.resistance_factor(soh_pct, p)
        age_note = f"growth law: R × {rgrow:.2f}"
    else:
        tr = ui.metrics_parquet("param_tracking.parquet")
        a_cell = c6.selectbox("Cell and cycle", sorted(tr.cell_id.unique()), key="b_age_cell")
        tc = tr[tr.cell_id == a_cell].sort_values("discharge_index")
        k_sel = c6.select_slider("Discharge", tc.discharge_index.tolist(), value=tc.discharge_index.iloc[len(tc) // 2],
                                 key="b_age_k", label_visibility="collapsed")
        row = tc[tc.discharge_index == k_sel].iloc[0]
        r_med = tc.R_total_ohm.rolling(9, center=True, min_periods=3).median()[tc.discharge_index == k_sel].iloc[0]
        soh_pct = float(row.soh_pct)
        rgrow = float(r_med / (p.R0 + p.R1))
        age_note = f"{a_cell} #{k_sel}: Q {row.capacity_Ah:.3f} Ah, tracked R0+R1 {1000*r_med:.0f} mΩ"
    vmin = c7.number_input("Discharge cut-off (V)", 2.0, 3.2, 2.7, 0.05, key="b_vmin")
    vmax = c8.number_input("Charge limit (V)", 4.0, 4.25, 4.2, 0.01, key="b_vmax")
    modes = ["Single run", "Sweep C-rate", "Sweep ambient", "Sweep initial SOC", "Sweep ageing (SOH)"]
    sweep = st.radio("Mode", modes, horizontal=True, key="b_mode")
    st.caption(f"Ageing state: SOH {soh_pct:.1f} % ({age_note}). Twin: {t['label']}.")

ui.domain_warnings(ambient=amb, crate=crate, soh=soh_pct, p=p)


def steps_for(I):
    if exp == "CC discharge":
        return [{"type": "cc", "current_A": I, "until_V": vmin, "max_s": 6 * 3600}]
    if exp == "CC-CV charge":
        return [{"type": "cc", "current_A": -I, "until_V": vmax, "max_s": 6 * 3600},
                {"type": "cv", "voltage_V": vmax, "until_A": 0.02, "max_s": 3 * 3600}]
    if exp.startswith("Pulse"):
        return [{"type": "pulse", "high_A": 2 * I, "low_A": 0.0, "period_s": 20, "duty": 0.5, "until_V": vmin,
                 "max_s": 6 * 3600}]
    return [{"type": "cc", "current_A": -I, "until_V": vmax, "max_s": 6 * 3600},
            {"type": "cv", "voltage_V": vmax, "until_A": 0.04, "max_s": 2 * 3600},
            {"type": "rest", "duration_s": 900},
            {"type": "cc", "current_A": I, "until_V": vmin, "max_s": 6 * 3600}]


def state_for(s0, ta, sh, rg):
    return sim.TwinState(soc0=s0 / 100, T0_C=ta, T_amb_C=ta, soh=(sh / 100 * 2.0) / p.Q_Ah, r_scale=rg,
                         v_min=vmin, v_max=vmax)


if sweep == "Single run":
    runs = [(f"{crate:.1f}C, {amb} °C, SOC₀ {soc0} %, SOH {soh_pct:.0f} %", crate, soc0, amb, soh_pct, rgrow)]
elif sweep == "Sweep C-rate":
    vals = st.multiselect("C-rates", [0.2, 0.5, 1.0, 1.5, 2.0, 3.0], default=[0.5, 1.0, 2.0, 3.0], key="b_sv1")
    runs = [(f"{v:.1f}C", v, soc0, amb, soh_pct, rgrow) for v in vals]
elif sweep == "Sweep ambient":
    vals = st.multiselect("Ambient (°C)", [-10, 0, 4, 10, 25, 35, 43, 50], default=[4, 10, 25, 43], key="b_sv2")
    runs = [(f"{v} °C", crate, soc0, v, soh_pct, rgrow) for v in vals]
elif sweep == "Sweep initial SOC":
    vals = st.multiselect("Initial SOC (%)", [0, 10, 20, 40, 60, 80, 100],
                          default=[20, 60, 100] if discharge_first else [0, 20, 50, 80], key="b_sv3")
    runs = [(f"SOC₀ {v} %", crate, v, amb, soh_pct, rgrow) for v in vals]
else:
    vals = st.multiselect("SOH (% of 2.0 Ah)", [95, 90, 85, 80, 75, 70, 65], default=[90, 80, 70], key="b_sv4")
    runs = [(f"SOH {v} %", crate, soc0, amb, v, ui.resistance_factor(v, p)) for v in vals]

if not runs:
    st.info("Choose at least one value to sweep.")
    st.stop()

results = []
for label, cr, s0, ta, sh, rg in runs:
    st_ = state_for(s0, ta, sh, rg)
    I = cr * p.Q_Ah * st_.soh
    df, summ = sim.run_protocol(p, th, st_, steps_for(I), dt=1.0)
    band, why = sim.thermal_stress_index(summ["T_max_C"], summ["dT_max_C"])
    summ.update(label=label, stress=band, current_A=I)
    results.append((label, df, summ))

fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.12, horizontal_spacing=0.08,
                    subplot_titles=("Terminal voltage (V)", "State of charge (%)", "Current (A, positive = discharge)",
                                    "Cell temperature (°C)"))
for i, (label, df, summ) in enumerate(results):
    col = ui.SERIES[i % len(ui.SERIES)]
    x = df.t_s / 60
    fig.add_trace(go.Scatter(x=x, y=df.V, name=label, legendgroup=label, line=dict(color=col, width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=100 * df.SOC, showlegend=False, legendgroup=label, line=dict(color=col, width=2)), 1, 2)
    fig.add_trace(go.Scatter(x=x, y=df.I_A, showlegend=False, legendgroup=label, line=dict(color=col, width=1.6)), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=df.T_C, showlegend=False, legendgroup=label, line=dict(color=col, width=2)), 2, 2)
fig.update_xaxes(title_text="Time (min)", row=2, col=1)
fig.update_xaxes(title_text="Time (min)", row=2, col=2)
ui.chart(fig, f"{exp}: simulated response", ("Simulated",), height=640, key="b_main")

if len(results) == 1:
    s = results[0][2]
    ui.readouts([("Duration", f"{s['duration_min']:.1f}", "min"), ("Charge moved", f"{abs(s['Ah_discharged']):.3f}", "Ah"),
                 ("Net energy", f"{s['Wh_net']:.2f}", "Wh"), ("Final SOC", f"{s['soc_final_pct']:.1f}", "%"),
                 ("Peak temperature", f"{s['T_max_C']:.1f}", "°C", f"rise {s['dT_max_C']:.1f} K"),
                 ("Thermal stress", s["stress"], "", "illustrative band")], cols=6, accent=ui.PROV["Simulated"])
summ_df = pd.DataFrame([dict(Scenario=s["label"], current=s["current_A"], dur=s["duration_min"], ah=abs(s["Ah_discharged"]),
                             wh=s["Wh_net"], soc=s["soc_final_pct"], vmin=s["V_min"], vmax=s["V_max"], tmax=s["T_max_C"],
                             dT=s["dT_max_C"], heat=s["heat_J"] / 1000, stress=s["stress"], end=s["end_reason"])
                        for _, _, s in results])
ui.table(summ_df, {"Scenario": ("Scenario",), "current": ("Current (A)", "%.2f"), "dur": ("Duration (min)", "%.1f"),
                   "ah": ("Charge moved (Ah)", "%.3f"), "wh": ("Net energy (Wh)", "%.2f"),
                   "soc": ("Final SOC (%)", "%.1f", "SOC 0 % is the 2 A cut-off point used in calibration; lower rates "
                           "deliver slightly more (rate-capacity effect), so SOC can end below 0 %."),
                   "tmax": ("Peak T (°C)", "%.1f"), "dT": ("Rise (K)", "%.1f"), "heat": ("Heat (kJ)", "%.2f"),
                   "stress": ("Stress",), "end": ("Termination",)})
st.caption("Thermal stress is an illustrative band against common 18650 datasheet limits (45 °C and 60 °C), not a safety "
           "prediction. Heat is the time integral of the heat generation I²R0 + V1²/R1 − I·T·dU/dT.")

a, b = st.columns(2, gap="large")
with a:
    f2 = go.Figure()
    for i, (label, df, summ) in enumerate(results):
        f2.add_trace(go.Scatter(x=df.Ah_out.abs(), y=df.V, name=label, line=dict(color=ui.SERIES[i % 10], width=2)))
    f2.update_layout(xaxis_title="Charge moved (Ah)", yaxis_title="Terminal voltage (V)")
    ui.chart(f2, "Voltage vs charge throughput", ("Simulated",), height=340, key="b_vq")
with b:
    f3 = go.Figure()
    for i, (label, df, summ) in enumerate(results):
        f3.add_trace(go.Scatter(x=df.t_s / 60, y=df.Qgen_W, name=label, line=dict(color=ui.SERIES[i % 10], width=1.6)))
    f3.update_layout(xaxis_title="Time (min)", yaxis_title="Heat generation (W)")
    ui.chart(f3, "Heat generation", ("Simulated",), height=340, key="b_q")

with st.expander("Parameter sensitivity (tornado, ±20 % one at a time)"):
    st.caption("Which identified parameters drive this experiment's outcome? Each parameter is moved by ±20 % while the "
               "others stay fixed, at the first scenario's operating point.")
    out_key = st.radio("Output", list(OUTPUTS), format_func=OUTPUTS.get, horizontal=True, key="b_tor_out")
    n_sim = 2 * sum(1 for k in PARAMS if k != "R1_chg" or p.R1_chg is not None)
    if st.button(f"Run sensitivity analysis ({n_sim} simulations)", icon=":material/insights:", key="b_tor_go"):
        label, cr, s0, ta, sh, rg = runs[0]
        st_ = state_for(s0, ta, sh, rg)
        with st.spinner("Running perturbed simulations…"):
            tor, base = tornado(p, th, st_, steps_for(cr * p.Q_Ah * st_.soh), rel=0.2, dt=2.0)
        st.session_state["b_tornado"] = (tor, base, label)
    if "b_tornado" in st.session_state:
        tor, base, label = st.session_state["b_tornado"]
        d = tor[tor.output == out_key].copy()
        order = d.groupby("parameter").delta.apply(lambda x: x.abs().max()).sort_values().index.tolist()
        fig = go.Figure()
        for chg, col in (("-20 %", ui.SERIES[1]), ("+20 %", ui.SERIES[2])):
            dd = d[d.change == chg].set_index("parameter").loc[order]
            fig.add_trace(go.Bar(y=order, x=dd.delta, orientation="h", name=f"parameter {chg}", marker_color=col))
        fig.update_layout(barmode="overlay", xaxis_title=f"Change in {OUTPUTS[out_key].lower()}", bargap=0.35)
        ui.chart(fig, f"Sensitivity of {OUTPUTS[out_key].lower()}, base {base[out_key]:.3g} ({label})", ("Simulated",),
                 height=360, key="b_tor_fig")

ui.section("Scenarios", "Save single runs and overlay them (A/B/C comparison).")
c1, c2, c3 = st.columns([2.2, 1, 1], vertical_alignment="bottom")
name = c1.text_input("Scenario name", value=results[0][0] if len(results) == 1 else sweep, key="b_name")
if c2.button("Save current run", disabled=len(results) != 1, icon=":material/bookmark_add:", width="stretch",
             help="Available in single-run mode."):
    st.session_state.setdefault("scenarios", []).append(dict(name=name, df=results[0][1], summary=results[0][2]))
    st.toast(f"Saved scenario '{name}'")
if c3.button("Clear scenarios", icon=":material/delete_sweep:", width="stretch"):
    st.session_state["scenarios"] = []
scen = st.session_state.get("scenarios", [])
if scen:
    f4 = make_subplots(rows=1, cols=3, horizontal_spacing=0.07,
                       subplot_titles=("Terminal voltage (V)", "State of charge (%)", "Cell temperature (°C)"))
    for i, s in enumerate(scen):
        col = ui.SERIES[i % 10]
        x = s["df"].t_s / 60
        f4.add_trace(go.Scatter(x=x, y=s["df"].V, name=s["name"], legendgroup=s["name"], line=dict(color=col, width=2)), 1, 1)
        f4.add_trace(go.Scatter(x=x, y=100 * s["df"].SOC, showlegend=False, legendgroup=s["name"], line=dict(color=col, width=2)), 1, 2)
        f4.add_trace(go.Scatter(x=x, y=s["df"].T_C, showlegend=False, legendgroup=s["name"], line=dict(color=col, width=2)), 1, 3)
    for c in (1, 2, 3):
        f4.update_xaxes(title_text="Time (min)", row=1, col=c)
    ui.chart(f4, "Saved scenarios", ("Simulated",), height=380, key="b_scen")
    sc = pd.DataFrame([dict(Scenario=s["name"], dur=s["summary"]["duration_min"], ah=abs(s["summary"]["Ah_discharged"]),
                            wh=s["summary"]["Wh_net"], tmax=s["summary"]["T_max_C"], dT=s["summary"]["dT_max_C"])
                       for s in scen])
    ui.table(sc, {"Scenario": ("Scenario",), "dur": ("Duration (min)", "%.1f"), "ah": ("Charge moved (Ah)", "%.3f"),
                  "wh": ("Net energy (Wh)", "%.2f"), "tmax": ("Peak T (°C)", "%.1f"), "dT": ("Rise (K)", "%.1f")})

cfg = dict(experiment=exp, mode=sweep, c_rate=crate, soc0_pct=soc0, ambient_C=amb, soh_pct=soh_pct, ageing_source=src,
           resistance_factor=rgrow, v_min=vmin, v_max=vmax, twin=t["label"], twin_params=t["ecm"] | {"thermal": t["thermal"]})
allts = pd.concat([df.assign(scenario=label) for label, df, _ in results], ignore_index=True)
ui.export_bar("experiment bench", cfg, summ_df.to_dict(orient="records"), allts, fig, key="bench")
ui.footer()
