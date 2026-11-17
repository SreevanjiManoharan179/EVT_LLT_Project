import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from g14lab import ecm, thermal, ui
from g14lab import evaluation as ev
from g14lab import simulator as sim

ui.page_header("Thermal & fast charging",
               "How strongly the cell heats up as C-rate and ambient temperature change, according to the calibrated "
               "twin, next to real DC fast-charging sessions of two Tesla Model Y packs.")
p, th, t = ui.twin()
tab_map, tab_val, tab_fc = st.tabs(["Thermal stress map (twin)", "Measured vs predicted temperature",
                                    "Tesla DC fast charging (measured)"])

with tab_map:
    with st.container(border=True):
        c1, c2, c3 = st.columns(3, gap="medium")
        mode = c1.radio("Operation", ["Discharge", "Charge (CC phase)"], horizontal=True, key="t_mode")
        hA_mult = c2.slider("Cooling: multiplier on hA", 0.5, 5.0, 1.0, 0.5, key="t_h",
                            help="1 = the calibrated still-air value; larger values approximate forced-air or liquid cooling.")
        soh_pct = c3.slider("Ageing: SOH (% of 2.0 Ah)", 60, 100, int(round(ui.twin_soh_pct(p))), key="t_soh")
    crates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ambs = [0, 4, 10, 25, 35, 45]

    @st.cache_data(show_spinner=False)
    def grid(pd_, thd, mode, hm, sh, rg):
        pp = ecm.ECMParams.from_dict(pd_)
        tt = thermal.ThermalParams(**thd)
        tt.hA *= hm
        rows = []
        for a in ambs:
            for c in crates:
                soh_ratio = (sh / 100 * 2.0) / pp.Q_Ah
                I = c * pp.Q_Ah * soh_ratio
                st_ = sim.TwinState(soc0=1.0 if mode == "Discharge" else 0.05, T0_C=a, T_amb_C=a, soh=soh_ratio, r_scale=rg)
                steps = [{"type": "cc", "current_A": I if mode == "Discharge" else -I,
                          "until_V": 2.7 if mode == "Discharge" else 4.2}]
                _, s = sim.run_protocol(pp, tt, st_, steps, dt=2.0)
                rows.append(dict(ambient=a, crate=c, Tmax=s["T_max_C"], dT=s["dT_max_C"], Ah=abs(s["Ah_discharged"]),
                                 minutes=s["duration_min"]))
        return pd.DataFrame(rows)

    with st.spinner("Running 36 coupled electro-thermal simulations…"):
        G = grid(t["ecm"], t["thermal"], mode, hA_mult, soh_pct, ui.resistance_factor(soh_pct, p))
    xl = [f"{c:g}C" for c in crates]
    yl = [f"{a} °C" for a in ambs]
    a, b = st.columns(2, gap="large")
    with a:
        z = G.pivot(index="ambient", columns="crate", values="dT").loc[ambs, crates].values
        fig = go.Figure(go.Heatmap(z=z, x=xl, y=yl, colorscale="YlOrRd", text=np.round(z, 1), texttemplate="%{text}",
                                   colorbar=dict(title="ΔT (K)", thickness=12), hovertemplate="%{y}, %{x}: %{z:.1f} K<extra></extra>"))
        ui.heatmap_axes(fig, "C-rate", "Ambient temperature")
        ui.chart(fig, "Peak temperature rise ΔT (K)", ("Simulated",), height=390, key="t_dT", legend="none")
    with b:
        z = G.pivot(index="ambient", columns="crate", values="Ah").loc[ambs, crates].values
        fig = go.Figure(go.Heatmap(z=z, x=xl, y=yl, colorscale="Blues", text=np.round(z, 2), texttemplate="%{text}",
                                   colorbar=dict(title="Ah", thickness=12), hovertemplate="%{y}, %{x}: %{z:.2f} Ah<extra></extra>"))
        ui.heatmap_axes(fig, "C-rate", "Ambient temperature")
        ui.chart(fig, "Charge moved before the voltage limit (Ah)", ("Simulated",), height=390, key="t_ah", legend="none")
    ui.note("Self-heating scales with I²R, and R rises in the cold (Arrhenius, Ea fitted on the NASA cells). Cold, high-rate "
            "operation therefore hits the voltage limit early, while warm operation heats the cell further. Charging warms "
            "the cell much less than discharging at the same C-rate because its polarisation resistance is far lower. Rows at 0 °C "
            "and columns above 2.2C lie outside the NASA test conditions and are extrapolations; at 4 A the reference twin's validation shows "
            "peak-temperature errors of −7 to +3 K. The map is a relative comparison from a lumped model; it does not "
            "predict thermal runaway.", warn=True)

with tab_val:
    dc = ui.discharge_cycles()
    choices = {"B0005: 24 °C, 2 A (calibration cell)": "B0005", "B0036: 24 °C, 2 A": "B0036", "B0033: 24 °C, 4 A": "B0033",
               "B0025: 24 °C, 4 A pulses": "B0025", "B0029: 43 °C, 4 A": "B0029", "B0047: 4 °C, 1 A": "B0047"}
    c1, c2, _ = st.columns([1.6, 1, 1.4], vertical_alignment="bottom")
    lab_ = c1.selectbox("Measured cycle from", list(choices), key="tv_c")
    cell = choices[lab_]
    g = dc[(dc.cell_id == cell) & dc.usable_for_soh & ~dc.flag_capacity_mismatch]
    k = c2.selectbox("Discharge", g.discharge_index.tolist(), index=min(10, len(g) - 1), key="tv_k")
    c = ui.cycle(cell, k)
    soc = ecm.soc_reference(c["t"], c["I"], c["Q"])
    _, v1 = ecm.simulate_voltage(p, c["t"], c["I"], soc, T_C=c["T"])
    R0, R1 = p.r0_at(c["T"]), p.r1_at(c["T"], 1.0, c["I"])
    Ts, q = thermal.simulate_temperature(th, c["t"], c["I"], v1, R0, R1, c["ambient"], c["T"][0])
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.62, 0.38],
                        subplot_titles=("Cell temperature (°C)", "Heat generation (W)"))
    fig.add_trace(go.Scatter(x=c["t"] / 60, y=c["T"], name="Measured", line=dict(color=ui.PROV["Measured"], width=2.4)), 1, 1)
    fig.add_trace(go.Scatter(x=c["t"] / 60, y=Ts, name="Predicted by twin", line=dict(color=ui.PROV["Simulated"], width=2.4, dash="dash")), 1, 1)
    fig.add_trace(go.Scatter(x=c["t"] / 60, y=c["I"] ** 2 * R0, name="ohmic I²R0", stackgroup="q",
                             line=dict(color=ui.SERIES[2], width=0.5)), 2, 1)
    fig.add_trace(go.Scatter(x=c["t"] / 60, y=v1 ** 2 / np.maximum(R1, 1e-6), name="polarisation V1²/R1", stackgroup="q",
                             line=dict(color=ui.SERIES[3], width=0.5)), 2, 1)
    if abs(th.dUdT) >= 1e-4:
        fig.add_trace(go.Scatter(x=c["t"] / 60, y=-c["I"] * (Ts + 273.15) * th.dUdT, name="reversible −I·T·dU/dT",
                                 stackgroup="q", line=dict(color=ui.SERIES[4], width=0.5)), 2, 1)
    fig.update_xaxes(title_text="Time (min)", row=2, col=1)
    ui.chart(fig, f"{cell} discharge {k}", ("Measured", "Simulated"), height=560, key="tv_fig")
    ui.readouts([("Temperature RMSE", f"{ev.rmse(c['T'], Ts):.2f}", "K"),
                 ("Peak temperature, measured", f"{c['T'].max():.1f}", "°C"),
                 ("Peak temperature, predicted", f"{Ts.max():.1f}", "°C"),
                 ("Chamber ambient", f"{c['ambient']:.0f}", "°C"),
                 ("Heat released", f"{np.trapezoid(q, c['t']):.0f}", "J")], cols=5, accent=ui.PROV["Simulated"])
    st.caption("Heat model I²R0 + V1²/R1 − I·T·dU/dT with resistances at the measured cell temperature. C_th, hA and the "
               "effective entropic coefficient dU/dT are identified on B0005 discharges and charges (cell plus fixture), "
               "so errors grow at 4 A and 43 °C.")

with tab_fc:
    if not ui.bev_available():
        st.info("Vehicle data not available.")
    else:
        ss = ui.bev_sessions()
        fc = ss[ss.kind == "fast_charge"]
        qv = ss[ss.q_implied_Ah.notna()].groupby("vehicle").q_implied_Ah.median()
        F = pd.concat([ui.bev_ts("fast_charge", s) for s in fc.session_id], ignore_index=True)
        F["P_kW"] = -F.v_pack_V * F.i_pack_A / 1000
        F["C_rate"] = -F.i_pack_A / F.vehicle.map(qv)
        F = F[F.i_pack_A < -5]
        cmap = {"BEV1": ui.SERIES[0], "BEV2": ui.SERIES[2]}
        a, b = st.columns(2, gap="large")
        with a:
            fig = px.scatter(F.iloc[::3], x="soc_bms_pct", y="P_kW", color="vehicle", opacity=0.4, color_discrete_map=cmap,
                             labels=dict(soc_bms_pct="BMS SOC (%)", P_kW="Charging power (kW)", vehicle="Vehicle"))
            fig.update_traces(marker=dict(size=4))
            ui.chart(fig, f"DC charging power taper, {fc.session_id.nunique()} sessions", ("Measured", "Reference"),
                     height=400, key="fc_p")
        with b:
            fig = px.scatter(F.iloc[::3], x="soc_bms_pct", y="C_rate", color="vehicle", opacity=0.4, color_discrete_map=cmap,
                             labels=dict(soc_bms_pct="BMS SOC (%)", C_rate="Pack C-rate (1/h)", vehicle="Vehicle"))
            fig.update_traces(marker=dict(size=4))
            ui.chart(fig, "Pack C-rate during DC charging", ("Measured", "Reference"), height=400, key="fc_c",
                     sub=f"C-rate = |I| / implied pack capacity ({', '.join(f'{v}: {q:.0f} Ah' for v, q in qv.items())}).")
        a, b = st.columns(2, gap="large")
        with a:
            fig = px.scatter(F.iloc[::3], x="soc_bms_pct", y="t_max_C", color="vehicle", opacity=0.4, color_discrete_map=cmap,
                             labels=dict(soc_bms_pct="BMS SOC (%)", t_max_C="Max pack temperature (°C)", vehicle="Vehicle"))
            fig.update_traces(marker=dict(size=4))
            ui.chart(fig, "Pack temperature during DC charging", ("Measured",), height=380, key="fc_t")
        with b:
            fc2 = fc.copy()
            fc2["temp_rise_K"] = fc2.t_pack_max_C - fc2.t_pack_start_C
            fig = px.scatter(fc2, x="p_max_charge_kW", y="temp_rise_K", color="vehicle", size="duration_min", hover_name="session_id",
                             color_discrete_map=cmap, labels=dict(p_max_charge_kW="Peak charging power (kW)",
                                                                  temp_rise_K="Pack temperature rise (K)", vehicle="Vehicle",
                                                                  duration_min="Duration (min)"))
            ui.chart(fig, "Charging intensity vs pack heating, per session", ("Measured",), height=380, key="fc_s")
        ui.note("The BMS tapers charging power as SOC rises (approach to the voltage limit) and with pack temperature: the "
                "same two limits that end the CC phase and heat the cell on the twin's stress map. Peak pack C-rates reach "
                "about 2C, inside the range the NASA-calibrated cell twin covers. The pack data are measured, the SOC is "
                "BMS-reported, and the LFP and NCA vehicles are kept separate.")
        ui.table(fc.sort_values("start"), {"session_id": ("Session",), "vehicle": ("Vehicle",), "duration_min": ("Duration (min)", "%.1f"),
                                           "soc_start_pct": ("SOC start (%)", "%.1f"), "soc_end_pct": ("SOC end (%)", "%.1f"),
                                           "p_max_charge_kW": ("Peak power (kW)", "%.0f"), "i_max_charge_A": ("Peak current (A)", "%.0f"),
                                           "t_pack_start_C": ("Pack T start (°C)", "%.1f"), "t_pack_max_C": ("Pack T max (°C)", "%.1f"),
                                           "t_amb_mean_C": ("Ambient (°C)", "%.1f")}, height=280)
ui.footer()
