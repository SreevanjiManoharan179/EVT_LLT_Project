import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from g14lab import ecm, thermal, ui
from g14lab import evaluation as ev
from g14lab import simulator as sim

ui.page_header("Calibrate the twin",
               "Identify the battery model from measured data: open-circuit-voltage curve, ohmic resistance R0, "
               "polarisation R1–C1 and the lumped thermal parameters. The identified twin is used by every experiment "
               "page; its parameters are shown in the sidebar passport.")

dc = ui.discharge_cycles()
ok_cells = ["B0005", "B0006", "B0007", "B0018", "B0036", "B0025", "B0026", "B0027", "B0028", "B0029", "B0030",
            "B0031", "B0032", "B0033", "B0034"]
p_now, th_now, t_now = ui.twin()
INHERIT = ("arrhenius_fit_R2", "arrhenius_n", "arrhenius_cells", "map_cells_attempted", "current_exponent_beta",
           "calibrated_T_range_C", "map_note", "map_R_ref_ohm", "map_T_ref_C", "resistance_growth")

with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1, 2.2, 1, 1.25], vertical_alignment="bottom")
    cell = c1.selectbox("Calibration cell", ok_cells, index=0, help="Cells with stable early-life data.")
    g = dc[(dc.cell_id == cell) & dc.usable_for_soh & ~dc.flag_capacity_mismatch]
    idx = g.discharge_index.tolist()
    train = c2.multiselect("Training discharges (early life recommended)", idx, default=idx[1:6])
    rest = [i for i in idx if i not in train]
    val_idx = c3.selectbox("Held-out discharge", rest, index=min(40, max(0, len(rest) - 1)))
    run = c4.button("Identify from measured data", type="primary", disabled=len(train) == 0, width="stretch",
                    icon=":material/tune:")
    re_cell = ui.eis_re(cell)
    o1, o2, o3 = st.columns([1.15, 1.15, 1])
    use_chg = o1.toggle("Fit the CC-CV charge after each discharge too", value=True,
                        help="Identifies the charge-direction polarisation resistance. Fitted on discharges alone, the "
                             "model applies the large discharge overpotential to charging and reaches 4.2 V far too early.")
    use_eis = o2.toggle(f"Bound R0 by the EIS ohmic resistance ({re_cell*1000:.1f} mΩ)" if re_cell else
                        "Bound R0 by the EIS ohmic resistance (no EIS for this cell)", value=bool(re_cell),
                        disabled=not re_cell,
                        help="The instantaneous resistance seen at a 2.5–19 s sampling interval cannot be smaller than "
                             "the high-frequency resistance Re that NASA measured by EIS on the same cell.")
    use_T = o3.toggle(f"Temperature-aware fit (Ea {p_now.Ea_R0/1000:.1f} kJ/mol)", value=p_now.Ea_R0 > 0,
                      disabled=p_now.Ea_R0 <= 0,
                      help="R0 and R1 follow the Arrhenius law along the measured cell temperature during the fit "
                           "(the cell warms by 10–15 K in a 2 A discharge), so they refer to 24 °C.")

if run:
    with st.spinner("Non-linear least squares on measured voltage, then on measured temperature…"):
        cyc = [ui.cycle(cell, i) for i in train]
        chg = [c for c in (ui.charge_after(cell, i) for i in train) if c is not None] if use_chg else []
        src = f"{cell} discharges {min(train)}–{max(train)}" + (f" + {len(chg)} charges" if chg else "")
        p = ecm.identify_ecm(cyc + chg, source=src, Ea=p_now.Ea_R0 if use_T else None, T_ref_C=24.0,
                             r0_min=re_cell if (use_eis and re_cell) else 0.005)
        if not use_T:
            p.Ea_R0, p.Ea_R1 = p_now.Ea_R0, p_now.Ea_R1  # temperature law from the multi-cell map
        p.extra.update({k: v for k, v in p_now.extra.items() if k in INHERIT})
        if use_eis and re_cell:
            p.extra["eis_Re_ohm"] = re_cell
        th = thermal.identify_thermal(cyc + chg, p, [c_["ambient"] for c_ in cyc + chg], source=src)
        st.session_state["cal_candidate"] = dict(ecm=p.to_dict(), thermal=th.to_dict(), cell=cell, train=train)

cand = st.session_state.get("cal_candidate")
if cand:
    p = ecm.ECMParams.from_dict(cand["ecm"])
    th = thermal.ThermalParams(**cand["thermal"])
    head = f"Candidate calibration from {cand['cell']} (not active yet)"
else:
    p, th = p_now, th_now
    head = f"Active twin: {t_now['label']}"

ui.section("Identified parameters", head)
a, b = st.columns([1.5, 1], gap="large", vertical_alignment="top")
with a:
    ex = p.extra
    n_d, n_c = ex.get("n_discharges", ex.get("n_cycles", 0)), ex.get("n_charges", 0)
    bound = ex.get("r0_lower_bound_ohm", 0.0)
    r0_note = f"at T_ref = {p.T_ref_C:.0f} °C" if ex.get("temperature_aware") else \
        f"at {p.T_ref_C:.1f} °C, the mean cell temperature in training"
    if bound > 0.006 and p.R0 <= bound * 1.001:
        r0_note += "; held at its lower bound, the EIS Re of the same cell"
    ui.sheet([
        ("R0, ohmic resistance", f"{p.R0*1000:.1f} mΩ", r0_note),
        ("R1, polarisation (discharge)", f"{p.R1*1000:.1f} mΩ", "single RC branch, same reference temperature"),
        *([("R1, polarisation (charge)", f"{p.R1_chg*1000:.1f} mΩ",
            "identified on the charges; R0 is shared by both directions")] if p.R1_chg is not None else []),
        ("τ1 = R1·C1", f"{p.tau1:.1f} s", f"C1 = {p.C1:,.0f} F"),
        ("Q, normalising capacity", f"{p.Q_Ah:.3f} Ah", "mean charge delivered to cut-off in the training discharges"),
        ("Ea, activation energy", f"{p.Ea_R0/1000:.1f} kJ/mol",
         f"multi-cell R(T) map, R² = {ex.get('arrhenius_fit_R2', float('nan')):.3f}, "
         f"{ex.get('arrhenius_n', '–')} cycles from {ex.get('arrhenius_cells', '–')} cells"),
        ("C_th, lumped heat capacity", f"{th.C_th:.1f} J/K", "effective value: cell plus test fixture"),
        ("hA, heat-transfer conductance", f"{th.hA*1000:.1f} mW/K", f"thermal time constant {th.tau_s/60:.0f} min"),
        ("dU/dT, entropic coefficient", f"{th.dUdT*1000:+.2f} mV/K",
         "not identified: needs charge data in training" if th.dUdT == 0 else
         ("negligible: the direction-dependent R1 already explains the charge/discharge heating difference"
          if abs(th.dUdT) < 1e-4 else "effective, SOC-averaged reversible-heat coefficient")),
        ("Training data", f"{n_d} discharges" + (f" + {n_c} charges" if n_c else ""),
         "temperature-aware fit" if ex.get("temperature_aware") else "constant resistances during the fit"),
        ("Voltage fit (training)", f"{p.fit_rmse_mV:.1f} mV RMSE", "unweighted residual over all training samples"),
        ("Temperature fit (training)", f"{th.fit_rmse_C:.2f} K RMSE", "same training data"),
    ])
    if cand:
        cc1, cc2 = st.columns(2)
        if cc1.button("Use this calibration as the active twin", type="primary", icon=":material/check:", width="stretch"):
            ui.set_twin(p, th, f"Calibrated on {cand['cell']}, {len(cand['train'])} discharges" +
                        (f" + {n_c} charges" if n_c else ""))
            st.session_state.pop("cal_candidate")
            st.toast("Active twin updated. Every experiment page now uses these parameters.")
            st.rerun()
        if cc2.button("Discard candidate", width="stretch"):
            st.session_state.pop("cal_candidate")
            st.rerun()
with b:
    with st.expander("Model equations", expanded=True):
        st.latex(r"V_t = \mathrm{OCV}(z) - I\,R_0(T) - V_1")
        st.latex(r"V_{1,k+1} = a\,V_{1,k} + R_1(T,\operatorname{sign} I_k)\,(1-a)\,I_k")
        st.latex(r"a = e^{-\Delta t/\tau_1},\qquad \tau_1 = R_1 C_1")
        st.latex(r"z_{k+1} = z_k - \frac{\eta\, I_k\, \Delta t}{3600\, Q}")
        st.latex(r"R(T) = R_{\mathrm{ref}}\, e^{\frac{E_a}{R_g}\left(\frac{1}{T}-\frac{1}{T_{\mathrm{ref}}}\right)}")
        st.latex(r"\begin{aligned} C_{th}\frac{dT}{dt} &= I^2R_0 + \frac{V_1^2}{R_1} - I\,T\,\frac{dU}{dT} \\ "
                 r"&\quad - hA\,(T-T_{amb}) \end{aligned}")
        st.caption("Current I > 0 means discharge. OCV(z) is a monotone PCHIP curve through 12 knots, fitted jointly "
                   "with R0, R1 (discharge), R1 (charge) and τ1 by non-linear least squares on discharges and the charges "
                   "that follow them. R0 is shared; the polarisation resistance differs with current direction, as in "
                   "HPPC testing. Rest samples are weighted 8× because they separate the OCV level from the resistive drop.")

a, b = st.columns(2, gap="large")
with a:
    s = np.linspace(-0.02, 1.0, 200)
    fig = go.Figure()
    k0 = train[0] if train else val_idx
    d0, c0 = ui.cycle(cell, k0), ui.charge_after(cell, k0)
    if c0 is not None:
        zc = ecm.soc_reference(c0["t"], c0["I"], c0["Q"], soc0=0.0)
        m = c0["I"] < -0.03                                  # CC and CV phases, rest excluded
        fig.add_trace(go.Scatter(x=100 * zc[m], y=c0["V"][m], name=f"measured charge (#{k0})",
                                 line=dict(color=ui.PROV["Measured"], width=1.6)))
    zd = ecm.soc_reference(d0["t"], d0["I"], d0["Q"])
    m = d0["I"] > 0.5
    fig.add_trace(go.Scatter(x=100 * zd[m], y=d0["V"][m], name=f"measured discharge (#{k0})",
                             line=dict(color=ui.PROV["Measured"], width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=100 * s, y=p.ocv(s), name="identified OCV", line=dict(color=ui.PROV["Simulated"], width=2.6)))
    fig.add_trace(go.Scatter(x=100 * np.array(p.soc_knots), y=p.ocv_knots, mode="markers", name="knots", showlegend=False,
                             marker=dict(color=ui.PROV["Simulated"], size=7, line=dict(color="white", width=1))))
    fig.update_layout(xaxis_title="SOC (%)", yaxis_title="Voltage (V)")
    ui.chart(fig, "Identified OCV vs measured curves", ("Measured", "Simulated"), height=360,
             key="cal_ocv", sub=f"{cell}, cycle {k0}. SOC 0 % is the cut-off point at the 2 A reference rate.")
with b:
    pm = ui.metrics_csv("parameter_map.csv")
    if len(pm):
        used = pm[pm.used_in_arrhenius_fit]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=used.mean_T_C, y=1000 * used.R_total_ohm, mode="markers", name="identified per cycle",
                                 marker=dict(color=used.mean_current_A, colorscale="Viridis", size=9, cmin=1, cmax=4,
                                             line=dict(color="white", width=0.6),
                                             colorbar=dict(title="Load (A)", thickness=12)),
                                 text=used.cell_id, hovertemplate="%{text}<br>%{x:.1f} °C, %{y:.0f} mΩ<extra></extra>"))
        T = np.linspace(5, 60, 100)
        ex_ = p.extra
        # the map regression itself (at the 2 A reference load); falls back to the twin's own law
        r_ref, t_ref = ex_.get("map_R_ref_ohm", p.R0 + p.R1), ex_.get("map_T_ref_C", p.T_ref_C)
        law = r_ref * np.exp(p.Ea_R0 / ecm.R_GAS * (1 / (T + 273.15) - 1 / (t_ref + 273.15)))
        fig.add_trace(go.Scatter(x=T, y=1000 * law, name="Arrhenius law (2 A)",
                                 line=dict(color=ui.PROV["Simulated"], dash="dash", width=2)))
        fig.update_layout(xaxis_title="Mean cell temperature (°C)", yaxis_title="R0 + R1 (mΩ)", showlegend=False)
        ui.chart(fig, "Temperature dependence across NASA cells", ("Measured", "Simulated"), height=360, key="cal_map",
                 sub=f"{len(used)} early-life cycles from {used.cell_id.nunique()} cells, OCV fixed, R0/R1/τ1 refitted per "
                     "cycle at its mean temperature. The fitted Ea gives the twin its temperature dependence.")

ui.section("Measured vs simulated", "The measured current is replayed through the twin; the voltage and temperature "
                                    "responses are compared with the measured ones.")
views = {f"Training discharge {train[0] if train else '–'}": ("discharge", train[0] if train else val_idx),
         f"Held-out discharge {val_idx}": ("discharge", val_idx),
         f"Held-out charge (after discharge {val_idx})": ("charge", val_idx)}
which = st.radio("Cycle", list(views), index=1, horizontal=True, label_visibility="collapsed", key="cal_view")
kind, k = views[which]
c = ui.cycle(cell, k) if kind == "discharge" else ui.charge_after(cell, k)
if c is None:
    st.info("NASA recorded no complete CC-CV charge directly after this discharge. Pick another held-out discharge.")
else:
    soc = ecm.soc_reference(c["t"], c["I"], c["Q"], soc0=1.0 if kind == "discharge" else 0.0)
    vm, v1 = ecm.simulate_voltage(p, c["t"], c["I"], soc, T_C=c["T"])
    Ts, q = thermal.simulate_temperature(th, c["t"], c["I"], v1, p.r0_at(c["T"]), p.r1_at(c["T"], 1.0, c["I"]), c["ambient"], c["T"][0])
    tm = c["t"] / 60
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.45, 0.2, 0.35],
                        subplot_titles=("Terminal voltage (V)", "Voltage residual, simulated − measured (mV)",
                                        "Cell temperature (°C)"))
    fig.add_trace(go.Scatter(x=tm, y=c["V"], name="Measured V", line=dict(color=ui.PROV["Measured"], width=2.2)), 1, 1)
    fig.add_trace(go.Scatter(x=tm, y=vm, name="Simulated V", line=dict(color=ui.PROV["Simulated"], width=2.2, dash="dash")), 1, 1)
    fig.add_trace(go.Scatter(x=tm, y=1000 * (vm - c["V"]), name="Residual", line=dict(color=ui.PROV["Simulated"], width=1.2),
                             fill="tozeroy", fillcolor="rgba(36,86,201,0.08)"), 2, 1)
    fig.add_trace(go.Scatter(x=tm, y=c["T"], name="Measured T", line=dict(color=ui.PROV["Measured"], width=2.2)), 3, 1)
    fig.add_trace(go.Scatter(x=tm, y=Ts, name="Simulated T", line=dict(color=ui.PROV["Simulated"], width=2.2, dash="dash")), 3, 1)
    fig.update_xaxes(title_text="Time (min)", row=3, col=1)
    title = f"{cell} discharge {k}" if kind == "discharge" else f"{cell} charge {c['idx']} (CC 1.5 A to 4.2 V, then CV)"
    ui.chart(fig, title, ("Measured", "Simulated"), height=660, key="cal_mvs")
    items = [("Voltage RMSE", f"{1000*ev.rmse(c['V'], vm):.1f}", "mV"), ("Voltage MAE", f"{1000*ev.mae(c['V'], vm):.1f}", "mV"),
             ("Temperature RMSE", f"{ev.rmse(c['T'], Ts):.2f}", "K"),
             ("Peak temperature, measured vs simulated", f"{c['T'].max():.1f} / {Ts.max():.1f}", "°C")]
    if kind == "discharge":
        items.append(("Cycle capacity (∫I dt)", f"{c['Q']:.3f}", "Ah"))
    else:
        # the twin as a charging simulator: CC-CV from SOC 0 with this cycle's capacity and ambient
        ch = ui.charge_cycles()
        r = ch[(ch.cell_id == cell) & (ch.charge_index == c["idx"])]
        df_, _ = sim.run_protocol(p, th, sim.TwinState(soc0=0.0, T0_C=float(c["T"][0]), T_amb_C=c["ambient"],
                                                       soh=c["Q"] / p.Q_Ah, r_scale=p.resistance_factor(100 * c["Q"] / 2.0)),
                                  [{"type": "cc", "current_A": -1.5, "until_V": 4.2},
                                   {"type": "cv", "voltage_V": 4.2, "until_A": 0.02}], dt=2.0)
        ccp = df_[df_.step == 1]
        t_sim = (ccp.t_s.iloc[-1] - ccp.t_s.iloc[0]) / 60
        t_meas = float(r.t_cc_s.iloc[0]) / 60 if len(r) else float("nan")
        items.append(("CC phase to 4.2 V, measured vs twin", f"{t_meas:.0f} / {t_sim:.0f}", "min"))
    ui.readouts(items, cols=5, accent=ui.PROV["Simulated"])
    if kind == "charge":
        st.caption("The CC-phase readout comes from a separate simulation: the twin runs the NASA charging protocol "
                   "(1.5 A to 4.2 V, then 4.2 V until 20 mA) from SOC 0 with this cycle's capacity, the measured "
                   "ambient and resistances scaled by the growth law. The measured CC phase starts from the rested "
                   "post-discharge state.")

# ------------------------------------------------------------------------------------------- digital shadow
ui.section("Digital shadow: how the identified parameters age",
           "R0, R1 and τ1 re-identified on every usable discharge of the reference cells (OCV fixed), compared with "
           "NASA's independent EIS resistances.")
tr = ui.metrics_parquet("param_tracking.parquet")
if len(tr):
    c1, _ = st.columns([1, 3])
    sc = c1.selectbox("Reference cell", sorted(tr.cell_id.unique()), key="cal_shadow_cell")
    t = tr[tr.cell_id == sc].sort_values("discharge_index")
    a, b = st.columns(2, gap="large")
    with a:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=t.discharge_index, y=1000 * t.R_total_ohm, name="R0 + R1 (twin, tracked)",
                                 mode="markers", marker=dict(color=ui.PROV["Simulated"], size=5, opacity=0.6)))
        sm = (1000 * t.R_total_ohm).rolling(9, center=True, min_periods=3).median()
        fig.add_trace(go.Scatter(x=t.discharge_index, y=sm, name="rolling median", line=dict(color=ui.PROV["Simulated"], width=2.5)))
        fig.add_trace(go.Scatter(x=t.discharge_index, y=1000 * (t.Re_ohm + t.Rct_ohm), name="Re + Rct (EIS)",
                                 line=dict(color=ui.PROV["Measured"], width=2)), secondary_y=True)
        fig.update_xaxes(title_text="Discharge cycle")
        fig.update_yaxes(title_text="Twin R0 + R1 (mΩ)", secondary_y=False)
        fig.update_yaxes(title_text="EIS Re + Rct (mΩ)", secondary_y=True, showgrid=False)
        fig.update_yaxes(tickformat=".0f", secondary_y=True)
        ui.chart(fig, f"{sc}: resistance growth, twin vs EIS", ("Simulated", "Measured"), height=380, key="cal_sh1")
    with b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t.discharge_index, y=t.V_RMSE_fresh_mV, name="reference twin (no update)",
                                 line=dict(color=ui.PROV["Reference"], width=2)))
        fig.add_trace(go.Scatter(x=t.discharge_index, y=t.V_RMSE_tracked_fit_mV, name="re-identified on that cycle (fit)",
                                 line=dict(color=ui.PROV["Simulated"], width=2)))
        fig.update_layout(xaxis_title="Discharge cycle", yaxis_title="Voltage RMSE (mV)")
        ui.chart(fig, f"{sc}: voltage error over life", ("Measured", "Simulated"), height=380, key="cal_sh2")
    g_ = p_now.extra.get("resistance_growth")
    growth_text = ""
    if g_ and g_.get("form") == "relative":
        growth_text = (f"A growth law fitted on all {g_['n']} tracked cycles, R/R_early = 1 + {g_['k_per_fade']:.2f}·ΔSOH "
                       f"(each cell normalised to its own first cycles, R² = {g_['R2']:.2f}), drives the ageing slider on "
                       "the experiment bench.")
    elif g_:
        growth_text = (f"A fitted growth law, R/R_ref = {g_['intercept']:.2f} + {g_['k_per_fade']:.2f}·(1 − SOH), "
                       f"R² = {g_['R2']:.2f}, drives the ageing slider on the experiment bench.")
    sv = ui.metrics_csv("shadow_validation.csv")
    if len(sv):
        stages = sv.groupby("life_stage")[["fresh_mV", "shadow_mV", "growth_mV"]].median().reindex(
            ["early (> 85 %)", "mid (75–85 %)", "late (SOH < 75 %)"]).reset_index()
        ui.table(stages, {"life_stage": ("Life stage",), "fresh_mV": ("Reference twin (mV)", "%.1f"),
                          "shadow_mV": ("Shadow: previous 5 cycles (mV)", "%.1f"),
                          "growth_mV": ("Growth law at this SOH (mV)", "%.1f")})
    rho = {c_: g2[g2.Re_ohm.notna()].R_total_ohm.corr(g2.Re_ohm + g2.Rct_ohm, method="spearman")
           for c_, g2 in tr.groupby("cell_id")}
    ui.note("What this shows: the twin's resistance rises with age alongside NASA's independent EIS measurements "
            f"(Spearman ρ of R0 + R1 with Re + Rct: {', '.join(f'{c_} {v:.2f}' for c_, v in rho.items())}). "
            f"{growth_text} "
            "In the fair one-cycle-ahead test above (every variant gets the cycle's measured capacity), updating "
            "resistances helps only late in life. Most of the remaining error comes from the change in the OCV-vs-SOC "
            "shape with ageing, which a fixed-OCV 1-RC model cannot represent. Each per-cycle fit matches its own cycle "
            "better, but that is a fit, not a prediction.")
else:
    st.info("Run `python -m g14lab.calibration` to build the parameter-tracking table.")
ui.footer()
