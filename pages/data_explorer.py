import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from g14lab import paths, ui
from g14lab.profiling import profile

ui.page_header("Data hub & explorer",
               "Inspect the downloaded datasets without opening raw files: test conditions, every measured signal, "
               "distributions, relationships, impedance spectra and data quality.")

tab_cells, tab_veh, tab_q = st.tabs(["Lab cells (NASA PCoE)", "Vehicles (KU Leuven Tesla)", "Data quality & dictionary"])
NUM = ["capacity_Ah", "energy_Wh", "mean_voltage_loaded_V", "T_max_C", "T_rise_C", "t_to_Tmax_s", "t_to_3p6V_s",
       "t_cc_s", "t_cv_s", "q_cc_Ah", "ic_peak_AhV", "ic_peak_V", "ic_area_Ah", "Re_ohm", "Rct_ohm",
       "r_dc_first_sample_ohm", "mean_current_A", "ambient_C"]

# ----------------------------------------------------------------------------------------------- NASA
with tab_cells:
    cells = ui.cells()
    dc = ui.discharge_cycles()
    ui.note("NASA Ames Prognostics Center of Excellence, <i>Li-ion Battery Aging Data Set</i> (Saha & Goebel). "
            "34 commercial 18650 cells rated 2 Ah, charged CC-CV at 1.5 A to 4.2 V and discharged at different loads "
            "and ambient temperatures until 20–30 % capacity fade. Public U.S. Government data.")
    ui.table(cells, {
        "cell_id": ("Cell",), "ambient_values_C": ("Ambient (°C)",), "discharge_profile": ("Discharge profile",),
        "cutoff_V": ("Cut-off (V)", "%.1f"), "n_discharge": ("Discharges", "%d"), "n_charge": ("Charges", "%d"),
        "n_impedance": ("EIS tests", "%d"), "first_capacity_Ah": ("First Q (Ah)", "%.3f"),
        "last_capacity_Ah": ("Last Q (Ah)", "%.3f"), "n_flagged_low_capacity": ("Flagged runs", "%d"),
        "note": ("NASA note",)}, height=250)

    c1, c2, c3 = st.columns([1, 1.2, 2.4], vertical_alignment="bottom")
    cell = c1.selectbox("Cell", cells.cell_id.tolist(), index=0, key="ex_cell")
    op = c2.selectbox("Operation", ["Discharge", "Charge (CC-CV)", "Impedance (EIS)"], key="ex_op")
    g = dc[dc.cell_id == cell]
    if op == "Discharge":
        n = int(g.discharge_index.max())
        sel = c3.multiselect("Discharge cycles to overlay", list(range(1, n + 1)), default=sorted({1, max(1, n // 2), n}),
                             key="ex_sel_d")
        if sel:
            ts = ui.discharge_ts(cell, tuple(sel))
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                                subplot_titles=("Terminal voltage (V)", "Current (A, positive = discharge)", "Cell temperature (°C)"))
            for i, (k, s) in enumerate(ts.groupby("discharge_index")):
                col = ui.SERIES[i % len(ui.SERIES)]
                for r, y in enumerate(["V", "I", "T"], start=1):
                    fig.add_trace(go.Scatter(x=s.t_s, y=s[y], mode="lines", line=dict(color=col, width=1.7),
                                             name=f"cycle {k}", legendgroup=str(k), showlegend=(r == 1)), r, 1)
            fig.update_xaxes(title_text="Time since start of discharge (s)", row=3, col=1)
            ui.chart(fig, f"{cell}: measured discharge signals", ("Measured",), height=640, key="ex_d")
            st.download_button("Download selected cycles (CSV)", ts.to_csv(index=False).encode(),
                               f"{cell}_discharge_cycles.csv", "text/csv", icon=":material/download:")
    elif op.startswith("Charge"):
        cg = ui.charge_cycles()
        cg = cg[cg.cell_id == cell]
        n = int(cg.charge_index.max())
        sel = c3.multiselect("Charge cycles to overlay", list(range(1, n + 1)), default=sorted({2, max(2, n // 2), n}),
                             key="ex_sel_c")
        if sel:
            ts = ui.charge_ts(cell, tuple(sel))
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                                subplot_titles=("Terminal voltage (V)", "Current (A, negative = charge)", "Cell temperature (°C)"))
            for i, (k, s) in enumerate(ts.groupby("charge_index")):
                col = ui.SERIES[i % len(ui.SERIES)]
                for r, y in enumerate(["V", "I", "T"], start=1):
                    fig.add_trace(go.Scatter(x=s.t_s, y=s[y], mode="lines", line=dict(color=col, width=1.5),
                                             name=f"charge {k}", legendgroup=str(k), showlegend=(r == 1)), r, 1)
            fig.update_xaxes(title_text="Time since start of charge (s)", row=3, col=1)
            ui.chart(fig, f"{cell}: measured CC-CV charge signals", ("Measured",), height=640, key="ex_c",
                     sub="CC at 1.5 A until 4.2 V, then CV until 20 mA. The CC phase shortens as the cell ages; the "
                         "SOH page turns this and the incremental-capacity curve into health indicators.")
    else:
        imp = ui.impedance()
        im = imp[imp.cell_id == cell]
        sp = ui.eis(cell)
        n = int(im.impedance_index.max())
        sel = c3.multiselect("EIS tests to overlay", list(range(1, n + 1)), default=sorted({1, max(1, n // 2), n}),
                             key="ex_sel_i")
        a, b = st.columns(2, gap="medium")
        with a:
            fig = go.Figure()
            for i, k in enumerate(sel):
                s = sp[sp.impedance_index == k]
                s = s[(s.Z_real_ohm > 0) & (s.Z_real_ohm < 0.6) & (s.Z_imag_ohm.abs() < 0.4)]
                fig.add_trace(go.Scatter(x=s.Z_real_ohm * 1000, y=-s.Z_imag_ohm * 1000, mode="markers+lines",
                                         name=f"EIS {k}", line=dict(color=ui.SERIES[i % 10], width=1), marker=dict(size=5)))
            fig.update_layout(xaxis_title="Re(Z) (mΩ)", yaxis_title="−Im(Z) (mΩ)")
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            ui.chart(fig, "Nyquist plot, 0.1 Hz – 5 kHz sweep", ("Measured",), height=400, key="ex_nyq")
        with b:
            ok = im[(im.Re_ohm > 0) & (im.Re_ohm < 0.5) & (im.Rct_ohm > 0) & (im.Rct_ohm < 1)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ok.impedance_index, y=1000 * ok.Re_ohm, name="Re (electrolyte)", mode="lines+markers",
                                     marker=dict(size=4)))
            fig.add_trace(go.Scatter(x=ok.impedance_index, y=1000 * ok.Rct_ohm, name="Rct (charge transfer)",
                                     mode="lines+markers", marker=dict(size=4)))
            fig.update_layout(xaxis_title="EIS test number", yaxis_title="Resistance (mΩ)")
            ui.chart(fig, "NASA-fitted EIS resistances over the test", ("Measured",), height=400, key="ex_rerct",
                     sub="Rows with Re outside 0–0.5 Ω or Rct outside 0–1 Ω are hidden as logging glitches.")

    ui.section("Cycle-level view")
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        fig = go.Figure()
        good = g[g.usable_for_soh]
        bad = g[~g.usable_for_soh]
        fig.add_trace(go.Scatter(x=good.discharge_index, y=good.capacity_Ah, mode="lines+markers", name="usable",
                                 line=dict(color=ui.PROV["Measured"], width=1.4), marker=dict(size=4)))
        if len(bad):
            fig.add_trace(go.Scatter(x=bad.discharge_index, y=bad.capacity_Ah, mode="markers", name="flagged",
                                     marker=dict(color=ui.ALERT, size=7, symbol="x")))
        fig.update_layout(xaxis_title="Discharge cycle", yaxis_title="Capacity (Ah)")
        ui.chart(fig, f"{cell}: discharge capacity per cycle", ("Measured",), height=360, key="ex_cap",
                 sub="Flagged runs (capacity < 0.5 Ah) stay in the data but are excluded from SOH models.")
    with b:
        c_1, c_2 = st.columns([1.6, 1], vertical_alignment="bottom")
        var = c_1.selectbox("Distribution of", NUM, index=3, format_func=ui.lab, key="ex_hist")
        scope = c_2.selectbox("Across", ["this cell", "all cells"], key="ex_scope")
        dd = g if scope == "this cell" else dc
        fig = px.histogram(dd, x=var, nbins=40, color_discrete_sequence=[ui.PROV["Measured"]])
        fig.update_layout(xaxis_title=ui.lab(var), yaxis_title="Cycles", bargap=0.04)
        ui.chart(fig, "Histogram", ("Measured",), height=290, key="ex_h")

    ui.section("Relationships between cycle-level variables")
    c_1, c_2, c_3 = st.columns([1, 1, 1.4], vertical_alignment="bottom")
    xv = c_1.selectbox("x", NUM, index=NUM.index("Re_ohm"), format_func=ui.lab, key="ex_x")
    yv = c_2.selectbox("y", NUM, index=0, format_func=ui.lab, key="ex_y")
    sc = c_3.selectbox("Scope", ["this cell", "reference cells B0005/6/7/18", "all cells"], key="ex_corr")
    base = g if sc == "this cell" else (dc[dc.cell_id.isin(["B0005", "B0006", "B0007", "B0018"])]
                                         if sc.startswith("ref") else dc)
    a, b = st.columns([1, 1.15], gap="large")
    with a:
        fig = px.scatter(base[base.usable_for_soh], x=xv, y=yv, color="discharge_index", symbol="cell_id",
                         color_continuous_scale="Viridis", opacity=0.8,
                         labels={xv: ui.lab(xv), yv: ui.lab(yv), "discharge_index": "Cycle", "cell_id": "Cell"})
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(coloraxis_colorbar=dict(title="Cycle", thickness=12))
        ui.chart(fig, "Scatter, coloured by cycle number", ("Measured",), height=520, key="ex_sc", legend="top")
    with b:
        cols = [c for c in NUM if base[c].nunique(dropna=True) > 1]
        corr = base[cols].corr(method="spearman")
        names = [ui.LABELS.get(c, (c,))[0] for c in cols]
        fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu_r",
                                   colorbar=dict(title="ρ", thickness=12),
                                   hovertemplate="%{y} vs %{x}: ρ = %{z:.2f}<extra></extra>"))
        fig.update_xaxes(tickangle=-45, tickfont=dict(size=11), showgrid=False)
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=11), showgrid=False)
        ui.chart(fig, "Spearman rank correlation (hover for values)", ("Measured",), height=520, key="ex_corrfig",
                 legend="none")
    st.caption("Constant columns are dropped from the correlation. Strong correlation with capacity marks a candidate "
               "health indicator; the SOH page tests them on cells held out of training.")
    st.download_button("Download the cycle table for this cell (CSV)", g.to_csv(index=False).encode(), f"{cell}_cycles.csv",
                       "text/csv", icon=":material/download:")

# ----------------------------------------------------------------------------------------------- Tesla
with tab_veh:
    if not ui.bev_available():
        st.info("Vehicle data not processed yet. Run `python -m g14lab.telemetry` after placing the KU Leuven zip in "
                "data/raw/kuleuven_bev/.")
    else:
        ss = ui.bev_sessions()
        ui.note("KU Leuven RDR, <i>BEV energy dynamics dataset</i> (Yasko et al., 2025, doi:10.48804/8KPDTW, CC BY 4.0). "
                "Two Tesla Model Y vehicles (BEV1, BEV2), CAN-bus signals decoded by the authors at 1 s. SOC is "
                "<b>BMS-reported</b>, not an independent measurement. GPS columns were dropped during processing.")
        kinds = {"fast_charge": "DC fast charging", "slow_charge": "AC slow charging", "drive": "Driving"}
        a, b, c = st.columns([1, 1, 2.4], vertical_alignment="bottom")
        veh = a.selectbox("Vehicle", ["BEV1", "BEV2"], key="vx_v")
        kind = b.selectbox("Session type", list(kinds), format_func=kinds.get, key="vx_k")
        sub = ss[(ss.vehicle == veh) & (ss.kind == kind)].sort_values("start")
        lookup = sub.set_index("session_id")
        sid = c.selectbox("Session", sub.session_id.tolist(), key="vx_s",
                          format_func=lambda s: f"{s}   ({lookup.loc[s, 'duration_min']:.0f} min, SOC "
                                                f"{lookup.loc[s, 'soc_start_pct']:.0f} → {lookup.loc[s, 'soc_end_pct']:.0f} %)")
        ui.table(sub, {"session_id": ("Session",), "start": ("Start",), "duration_min": ("Duration (min)", "%.1f"),
                       "soc_start_pct": ("SOC start (%)", "%.1f"), "soc_end_pct": ("SOC end (%)", "%.1f"),
                       "kwh_net_discharge": ("Net energy out (kWh)", "%.2f"), "distance_km": ("Distance (km)", "%.1f"),
                       "p_max_charge_kW": ("Peak charge power (kW)", "%.0f"), "t_pack_max_C": ("Max pack T (°C)", "%.1f"),
                       "t_amb_mean_C": ("Mean ambient (°C)", "%.1f"), "logged_fraction": ("Logged time", "%.2f")},
                 height=210)
        if sid:
            ts = ui.bev_ts(kind, sid)
            ts["P_kW"] = ts.v_pack_V * ts.i_pack_A / 1000
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                                subplot_titles=("BMS SOC (%)", "Pack voltage (V)", "Pack current (A) and power (kW), positive = discharge",
                                                "Pack temperature and ambient (°C)"))
            x = ts.t_s / 60
            fig.add_trace(go.Scatter(x=x, y=ts.soc_bms_pct, name="SOC (BMS)", line=dict(color=ui.PROV["Reference"], width=2.2)), 1, 1)
            fig.add_trace(go.Scatter(x=x, y=ts.v_pack_V, name="Voltage", line=dict(color=ui.PROV["Measured"], width=1.4)), 2, 1)
            fig.add_trace(go.Scatter(x=x, y=ts.i_pack_A, name="Current", line=dict(color=ui.SERIES[0], width=1)), 3, 1)
            fig.add_trace(go.Scatter(x=x, y=ts.P_kW, name="Power", line=dict(color=ui.SERIES[2], width=1)), 3, 1)
            fig.add_trace(go.Scatter(x=x, y=ts.t_max_C, name="T max", line=dict(color=ui.ALERT, width=1.6)), 4, 1)
            fig.add_trace(go.Scatter(x=x, y=ts.t_min_C, name="T min", line=dict(color=ui.SERIES[0], width=1.6)), 4, 1)
            fig.add_trace(go.Scatter(x=x, y=ts.t_amb_C, name="Ambient", line=dict(color=ui.PROV["Reference"], dash="dot")), 4, 1)
            fig.update_xaxes(title_text="Time (min)", row=4, col=1)
            ui.chart(fig, f"{sid}: pack telemetry", ("Measured", "Reference"), height=780, key="vx_fig")
            a, b = st.columns([1, 1.2], gap="large")
            with a:
                var = st.selectbox("Distribution of", ["i_pack_A", "v_pack_V", "P_kW", "t_max_C", "soc_bms_pct", "speed_kph"],
                                   format_func=ui.lab, key="vx_hist")
                fig = px.histogram(ts, x=var, nbins=50, color_discrete_sequence=[ui.PROV["Measured"]])
                fig.update_layout(xaxis_title=ui.lab(var), yaxis_title="Samples (1 s)", bargap=0.04)
                ui.chart(fig, "Histogram", ("Measured",), height=300, key="vx_h")
            with b:
                st.markdown('<div class="g14-ch"><span class="g14-ch-t">Session profile</span></div>', unsafe_allow_html=True)
                pr = profile(ts.drop(columns=["session_id", "vehicle"]))
                ui.table(pr, {"column": ("Signal",), "dtype": ("Type",), "missing": ("Missing", "%d"),
                              "unique": ("Unique", "%d"), "min": ("Min", "%.3g"), "median": ("Median", "%.3g"),
                              "max": ("Max", "%.3g")}, height=330)
            st.download_button("Download this session (CSV)", ts.to_csv(index=False).encode(), f"{sid}.csv", "text/csv",
                               icon=":material/download:")

# ----------------------------------------------------------------------------------------------- quality
with tab_q:
    dc = ui.discharge_cycles()
    ui.readouts([
        ("Discharge records", f"{len(dc):,}", "", "34 NASA cells"),
        ("Flagged low capacity", f"{int(dc.flag_low_capacity.sum())}", "", "capacity < 0.5 Ah"),
        ("Capacity field mismatch", f"{int(dc.flag_capacity_mismatch.sum())}", "", "NASA field vs ∫I dt > 0.15 Ah"),
        ("Usable for SOH models", f"{int((dc.usable_for_soh & ~dc.flag_capacity_mismatch).sum()):,}", ""),
        ("Duplicate records", f"{int(dc.duplicated(subset=['cell_id', 'discharge_index']).sum())}", ""),
        ("Median sampling interval", f"{dc.median_dt_s.median():.1f}", "s",
         f"range {dc.median_dt_s.min():.1f}–{dc.median_dt_s.max():.1f} s"),
    ], cols=6)
    ui.section("Discharge-cycle table (NASA, derived): column profile")
    spec = {"column": ("Column",), "dtype": ("Type",), "non_null": ("Non-null", "%d"), "missing": ("Missing", "%d"),
            "missing_pct": ("Missing (%)", "%.2f"), "unique": ("Unique", "%d"), "min": ("Min", "%.4g"),
            "median": ("Median", "%.4g"), "max": ("Max", "%.4g")}
    ui.table(profile(dc), spec, height=360)
    if ui.bev_available():
        ui.section("Vehicle sessions (KU Leuven, derived summaries): column profile")
        ui.table(profile(ui.bev_sessions()), spec, height=300)
    ui.section("Data dictionary")
    p = paths.DATA / "data_dictionary.md"
    if p.exists():
        st.markdown(p.read_text())
ui.footer()
