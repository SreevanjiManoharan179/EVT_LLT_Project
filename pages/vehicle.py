import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from g14lab import estimators as est
from g14lab import evaluation as ev
from g14lab import ui

ui.page_header("Tesla telemetry replay",
               "Real driving and charging of two Tesla Model Y vehicles at 1 s resolution. Replay a session, check the BMS "
               "SOC against Coulomb counting, see how ambient temperature changes real energy use, and identify each "
               "pack's chemistry from its own data.")
if not ui.bev_available():
    st.info("Vehicle data not processed yet. Place the KU Leuven zip in data/raw/kuleuven_bev/ and run "
            "`python -m g14lab.telemetry`.")
    st.stop()
ss = ui.bev_sessions()
KINDS = {"drive": "Driving", "fast_charge": "DC fast charge", "slow_charge": "AC slow charge"}
qv = ss[ss.q_implied_Ah.notna()].groupby("vehicle").q_implied_Ah.median()
tab_rep, tab_soc, tab_fleet, tab_chem = st.tabs(["Session replay", "BMS SOC vs Coulomb counting",
                                                 "Fleet: ambient temperature effect", "Pack chemistry from data"])

with tab_rep:
    c1, c2, c3 = st.columns([1, 1, 2.4], vertical_alignment="bottom")
    veh = c1.selectbox("Vehicle", ["BEV1", "BEV2"], key="vr_v")
    kind = c2.selectbox("Type", list(KINDS), format_func=KINDS.get, key="vr_k")
    sub = ss[(ss.vehicle == veh) & (ss.kind == kind)].sort_values("duration_min", ascending=False)
    sid = c3.selectbox("Session (longest first)", sub.session_id.tolist(), key="vr_s")
    ts = ui.bev_ts(kind, sid)
    tmax = float(ts.t_s.max() / 60)
    win = st.slider("Replay window (min)", 0.0, tmax, (0.0, tmax), 0.5, key="vr_w")
    w = ts[(ts.t_s / 60 >= win[0]) & (ts.t_s / 60 <= win[1])]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Speed (km/h) and BMS SOC (%)", "Pack power and BMS power limits (kW, positive = discharge)",
                                        "Pack temperature and ambient (°C)"))
    x = w.t_s / 60
    fig.add_trace(go.Scatter(x=x, y=w.speed_kph, name="Speed", line=dict(color=ui.PROV["Measured"], width=1.1)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=w.soc_bms_pct, name="SOC (BMS)", line=dict(color=ui.PROV["Reference"], width=2.6)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=w.v_pack_V * w.i_pack_A / 1000, name="Power", line=dict(color=ui.SERIES[0], width=1)), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=w.p_dis_lim_kW, name="Discharge limit", line=dict(color=ui.ALERT, dash="dot", width=1.6)), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=-w.p_regen_lim_kW, name="Regen limit", line=dict(color=ui.SERIES[1], dash="dot", width=1.6)), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=w.t_max_C, name="T max", line=dict(color=ui.ALERT, width=1.8)), 3, 1)
    fig.add_trace(go.Scatter(x=x, y=w.t_min_C, name="T min", line=dict(color=ui.SERIES[0], width=1.8)), 3, 1)
    fig.add_trace(go.Scatter(x=x, y=w.t_amb_C, name="Ambient", line=dict(color=ui.PROV["Reference"], dash="dot", width=1.6)), 3, 1)
    fig.update_xaxes(title_text="Time (min)", row=3, col=1)
    ui.chart(fig, f"{sid}", ("Measured", "Reference"), height=720, key="vr_fig")
    r = sub.set_index("session_id").loc[sid]
    dt = np.clip(np.diff(w.t_s.to_numpy(), prepend=w.t_s.iloc[0]), 0, 10)
    e_win = float(np.sum(w.v_pack_V * w.i_pack_A * dt) / 3.6e6)
    d_win = float(np.sum(w.speed_kph.fillna(0) * dt) / 3600)
    items = [("Window length", f"{(win[1] - win[0]):.0f}", "min"),
             ("SOC change (BMS)", f"{w.soc_bms_pct.iloc[-1] - w.soc_bms_pct.iloc[0]:+.1f}", "pts",
              f"{w.soc_bms_pct.iloc[0]:.1f} → {w.soc_bms_pct.iloc[-1]:.1f} %"),
             ("Net energy out of pack", f"{e_win:.2f}", "kWh")]
    items += ([("Distance", f"{d_win:.1f}", "km"), ("Consumption", ui.fmt(100 * e_win / d_win, 1) if d_win > 1 else "–", "kWh/100 km")]
              if kind == "drive" else
              [("Peak charging power", f"{-(w.v_pack_V * w.i_pack_A).min() / 1000:.0f}", "kW"),
               ("Pack temperature rise", f"{w.t_max_C.max() - w.t_max_C.iloc[0]:.1f}", "K")])
    ui.readouts(items, cols=5)
    st.download_button("Download this window (CSV)", w.to_csv(index=False).encode(), f"{sid}_window.csv", "text/csv",
                       icon=":material/download:")

with tab_soc:
    c1, c2, c3, c4 = st.columns([1, 1, 2.2, 1.2], vertical_alignment="bottom")
    veh = c1.selectbox("Vehicle", ["BEV1", "BEV2"], key="vs_v")
    kind = c2.selectbox("Type", list(KINDS), format_func=KINDS.get, key="vs_k")
    sub = ss[(ss.vehicle == veh) & (ss.kind == kind) & (ss.d_soc_pct.abs() > 5) & (ss.logged_fraction > 0.9)]
    sub = sub.sort_values("duration_min", ascending=False)
    if sub.empty:
        st.info("No well-logged session with more than 5 % SOC change for this selection.")
    else:
        sid = c3.selectbox("Session", sub.session_id.tolist(), key="vs_s")
        q0 = float(qv.get(veh, 150.0))
        Q = c4.number_input("Pack capacity (Ah)", 1.0, 400.0, float(np.round(q0, 1)), 1.0, key="vs_q",
                            help="Default: median |Ah| / |ΔSOC| over well-logged sessions with ≥ 20 % SOC change "
                                 "(a BMS-referenced proxy, not an independent capacity test).")
        ts = ui.bev_ts(kind, sid)
        tt = ts.t_s.to_numpy(float)
        cc = est.coulomb_counting(tt, ts.i_pack_A.to_numpy(float), Q, ts.soc_bms_pct.iloc[0] / 100)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.65, 0.35],
                            subplot_titles=("SOC (%)", "Coulomb counting − BMS (percentage points)"))
        fig.add_trace(go.Scatter(x=tt / 60, y=ts.soc_bms_pct, name="BMS-reported SOC", line=dict(color=ui.PROV["Reference"], width=3.2)), 1, 1)
        fig.add_trace(go.Scatter(x=tt / 60, y=100 * cc, name="Coulomb counting", line=dict(color=ui.PROV["Estimated"], width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=tt / 60, y=100 * cc - ts.soc_bms_pct, showlegend=False, line=dict(color=ui.PROV["Estimated"], width=1.6),
                                 fill="tozeroy", fillcolor="rgba(14,138,125,0.1)"), 2, 1)
        fig.update_xaxes(title_text="Time (min)", row=2, col=1)
        ui.chart(fig, f"{sid}: Coulomb counting vs the car's BMS", ("Estimated", "Reference"), height=520, key="vs_fig")
        m = ev.summary(ts.soc_bms_pct.to_numpy() / 100, cc, 100)
        ui.readouts([("Mean |difference|", f"{m['MAE']:.2f}", "pts"), ("Max |difference|", f"{m['MaxAbs']:.2f}", "pts"),
                     ("Final difference", f"{100*cc[-1] - ts.soc_bms_pct.iloc[-1]:+.2f}", "pts"),
                     ("Implied capacity BEV1", ui.fmt(qv.get("BEV1", np.nan), 1), "Ah"),
                     ("Implied capacity BEV2", ui.fmt(qv.get("BEV2", np.nan), 1), "Ah")], cols=5, accent=ui.PROV["Estimated"])
        ui.note("The BMS SOC is not ground truth, and the implied capacity inherits its errors. This is a consistency check "
                "of field data: a few-point gap over a long session is the scale of correction a production BMS applies on "
                "top of current integration. It is not an independent SOH measurement.", warn=True)

with tab_fleet:
    dr = ss[(ss.kind == "drive") & ss.kwh_per_100km.between(5, 60) & (ss.distance_km > 5)].copy()
    fig = px.scatter(dr, x="t_amb_mean_C", y="kwh_per_100km", color="vehicle", opacity=0.45, hover_name="session_id",
                     color_discrete_map={"BEV1": ui.SERIES[0], "BEV2": ui.SERIES[2]},
                     labels=dict(t_amb_mean_C="Mean ambient temperature (°C)", kwh_per_100km="Pack energy use (kWh/100 km)",
                                 vehicle="Vehicle"))
    fig.update_traces(marker=dict(size=6))
    bins = pd.cut(dr.t_amb_mean_C, bins=np.arange(-10, 41, 5))
    med = dr.groupby([bins, "vehicle"], observed=True).kwh_per_100km.agg(["median", "size"]).reset_index()
    med["mid"] = med.t_amb_mean_C.apply(lambda b: b.mid)
    for vname, col in (("BEV1", ui.SERIES[0]), ("BEV2", ui.SERIES[2])):
        mm = med[(med.vehicle == vname) & (med["size"] >= 5)]
        fig.add_trace(go.Scatter(x=mm.mid, y=mm["median"], mode="lines+markers", name=f"{vname} median per 5 °C",
                                 line=dict(color=col, width=3), marker=dict(size=8)))
    ui.chart(fig, f"Real energy consumption vs ambient temperature, {len(dr)} drives over 5 km", ("Measured",), height=500,
             key="vf_fig")
    cold = dr[dr.t_amb_mean_C < 10].kwh_per_100km.median()
    mild = dr[dr.t_amb_mean_C.between(15, 25)].kwh_per_100km.median()
    ui.readouts([("Median below 10 °C", ui.fmt(cold, 1), "kWh/100 km", f"{(dr.t_amb_mean_C < 10).sum()} drives"),
                 ("Median at 15–25 °C", ui.fmt(mild, 1), "kWh/100 km", f"{dr.t_amb_mean_C.between(15, 25).sum()} drives"),
                 ("Cold penalty", ui.fmt(100 * (cold / mild - 1), 0) if mild else "–", "%", "pack energy per km")],
                cols=3)
    st.caption("Energy = ∫V·I dt at the pack terminals (includes cabin and battery heating); distance = ∫speed dt. Bins with "
               "fewer than 5 drives are not drawn. Short trips weigh heavily, because heating is a fixed cost per trip.")

with tab_chem:
    slow = ss[(ss.kind == "slow_charge") & (ss.logged_fraction > 0.9)].sort_values("duration_min", ascending=False)
    fig = go.Figure()
    slopes = {}
    for vname, col in (("BEV1", ui.SERIES[0]), ("BEV2", ui.SERIES[2])):
        sl = []
        for j, sid in enumerate(slow[slow.vehicle == vname].session_id.head(3)):
            d = ui.bev_ts("slow_charge", sid)
            d = d[d.i_pack_A < -3]
            fig.add_trace(go.Scatter(x=d.soc_bms_pct, y=d.v_pack_V, mode="lines", name=f"{vname} {sid[5:15]}",
                                     line=dict(color=col, width=1.8), opacity=1 - 0.25 * j))
            mid = d[d.soc_bms_pct.between(30, 70)]
            if len(mid) > 50:
                sl.append(np.polyfit(mid.soc_bms_pct, mid.v_pack_V, 1)[0])
        slopes[vname] = float(np.median(sl)) if sl else np.nan
    fig.update_layout(xaxis_title="BMS SOC (%)", yaxis_title="Pack voltage (V)")
    ui.chart(fig, "Pack voltage vs SOC during AC charging (low current: quasi-OCV plus a small IR offset)",
             ("Measured", "Reference"), height=460, key="vc_fig")
    flat = min(slopes, key=lambda k: slopes[k] if np.isfinite(slopes[k]) else 1e9)
    ui.readouts([("BEV1 mid-SOC slope", ui.fmt(1000 * slopes.get("BEV1", np.nan), 0), "mV per %", "30–70 % SOC, pack level"),
                 ("BEV2 mid-SOC slope", ui.fmt(1000 * slopes.get("BEV2", np.nan), 0), "mV per %", "30–70 % SOC, pack level"),
                 ("BEV1 implied capacity", ui.fmt(qv.get("BEV1", np.nan), 0), "Ah"),
                 ("BEV2 implied capacity", ui.fmt(qv.get("BEV2", np.nan), 0), "Ah")], cols=4)
    ui.note(f"The dataset paper states that one vehicle has an LFP pack and the other an NCA pack, but the files do not say "
            f"which. Two independent signatures agree: {flat} has the flatter mid-SOC voltage plateau typical of LFP, and the "
            f"smaller implied capacity, consistent with the standard-range LFP pack. A flat plateau is also why voltage-based "
            f"(EKF) SOC estimation is much harder on LFP packs. This is an inference from data, labelled as such.")
ui.footer()
