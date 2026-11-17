import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from g14lab import calibration, paths, ui

ui.page_header("Validation & export",
               "Every number on this page is computed by this application from the datasets; nothing is quoted from "
               "papers. Section 1 re-validates whichever twin is active right now.")
p, th, t = ui.twin()

# ------------------------------------------------------------------------------- 1. twin
ui.section("1. Twin: measured vs simulated, on held-out cycles and cells", f"Active twin: {t['label']}.")


@st.cache_data(show_spinner=False)
def live_validation(h, ecm_d, th_d):
    from g14lab.ecm import ECMParams
    from g14lab.thermal import ThermalParams
    return calibration.evaluate_twin(ECMParams.from_dict(ecm_d), ThermalParams(**th_d))


with st.spinner("Simulating 23 measured cycles with the active twin…"):
    tv = live_validation(ui.twin_hash(), t["ecm"], t["thermal"])
tv["case"] = tv.cell_id + " #" + tv.discharge_index.astype(str)


def domain(r):
    if r.cell_id == "B0005":
        return "same cell, across life"
    if r.ambient_C <= 10:
        return "other cells, 4 °C"
    if r.ambient_C >= 40:
        return "other cells, 43 °C"
    if r.mean_current_A > 3 or r.cell_id in ("B0025", "B0026", "B0027", "B0028"):
        return "other cells, 4 A or pulsed"
    return "other cells, 24 °C / 2 A"


tv["domain"] = tv.apply(domain, axis=1)
order = ["same cell, across life", "other cells, 24 °C / 2 A", "other cells, 4 A or pulsed", "other cells, 43 °C", "other cells, 4 °C"]
dcol = dict(zip(order, [ui.PROV["Simulated"], ui.SERIES[1], ui.SERIES[2], ui.ALERT, ui.SERIES[3]]))
fig = go.Figure()
for dname in order:
    d = tv[tv.domain == dname]
    fig.add_trace(go.Bar(x=d.case, y=d.V_RMSE_mV, name=dname, marker_color=dcol[dname]))
fig.update_layout(xaxis_title=None, yaxis_title="Voltage RMSE (mV)", bargap=0.25, xaxis=dict(tickangle=-40))
ui.chart(fig, "Voltage error by validation domain", ("Measured", "Simulated"), height=420, key="v_tv")
summ = tv.groupby("domain").agg(n=("case", "size"), v=("V_RMSE_mV", "median"), tt=("T_RMSE_C", "median")).reindex(order).reset_index()
ui.table(summ, {"domain": ("Validation domain",), "n": ("Cycles", "%d"), "v": ("Median voltage RMSE (mV)", "%.1f"),
                "tt": ("Median temperature RMSE (K)", "%.2f")})
with st.expander("All 23 validation cycles"):
    ui.table(tv, {"case": ("Cycle",), "domain": ("Domain",), "ambient_C": ("Ambient (°C)", "%.0f"),
                  "mean_current_A": ("Load (A)", "%.1f"), "capacity_Ah": ("Capacity (Ah)", "%.3f"),
                  "V_RMSE_mV": ("V RMSE (mV)", "%.1f"), "V_MAE_mV": ("V MAE (mV)", "%.1f"), "T_RMSE_C": ("T RMSE (K)", "%.2f"),
                  "Tmax_meas_C": ("Peak T measured (°C)", "%.1f"), "Tmax_sim_C": ("Peak T simulated (°C)", "%.1f")})


@st.cache_data(show_spinner=False)
def live_charge_validation(h, ecm_d, th_d):
    from g14lab.ecm import ECMParams
    from g14lab.thermal import ThermalParams
    return calibration.evaluate_charges(ECMParams.from_dict(ecm_d), ThermalParams(**th_d))


with st.spinner("Replaying 10 measured CC-CV charges and simulating the charging protocol…"):
    cv = live_charge_validation(ui.twin_hash(), t["ecm"], t["thermal"])
cv["case"] = cv.cell_id + " #" + cv.discharge_index.astype(str)
a, b = st.columns([1.5, 1], gap="large", vertical_alignment="top")
with a:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cv.case, y=cv.t_cc_meas_min, name="measured", marker_color=ui.PROV["Measured"]))
    fig.add_trace(go.Bar(x=cv.case, y=cv.t_cc_sim_aged_min, name="twin, growth-law resistance", marker_color=ui.PROV["Simulated"]))
    fig.update_layout(barmode="group", bargap=0.28, yaxis_title="CC phase to 4.2 V (min)", xaxis=dict(tickangle=-40))
    ui.chart(fig, "Charging: time to reach 4.2 V at 1.5 A", ("Measured", "Simulated"), height=380, key="v_cc",
             sub="Case = the CC-CV charge after that discharge. The twin starts at SOC 0 with the cycle's capacity and ambient.")
with b:
    err = (cv.t_cc_sim_aged_min - cv.t_cc_meas_min)
    early = cv.soh_pct >= 85
    ui.readouts([("CC-phase error, SOH ≥ 85 %", f"{err[early].abs().median():.1f}", "min",
                  f"median of {int(early.sum())} charges"),
                 ("CC-phase error, aged or 4 °C", f"{err[~early].abs().median():.1f}", "min",
                  f"median of {int((~early).sum())}; the twin overestimates"),
                 ("Charge voltage RMSE", f"{cv.V_RMSE_mV.median():.0f}", "mV", "measured current replayed"),
                 ("Charge temperature RMSE", f"{cv.T_RMSE_C.median():.1f}", "K", "measured current replayed")],
                cols=2, accent=ui.PROV["Simulated"])
with st.expander("All 10 charge validation cases"):
    ui.table(cv, {"case": ("After discharge",), "charge_index": ("Charge", "%d"), "ambient_C": ("Ambient (°C)", "%.0f"),
                  "soh_pct": ("SOH (%)", "%.1f"), "V_RMSE_mV": ("V RMSE (mV)", "%.1f"), "T_RMSE_C": ("T RMSE (K)", "%.2f"),
                  "t_cc_meas_min": ("CC measured (min)", "%.1f"), "t_cc_sim_min": ("CC twin, fresh R (min)", "%.1f"),
                  "t_cc_sim_aged_min": ("CC twin, growth-law R (min)", "%.1f"), "q_cc_meas_Ah": ("CC charge measured (Ah)", "%.3f"),
                  "q_cc_sim_aged_Ah": ("CC charge twin (Ah)", "%.3f")})
sv = ui.metrics_csv("shadow_validation.csv")
if len(sv):
    parts = []
    for stage in ("early (> 85 %)", "mid (75–85 %)", "late (SOH < 75 %)"):
        g = sv[sv.life_stage == stage]
        if len(g):
            parts.append(f"{stage}: reference {g.fresh_mV.median():.0f}, shadow {g.shadow_mV.median():.0f}, "
                         f"growth law {g.growth_mV.median():.0f}")
    st.caption("Digital shadow, one-cycle-ahead (median voltage RMSE, mV): " + "; ".join(parts) +
               ". Details on the calibration page.")

# ------------------------------------------------------------------------------- 2. SOC
ui.section("2. SOC estimators", "10 held-out cycles; initial guess 80 % (true 100 %), current gain +2 %, offset +20 mA.")
sb = ui.metrics_csv("soc_benchmark.csv")
if len(sb):
    names = {"CC": "Coulomb counting", "EKF": "EKF", "AEKF": "Adaptive EKF", "UKF": "UKF"}
    caps = {"twin capacity (fresh)": ui.PROV["Simulated"], "RF-predicted capacity": ui.PROV["Predicted"],
            "measured capacity (oracle)": ui.PROV["Measured"]}
    m = sb.groupby(["estimator", "capacity_used"]).MAE_pct.median().unstack().reindex(["CC", "EKF", "AEKF", "UKF"])
    fig = go.Figure()
    for cap, col in caps.items():
        if cap in m.columns:
            fig.add_trace(go.Bar(x=[names[e] for e in m.index], y=m[cap], name=cap, marker_color=col))
    fig.update_layout(yaxis_title="Median MAE (percentage points)", barmode="group", bargap=0.3)
    ui.chart(fig, "SOC error by estimator and by the capacity it is given", ("Estimated",), height=360, key="v_soc")
    s2 = sb.groupby(["capacity_used", "estimator"])[["MAE_pct", "RMSE_pct", "final_error_pct"]].median().reset_index()
    s2["estimator"] = s2.estimator.map(names)
    ui.table(s2, {"capacity_used": ("Capacity given",), "estimator": ("Estimator",), "MAE_pct": ("Median MAE (pts)", "%.2f"),
                  "RMSE_pct": ("Median RMSE (pts)", "%.2f"), "final_error_pct": ("Median final error (pts)", "%.2f")})

# ------------------------------------------------------------------------------- 3. SOH
ui.section("3. SOH models", "Leave-one-cell-out (honest) vs random K-fold (leaky); reference protocol cells.")
bm = ui.metrics_csv("soh_benchmark.csv")
if len(bm):
    r = bm[bm.cell_set.str.startswith("Reference")].copy()
    r["label"] = r.model + " · " + r.feature_set.str.replace(" (on-board measurable)", "", regex=False)
    r = r.sort_values("LOCO_RMSE_mAh", ascending=False)
    fig = go.Figure()
    for _, x in r.iterrows():
        fig.add_trace(go.Scatter(x=[x.randomKFold_RMSE_mAh, x.LOCO_RMSE_mAh], y=[x.label, x.label], mode="lines",
                                 line=dict(color="#C9D1D9", width=4), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=r.randomKFold_RMSE_mAh, y=r.label, mode="markers", name="random K-fold (leaky)",
                             marker=dict(color=ui.PROV["Reference"], size=11)))
    fig.add_trace(go.Scatter(x=r.LOCO_RMSE_mAh, y=r.label, mode="markers", name="leave-one-cell-out (honest)",
                             marker=dict(color=ui.PROV["Predicted"], size=11)))
    fig.update_layout(xaxis_title="RMSE (mAh)", yaxis=dict(tickfont=dict(size=11)))
    ui.chart(fig, "Capacity error, reference cells B0005/6/7/18", ("Predicted",), height=460, key="v_soh")
    best = r.sort_values("LOCO_RMSE_mAh").iloc[0]
    rf = r[r.model == "Random Forest"].sort_values("LOCO_RMSE_mAh").iloc[0]
    ui.readouts([("Best model, held-out cells", f"{best.LOCO_RMSE_mAh:.1f}", "mAh", f"{best.model}; {best.feature_set}"),
                 ("Random Forest baseline", f"{rf.LOCO_RMSE_mAh:.1f}", "mAh", f"R² {rf.LOCO_R2:.3f}, {rf.LOCO_RMSE_SOH_pct:.2f} % SOH"),
                 ("Same RF, leaky K-fold", f"{rf.randomKFold_RMSE_mAh:.1f}", "mAh", "what a random split would claim"),
                 ("Conformal coverage (RF)", f"{100*rf.conformal_coverage:.0f}", "%", "nominal 90 %"),
                 ("RF tree-interval coverage", f"{100*rf.RF_interval_coverage:.0f}", "%", "nominal 90 %")],
                cols=5, accent=ui.PROV["Predicted"])

# ------------------------------------------------------------------------------- 4. laws
ui.section("4. Identified physical laws")
g = p.extra.get("resistance_growth", {})
ui.sheet([
    ("Arrhenius activation energy", f"{p.Ea_R0/1000:.1f} kJ/mol",
     f"ln(R0+R1) vs 1/T and ln I, {p.extra.get('arrhenius_n', '–')} cycles from {p.extra.get('arrhenius_cells', '–')} cells, "
     f"R² = {p.extra.get('arrhenius_fit_R2', float('nan')):.3f}"),
    ("Current exponent β", f"{p.extra.get('current_exponent_beta', float('nan')):.3f}", "weak: resistance nearly current-independent 1–4 A"),
    ("Resistance growth with ageing", f"k = {g.get('k_per_fade', float('nan')):.2f}",
     (f"R/R_early = 1 + k·(SOH_early − SOH), each cell normalised to its first cycles; R² = {g.get('R2', float('nan')):.2f}, "
      f"n = {g.get('n', '–')}") if g.get("form") == "relative" else
     f"R/R_ref = {g.get('intercept', float('nan')):.2f} + k·(1 − SOH), R² = {g.get('R2', float('nan')):.2f}, n = {g.get('n', '–')}"),
    ("Thermal time constant", f"{th.tau_s/60:.0f} min", f"C_th {th.C_th:.0f} J/K, hA {th.hA*1000:.1f} mW/K"),
    ("Entropic coefficient dU/dT", f"{th.dUdT*1000:+.2f} mV/K",
     "effective, SOC-averaged; identifiable because reversible heat changes sign with the current direction" +
     ("; negligible here" if 0 < abs(th.dUdT) < 1e-4 else "")),
])

ui.section("Limitations")
st.markdown("""
- The 1-RC model is identified from 2.5–19 s sampled data, so the split between R0 and R1 is only weakly identifiable (their sum is well determined); R0 is bounded below by the cell's EIS ohmic resistance. Resistances are constant over SOC, so the large low-SOC discharge overpotential is averaged into R1.
- Hysteresis is not modelled; the charge–discharge asymmetry is carried by the direction-dependent polarisation resistance. In CV charging the twin's current settles near 35 mA instead of tapering to 20 mA (its OCV saturates a few mV below 4.2 V); the simulator ends the CV step there and says so. The OCV curve is identified jointly with the resistances, not from a slow-rate OCV test; SOC 0 % is the 2 A cut-off point. With ageing the OCV-vs-SOC shape changes, which a fixed-OCV twin cannot follow: this dominates late-life voltage error and makes the twin overestimate the CC charging time of aged cells.
- Lumped thermal parameters and the entropic coefficient are effective values (cell plus fixture) identified at 24 °C on 2 A discharges and 1.5 A charges; errors grow at 4 A and 43 °C. This is not a thermal-runaway model.
- Results below 4 °C, above 44 °C or above 2.2C are extrapolations; the app flags them.
- SOH models are trained on four reference cells; leave-one-cell-out with four cells is a small validation set, so treat errors as indicative. Conformal coverage below 90 % reflects that cells are not exchangeable.
- Tesla SOC is BMS-reported; the implied pack capacity is a proxy. Cell parameters are never transferred to the packs, and the LFP and NCA vehicles are never merged.
- The twin is offline and data-calibrated, with replay of recorded telemetry; it is not synchronised with a live battery.
""")

# ------------------------------------------------------------------------------- export
ui.section("Experiment log and export")
log = st.session_state.get("log", [])
if not log:
    st.info("No experiments logged yet. Use **Add to experiment log** on the bench, SOC or SOH pages.")
else:
    ui.table(pd.DataFrame([dict(id=r_["id"], kind=r_["kind"], time=r_["time"],
                                config=json.dumps(r_["config"], default=str)[:140]) for r_ in log]),
             {"id": ("#", "%d"), "kind": ("Experiment",), "time": ("Logged at",), "config": ("Configuration (excerpt)",)})
    c1, c2, _ = st.columns([1.6, 1.4, 1.5], vertical_alignment="center")
    c1.download_button("Download all experiments (ZIP)", ui.bundle(log), "group14_experiments.zip", "application/zip",
                       type="primary", icon=":material/folder_zip:", width="stretch")
    if c2.button("Save bundle to outputs/experiment_results", icon=":material/save:", width="stretch"):
        out = paths.EXPERIMENTS / f"experiments_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip"
        out.write_bytes(ui.bundle(log))
        st.toast(f"Saved {out.relative_to(paths.ROOT)}")
c1, c2, _ = st.columns([1.6, 1.4, 1.5])
c1.download_button("Download active twin parameters (JSON)", json.dumps(t, indent=2, default=str).encode(), "active_twin.json",
                   "application/json", icon=":material/data_object:", width="stretch")
c2.download_button("Download validation tables (CSV)", pd.concat([tv.assign(kind="discharge"), cv.assign(kind="charge")],
                                                                 ignore_index=True).to_csv(index=False).encode(),
                   "twin_validation_active.csv", "text/csv", icon=":material/download:", width="stretch")
ui.footer()
