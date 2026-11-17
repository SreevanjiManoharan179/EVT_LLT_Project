import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from g14lab import ecm, ui
from g14lab import estimators as est
from g14lab import evaluation as ev

ui.page_header("SOC estimation",
               "Compare Coulomb counting with three model-based filters on measured NASA discharges that were not used "
               "for calibration. Corrupt the current sensor, add voltage noise and start from a wrong SOC to see which "
               "estimator recovers.")
p, th, t = ui.twin()
dc = ui.discharge_cycles()
cands = ["B0005", "B0006", "B0007", "B0018", "B0036", "B0025", "B0026", "B0027", "B0028"]
QMODES = {"rf": "SOH model (Random Forest)", "twin": "Twin's fresh capacity", "oracle": "Measured capacity (oracle)"}
NAMES = {"cc": "Coulomb counting", "ekf": "EKF", "aekf": "Adaptive EKF", "ukf": "UKF"}
COLS = {"cc": ui.SERIES[2], "ekf": ui.PROV["Estimated"], "aekf": ui.SERIES[0], "ukf": ui.SERIES[3]}

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    cell = c1.selectbox("Cell", cands, index=0, key="s_cell",
                        help="B0025–B0028 use a 4 A square-wave load, the most dynamic profile in the dataset.")
    g = dc[(dc.cell_id == cell) & dc.usable_for_soh & ~dc.flag_capacity_mismatch]
    k = c2.selectbox("Discharge cycle", g.discharge_index.tolist(), index=min(60, len(g) - 1), key="s_k")
    soc0g = c3.slider("Initial SOC guess (%)", 50, 100, 80, key="s_soc0", help="The true initial SOC is 100 % (fully charged).")
    qmode = c4.selectbox("Capacity given to the estimators", list(QMODES), format_func=QMODES.get, key="s_q")
    c5, c6, c7, c8 = st.columns(4, gap="medium")
    gain = c5.slider("Current-sensor gain error (%)", -5.0, 5.0, 2.0, 0.5, key="s_gain")
    offs = c6.slider("Current-sensor offset (mA)", -100, 100, 20, 5, key="s_off")
    vnoise = c7.slider("Added voltage noise σ (mV)", 0, 30, 0, key="s_vn")
    rmeas = c8.select_slider("Measurement noise R (V²)", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3, key="s_r",
                             help="Initial R for the adaptive EKF; fixed R for EKF and UKF.")

c = ui.cycle(cell, k)
ref = ecm.soc_reference(c["t"], c["I"], c["Q"])
rng = np.random.default_rng(14)
Im = c["I"] * (1 + gain / 100) + offs / 1000
Vm = c["V"] + rng.normal(0, vnoise / 1000, len(c["V"]))

if qmode == "rf":
    Qe, how = ui.rf_capacity(cell, k)
    if not np.isfinite(Qe):
        Qe, how = p.Q_Ah, f"RF estimate unavailable ({how}); fell back to the twin capacity"
    qnote = f"RF-estimated capacity {Qe:.3f} Ah, {how}; measured {c['Q']:.3f} Ah"
elif qmode == "twin":
    Qe, qnote = p.Q_Ah, f"twin capacity {p.Q_Ah:.3f} Ah; measured {c['Q']:.3f} Ah"
else:
    Qe, qnote = c["Q"], f"measured capacity {c['Q']:.3f} Ah"

res = dict(cc=est.coulomb_counting(c["t"], Im, Qe, soc0g / 100))
e = est.ekf(p, c["t"], Im, Vm, soc0g / 100, Q_Ah=Qe, T_C=c["T"], r=rmeas)
ae = est.ekf(p, c["t"], Im, Vm, soc0g / 100, Q_Ah=Qe, T_C=c["T"], r=rmeas, adaptive=True)
u = est.ukf(p, c["t"], Im, Vm, soc0g / 100, Q_Ah=Qe, T_C=c["T"], r=rmeas)
res.update(ekf=e["soc"], aekf=ae["soc"], ukf=u["soc"])
show = st.multiselect("Estimators shown", list(NAMES), default=list(NAMES), format_func=NAMES.get, key="s_show")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.62, 0.38],
                    subplot_titles=("State of charge (%)", "Estimation error (percentage points)"))
x = c["t"] / 60
if "ekf" in show:
    fig.add_trace(go.Scatter(x=x, y=100 * (e["soc"] + 3 * e["sigma_soc"]), line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=100 * (e["soc"] - 3 * e["sigma_soc"]), line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(14,138,125,0.13)", name="EKF ±3σ"), 1, 1)
fig.add_trace(go.Scatter(x=x, y=100 * ref, name="Reference (Ah-integrated)", line=dict(color=ui.PROV["Reference"], width=3.2)), 1, 1)
for kname in show:
    dash = "dash" if kname == "ukf" else None
    fig.add_trace(go.Scatter(x=x, y=100 * res[kname], name=NAMES[kname], legendgroup=kname,
                             line=dict(color=COLS[kname], width=2, dash=dash)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=100 * (res[kname] - ref), showlegend=False, legendgroup=kname,
                             line=dict(color=COLS[kname], width=1.6, dash=dash)), 2, 1)
fig.add_hrect(y0=-2, y1=2, row=2, col=1, fillcolor="rgba(30,42,56,0.07)", line_width=0)
fig.update_xaxes(title_text="Time (min)", row=2, col=1)
ui.chart(fig, f"{cell} discharge {k}: estimated vs reference SOC", ("Measured", "Estimated", "Reference"), height=620,
         key="s_main", sub=f"Capacity used: {qnote}. Grey band = ±2 points. The reference is the Ah-integral of the "
                           "uncorrupted measured current; it is never fed back into the estimators.")

rows = []
for kname, s in res.items():
    m = ev.summary(ref, s, 100)
    ct = est.convergence_time(c["t"], s - ref)
    rows.append(dict(est=NAMES[kname], mae=m["MAE"], rmse=m["RMSE"], mx=m["MaxAbs"], fin=100 * (s[-1] - ref[-1]),
                     conv=f"{ct / 60:.1f}" if np.isfinite(ct) else "never"))
mt = pd.DataFrame(rows)
best = mt.iloc[mt.mae.idxmin()]
ui.readouts([("Best estimator (MAE)", best.est, "", f"{best.mae:.2f} points"),
             ("Coulomb counting MAE", f"{mt.mae.iloc[0]:.2f}", "pts"),
             ("EKF MAE", f"{mt.mae.iloc[1]:.2f}", "pts"), ("Adaptive EKF MAE", f"{mt.mae.iloc[2]:.2f}", "pts"),
             ("UKF MAE", f"{mt.mae.iloc[3]:.2f}", "pts")], cols=5, accent=ui.PROV["Estimated"])
ui.table(mt, {"est": ("Estimator",), "mae": ("MAE (pts)", "%.2f"), "rmse": ("RMSE (pts)", "%.2f"),
              "mx": ("Max |error| (pts)", "%.2f"), "fin": ("Final error (pts)", "%.2f"),
              "conv": ("Settles within ±2 pts after (min)", None, "Time after which the error stays inside ±2 points.")})

a, b = st.columns(2, gap="large")
with a:
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=x, y=c["V"], name="Measured", line=dict(color=ui.PROV["Measured"], width=2.2)))
    f2.add_trace(go.Scatter(x=x, y=e["v_pred"], name="EKF prediction", line=dict(color=ui.PROV["Estimated"], dash="dash", width=2)))
    f2.update_layout(xaxis_title="Time (min)", yaxis_title="Terminal voltage (V)")
    ui.chart(f2, "Voltage the EKF expects vs measured", ("Measured", "Estimated"), height=340, key="s_v")
with b:
    f3 = make_subplots(specs=[[{"secondary_y": True}]])
    f3.add_trace(go.Scatter(x=x, y=1000 * e["innovation"], name="EKF innovation", line=dict(color=ui.PROV["Estimated"], width=1.2)))
    f3.add_trace(go.Scatter(x=x, y=np.sqrt(ae["r_trace"]) * 1000, name="adaptive EKF √R", line=dict(color=ui.SERIES[0], width=2)),
                 secondary_y=True)
    f3.update_xaxes(title_text="Time (min)")
    f3.update_yaxes(title_text="Innovation (mV)", secondary_y=False)
    f3.update_yaxes(title_text="√R (mV)", secondary_y=True, showgrid=False)
    ui.chart(f3, "Innovation and adaptive noise level", ("Estimated",), height=340, key="s_inn")

with st.expander("Estimator equations"):
    st.latex(r"x_k = \begin{bmatrix} z_k \\ V_{1,k} \end{bmatrix},\quad "
             r"x_{k+1} = \begin{bmatrix} z_k - \dfrac{I_k\,\Delta t}{3600\,Q} \\ a\,V_{1,k} + R_1(1-a)\,I_k \end{bmatrix} + w_k")
    st.latex(r"y_k = \mathrm{OCV}(z_k) - I_k R_0 - V_{1,k} + v_k,\qquad H_k = \begin{bmatrix} \partial\mathrm{OCV}/\partial z & -1 \end{bmatrix}")
    st.latex(r"K_k = P_k^- H_k^\top (H_k P_k^- H_k^\top + R_k)^{-1},\qquad \hat x_k = \hat x_k^- + K_k\,(y_k - \hat y_k^-)")
    st.latex(r"\text{Adaptive EKF: } R_k = \max\!\Big(\tfrac{1}{N}\textstyle\sum_{j=k-N+1}^{k} e_j^2 - H_k P_k^- H_k^\top,\ R_{\min}\Big),\ N = 25")
    st.caption("P0 = diag(0.1, 0.01), Q = diag(1e-7, 1e-6)·Δt. The UKF propagates 5 scaled sigma points (α = 0.3, β = 2, "
               "κ = 0) through the same model, so it needs no Jacobian of the OCV curve.")

ui.note("Coulomb counting integrates the corrupted current from a wrong start and never corrects itself. The filters use "
        "the twin's OCV curve to pull the estimate towards the voltage evidence; their accuracy is limited by the model "
        "voltage error times the OCV slope, which is why the error grows on the flat mid-SOC plateau. The capacity selector "
        "is where SOH meets SOC: the Random Forest estimate (for reference cells, a prediction made with the cell held out) "
        "replaces an oracle capacity.")
bm = ui.metrics_csv("soc_benchmark.csv")
if len(bm):
    ui.section("Benchmark over 10 held-out cycles", "Fixed disturbance: 80 % initial guess, +2 % gain error, +20 mA offset.")
    sm = bm.groupby(["capacity_used", "estimator"])[["MAE_pct", "RMSE_pct", "final_error_pct"]].median().reset_index()
    sm["estimator"] = sm.estimator.map({"CC": "Coulomb counting", "EKF": "EKF", "AEKF": "Adaptive EKF", "UKF": "UKF"})
    ui.table(sm, {"capacity_used": ("Capacity given",), "estimator": ("Estimator",), "MAE_pct": ("Median MAE (pts)", "%.2f"),
                  "RMSE_pct": ("Median RMSE (pts)", "%.2f"), "final_error_pct": ("Median final error (pts)", "%.2f")})
cfg = dict(cell=cell, cycle=int(k), soc0_guess_pct=soc0g, gain_err_pct=gain, offset_mA=offs, vnoise_mV=vnoise,
           R=rmeas, capacity_mode=QMODES[qmode], capacity_used_Ah=Qe, twin=t["label"])
out = pd.DataFrame(dict(t_s=c["t"], I_meas=c["I"], I_corrupt=Im, V=Vm, soc_ref=ref, soc_cc=res["cc"], soc_ekf=e["soc"],
                        soc_ekf_sigma=e["sigma_soc"], soc_aekf=ae["soc"], soc_ukf=u["soc"]))
ui.export_bar("SOC estimation", cfg, mt.to_dict(orient="records"), out, fig, key="soc")
ui.footer()
