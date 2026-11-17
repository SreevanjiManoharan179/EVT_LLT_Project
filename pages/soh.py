import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from g14lab import soh, ui
from g14lab.ica import ica_curve

ui.page_header("SOH & ageing",
               "Capacity fade measured across 34 cells and three ambient temperatures, incremental-capacity analysis of "
               "the ageing mechanism, machine-learning SOH estimation tested on cells held out of training, and "
               "forecasting of remaining capacity to end of life.")
dc = ui.discharge_cycles()
tab_fade, tab_ica, tab_ml, tab_fc = st.tabs(["Capacity fade (measured)", "Incremental capacity (dQ/dV)",
                                             "ML SOH estimation", "Fade forecast & RUL"])

# ------------------------------------------------------------------------------------------- fade
with tab_fade:
    groups = {"Reference: 24 °C, 2 A (B0005/6/7/18)": ["B0005", "B0006", "B0007", "B0018"],
              "Load: 24 °C at 2 A vs 4 A (B0036, B0033, B0034)": ["B0036", "B0033", "B0034"],
              "Hot: 43 °C, 4 A (B0029–32)": ["B0029", "B0030", "B0031", "B0032"],
              "Cold: 4 °C, 1 A (B0045–48)": ["B0045", "B0046", "B0047", "B0048"],
              "Cold: 4 °C, 2 A (B0053–56)": ["B0053", "B0054", "B0055", "B0056"],
              "Pulsed: 4 A square wave, 24 °C (B0025–28)": ["B0025", "B0026", "B0027", "B0028"]}
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    sel = c1.multiselect("Cell groups", list(groups), default=[list(groups)[0], list(groups)[3]], key="f_g")
    yv = c2.selectbox("Show", ["SOH (% of 2 Ah)", "Capacity (Ah)"], key="f_y")
    fig = go.Figure()
    i = 0
    for gname in sel:
        for cid in groups[gname]:
            g = dc[(dc.cell_id == cid) & dc.usable_for_soh]
            fig.add_trace(go.Scatter(x=g.discharge_index, y=g.soh_pct if yv.startswith("SOH") else g.capacity_Ah,
                                     mode="lines+markers", marker=dict(size=3), name=f"{cid}, {gname.split(':')[0].lower()}",
                                     line=dict(color=ui.SERIES[i % len(ui.SERIES)], width=1.6)))
            i += 1
    if yv.startswith("SOH"):
        fig.add_hline(y=70, line=dict(dash="dash", color=ui.ALERT, width=1.2), annotation_text="EOL 70 %",
                      annotation_font=dict(color=ui.ALERT))
    fig.update_layout(xaxis_title="Discharge cycle", yaxis_title=yv)
    ui.chart(fig, "Capacity fade under different operating conditions", ("Measured",), height=470, key="f_main")
    ui.note("Capacity measured at 4 °C or 4 A is the charge <i>delivered at that condition</i>: it is lower partly because "
            "resistance rises in the cold, not only because of ageing. A true SOH needs a reference test at fixed "
            "conditions, which only the 24 °C / 2 A group provides; this is why the main ML model trains on that group and "
            "why the cross-condition experiment is harder.", warn=True)
    first = dc[dc.usable_for_soh].groupby("cell_id").head(3).groupby("cell_id").agg(
        ambient_C=("ambient_C", "first"), current_A=("mean_current_A", "mean"), early_capacity_Ah=("capacity_Ah", "mean"),
        T_rise_C=("T_rise_C", "mean")).reset_index()
    fig2 = px.scatter(first, x="ambient_C", y="early_capacity_Ah", color="current_A", size="T_rise_C", hover_name="cell_id",
                      color_continuous_scale="Viridis", range_color=(1, 4),
                      labels=dict(ambient_C="Ambient temperature (°C)", early_capacity_Ah="Early-life delivered capacity (Ah)",
                                  current_A="Load (A)", T_rise_C="Temperature rise (K)"))
    fig2.update_layout(coloraxis_colorbar=dict(thickness=12))
    ui.chart(fig2, "Delivered capacity vs ambient temperature and load (first 3 cycles per cell, size = self-heating)",
             ("Measured",), height=400, key="f_amb", legend="none")

# ------------------------------------------------------------------------------------------- ICA
with tab_ica:
    c1, c2, c3 = st.columns([1, 2.5, 1], vertical_alignment="bottom")
    cell = c1.selectbox("Cell", ["B0005", "B0006", "B0007", "B0018", "B0036", "B0033", "B0034"], key="i_cell")
    cg = ui.charge_cycles()
    cg = cg[(cg.cell_id == cell) & cg["valid"]]
    ks = cg.charge_index.tolist()
    picks = c2.multiselect("Charges to compare", ks, default=[ks[min(2, len(ks) - 1)], ks[len(ks) // 3], ks[2 * len(ks) // 3], ks[-1]],
                           key="i_picks")
    ts = ui.charge_ts(cell, tuple(picks)) if picks else None
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        fig = go.Figure()
        cols = px.colors.sample_colorscale("Viridis", [i / max(1, len(picks) - 1) for i in range(len(picks))]) if picks else []
        for j, kk in enumerate(sorted(picks)):
            s = ts[ts.charge_index == kk]
            r = ica_curve(s.t_s.to_numpy(float), s.V.to_numpy(float), -s.I.to_numpy(float))
            if r is None:
                continue
            v, dq = r
            fig.add_trace(go.Scatter(x=v, y=dq, name=f"charge {kk}", line=dict(color=cols[j], width=2.2)))
        fig.update_layout(xaxis_title="Terminal voltage (V)", yaxis_title="dQ/dV (Ah/V)")
        ui.chart(fig, f"{cell}: incremental-capacity curves of the CC charge", ("Measured",), height=420, key="i_curves",
                 sub="Savitzky–Golay smoothed on a 5 mV grid. Darker = earlier in life.")
    with b:
        d = dc[(dc.cell_id == cell) & dc.usable_for_soh & dc.ic_peak_AhV.between(0.5, 15)].sort_values("discharge_index")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=d.discharge_index, y=d.ic_peak_AhV, name="ICA peak height", mode="markers",
                                 marker=dict(color=ui.SERIES[3], size=5)))
        fig.add_trace(go.Scatter(x=d.discharge_index, y=d.capacity_Ah, name="discharge capacity",
                                 line=dict(color=ui.PROV["Measured"], width=2)), secondary_y=True)
        fig.update_xaxes(title_text="Discharge cycle")
        fig.update_yaxes(title_text="Peak height (Ah/V)", secondary_y=False)
        fig.update_yaxes(title_text="Capacity (Ah)", secondary_y=True, showgrid=False)
        ui.chart(fig, "ICA peak height tracks capacity", ("Measured",), height=420, key="i_feat")
    if len(d) > 5:
        rho = d[["ic_peak_AhV", "ic_peak_V", "ic_area_Ah", "capacity_Ah"]].corr(method="spearman")["capacity_Ah"]
        ui.readouts([("ρ(peak height, capacity)", f"{rho.ic_peak_AhV:.2f}", "", "Spearman"),
                     ("ρ(peak voltage, capacity)", f"{rho.ic_peak_V:.2f}", "", "Spearman"),
                     ("ρ(window charge, capacity)", f"{rho.ic_area_Ah:.2f}", "", "3.80–4.10 V"),
                     ("Peak shift over life", f"{1000*(d.ic_peak_V.iloc[-5:].median() - d.ic_peak_V.iloc[:5].median()):+.0f}",
                      "mV", "last 5 vs first 5 cycles")], cols=4, accent=ui.SERIES[3])
    ui.note("dQ/dV turns the voltage plateaus of the electrodes into peaks. Ageing lowers the main peak (loss of lithium "
            "inventory and active material) and shifts it to higher voltage (rising polarisation). Peak height, peak "
            "voltage and the charge inside 3.80–4.10 V are measured on the charge before each discharge, so they are "
            "available on board before the discharge happens. They form one of the indicator sets on the next tab.")

# ------------------------------------------------------------------------------------------- ML
with tab_ml:
    preds = ui.metrics_parquet("soh_loco_predictions.parquet")
    bm = ui.metrics_csv("soh_benchmark.csv")
    if preds.empty or bm.empty:
        st.warning("Run `python -m g14lab.calibration` to build the SOH benchmark.")
    else:
        c1, c2, c3 = st.columns(3, gap="medium")
        fs = c1.selectbox("Health indicators", list(soh.FEATURE_SETS), index=list(soh.FEATURE_SETS).index(soh.DEFAULT_FEATURE_SET),
                          key="m_fs")
        cs = c2.selectbox("Cells", ["Reference protocol (B0005/6/7/18)", "All usable cells (cross-condition)"], key="m_cs")
        model = c3.selectbox("Model", soh.MODELS, index=1, key="m_model")
        pr = preds[(preds.feature_set == fs) & (preds.cell_set == cs) & (preds.model == model)]
        row = bm[(bm.feature_set == fs) & (bm.cell_set == cs) & (bm.model == model)].iloc[0]
        ui.readouts([("Leave-one-cell-out RMSE", f"{row.LOCO_RMSE_mAh:.0f}", "mAh", f"{row.LOCO_RMSE_SOH_pct:.2f} % SOH"),
                     ("Leave-one-cell-out R²", f"{row.LOCO_R2:.3f}", ""),
                     ("Leaky random K-fold RMSE", f"{row.randomKFold_RMSE_mAh:.0f}", "mAh",
                      f"{row.LOCO_RMSE_mAh / row.randomKFold_RMSE_mAh:.1f}× too optimistic"),
                     ("Conformal 90 % interval coverage", f"{100*row.conformal_coverage:.0f}", "%",
                      f"half-width {row.conformal_halfwidth_mAh:.0f} mAh"),
                     ("Cells / cycles", f"{row.n_cells} / {row.n_rows}", "")], cols=5, accent=ui.PROV["Predicted"])
        a, b = st.columns(2, gap="large")
        with a:
            fig = px.scatter(pr, x="capacity_Ah", y="pred_Ah", color="cell_id", opacity=0.75,
                             labels=dict(capacity_Ah="Measured capacity (Ah)", pred_Ah="Predicted capacity (Ah)", cell_id="Held-out cell"))
            lo, hi = pr.capacity_Ah.min(), pr.capacity_Ah.max()
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=ui.PROV["Reference"], dash="dash"),
                                     name="ideal"))
            ui.chart(fig, "Held-out predictions vs measurement", ("Measured", "Predicted"), height=420, key="m_sc")
        with b:
            cell = st.selectbox("Held-out cell", sorted(pr.cell_id.unique()), key="m_cell")
            q = pr[pr.cell_id == cell].sort_values("discharge_index")
            band = st.radio("Interval", ["cross-conformal 90 %", "RF tree 5–95 %"] if model == "Random Forest" else ["cross-conformal 90 %"],
                            horizontal=True, key="m_band")
            lo_c, hi_c = ("conf_lo_Ah", "conf_hi_Ah") if band.startswith("cross") else ("lo_Ah", "hi_Ah")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=q.discharge_index, y=q[hi_c], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=q.discharge_index, y=q[lo_c], line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(196,106,18,0.17)", name=band))
            fig.add_trace(go.Scatter(x=q.discharge_index, y=q.capacity_Ah, name="Measured", line=dict(color=ui.PROV["Measured"], width=2)))
            fig.add_trace(go.Scatter(x=q.discharge_index, y=q.pred_Ah, name=f"{model} (cell unseen)",
                                     line=dict(color=ui.PROV["Predicted"], width=2)))
            fig.update_layout(xaxis_title="Discharge cycle", yaxis_title="Capacity (Ah)")
            cov = ((q.capacity_Ah >= q[lo_c]) & (q.capacity_Ah <= q[hi_c])).mean()
            ui.chart(fig, f"{cell} predicted while held out of training", ("Measured", "Predicted"), height=360, key="m_ts",
                     sub=f"{100*cov:.0f} % of this cell's measurements fall inside the band.")

        ui.section("Why the validation split matters", "Every model and indicator set, leave-one-cell-out vs leaky random K-fold.")
        sub = bm[bm.cell_set == cs].copy()
        sub["label"] = sub.model + " · " + sub.feature_set.str.replace(" (on-board measurable)", "", regex=False)
        sub = sub.sort_values("LOCO_RMSE_mAh", ascending=False)
        fig = go.Figure()
        for _, r_ in sub.iterrows():
            fig.add_trace(go.Scatter(x=[r_.randomKFold_RMSE_mAh, r_.LOCO_RMSE_mAh], y=[r_.label, r_.label], mode="lines",
                                     line=dict(color="#C9D1D9", width=4), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=sub.randomKFold_RMSE_mAh, y=sub.label, mode="markers", name="random K-fold (leaky)",
                                 marker=dict(color=ui.PROV["Reference"], size=11)))
        fig.add_trace(go.Scatter(x=sub.LOCO_RMSE_mAh, y=sub.label, mode="markers", name="leave-one-cell-out (honest)",
                                 marker=dict(color=ui.PROV["Predicted"], size=11)))
        fig.update_layout(xaxis_title="RMSE (mAh)", yaxis=dict(tickfont=dict(size=11)))
        ui.chart(fig, f"Validation gap, {cs.split(' (')[0].lower()} cells", ("Predicted",), height=470, key="m_dumb")
        c1, _ = st.columns([1.3, 2])
        if c1.button("Compute permutation importance on the held-out cell", icon=":material/insights:", key="m_pi"):
            d_ = soh.dataset(dc, soh.FEATURE_SETS[fs], soh.REFERENCE_CELLS if cs.startswith("Ref") else None,
                             not cs.startswith("Ref"))
            cols_ = soh.FEATURE_SETS[fs] + ([] if cs.startswith("Ref") else soh.CONDITION_FEATURES)
            with st.spinner("Re-training without the held-out cell and permuting each indicator…"):
                st.session_state["m_imp"] = (soh.permutation_importance_loco(d_, cols_, model, cell), cell, model)
        if "m_imp" in st.session_state:
            imp, icell, imodel = st.session_state["m_imp"]
            fig = px.bar(imp, x="importance_mAh", y="feature", orientation="h", error_x="std_mAh",
                         color_discrete_sequence=[ui.PROV["Predicted"]])
            fig.update_layout(xaxis_title="Increase in RMSE when shuffled (mAh)", yaxis_title="", yaxis=dict(autorange="reversed"))
            ui.chart(fig, f"Permutation importance, {imodel}, {icell} held out", ("Predicted",), height=340, key="m_imp_fig")
        ui.note("Random K-fold mixes neighbouring cycles of the same cell into training and test sets, so it reports a fraction "
                "of the honest leave-one-cell-out error. Random-Forest tree quantiles cover far fewer than 90 % of held-out "
                "measurements; cross-conformal intervals, calibrated on the other cells' held-out errors, come much closer. "
                "Across operating conditions the linear model generalises best, because trees cannot extrapolate to unseen "
                "temperatures or loads.")
        ui.table(bm[bm.cell_set == cs], {"feature_set": ("Indicators",), "model": ("Model",), "n_cells": ("Cells", "%d"),
                                         "n_rows": ("Cycles", "%d"), "LOCO_MAE_mAh": ("LOCO MAE (mAh)", "%.1f"),
                                         "LOCO_RMSE_mAh": ("LOCO RMSE (mAh)", "%.1f"), "LOCO_R2": ("LOCO R²", "%.3f"),
                                         "conformal_coverage": ("Conformal coverage", "%.2f"),
                                         "RF_interval_coverage": ("RF tree coverage", "%.2f"),
                                         "randomKFold_RMSE_mAh": ("Random K-fold RMSE (mAh)", "%.1f")}, height=320)

# ------------------------------------------------------------------------------------------- forecast
with tab_fc:
    c1, c2, c3, c4 = st.columns([1, 1.2, 1.4, 1], gap="medium")
    cell = c1.selectbox("Cell", ["B0005", "B0006", "B0007", "B0018", "B0036", "B0033", "B0034"], key="r_cell")
    model = c2.selectbox("Trend model", list(soh.FADE_MODELS), index=0, key="r_model",
                         help="Linear is the default because it won the rolling-origin backtest below.")
    g = dc[(dc.cell_id == cell) & dc.usable_for_soh].sort_values("discharge_index")
    n = len(g)
    ntr = c3.slider("Cycles available to the forecaster", 20, max(21, n - 10), min(80, n - 10), key="r_n")
    eol = c4.number_input("End-of-life threshold (Ah)", 1.0, 1.9, 1.4, 0.05, key="r_eol")
    r = soh.fade_forecast(g.discharge_index.to_numpy(), g.capacity_Ah.to_numpy(), ntr, eol_Ah=eol, model=model)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r["k"], y=r["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=r["k"], y=r["lo"], line=dict(width=0), fill="tonexty", fillcolor="rgba(196,106,18,0.17)",
                             name="5–95 % bootstrap band"))
    fig.add_trace(go.Scatter(x=g.discharge_index[:ntr], y=g.capacity_Ah[:ntr], mode="markers", name="measured, used",
                             marker=dict(color=ui.PROV["Measured"], size=5)))
    fig.add_trace(go.Scatter(x=g.discharge_index[ntr:], y=g.capacity_Ah[ntr:], mode="markers", name="measured, hidden",
                             marker=dict(color=ui.PROV["Measured"], size=5, symbol="circle-open")))
    fig.add_trace(go.Scatter(x=r["k"], y=r["curve"], name=f"forecast ({model})", line=dict(color=ui.PROV["Predicted"], width=2.6)))
    fig.add_hline(y=eol, line=dict(dash="dash", color=ui.ALERT, width=1.2), annotation_text=f"EOL {eol:.2f} Ah",
                  annotation_font=dict(color=ui.ALERT))
    fig.add_vline(x=ntr, line=dict(dash="dot", color=ui.PROV["Reference"]), annotation_text="forecast origin")
    lo_y = max(0.6, min(g.capacity_Ah.min(), eol) - 0.25)
    fig.update_layout(xaxis_title="Discharge cycle", yaxis_title="Capacity (Ah)", yaxis=dict(range=[lo_y, g.capacity_Ah.max() + 0.08]))
    ui.chart(fig, f"{cell}: remaining-capacity forecast", ("Measured", "Predicted"), height=460, key="r_fig")
    err = (r["eol_pred"] - r["eol_actual"]) if np.isfinite(r["eol_pred"]) and np.isfinite(r["eol_actual"]) else np.nan
    ui.readouts([("Predicted EOL cycle", ui.fmt(r["eol_pred"], 0), "",
                  f"band {ui.fmt(r['eol_band'][1], 0)}–{ui.fmt(r['eol_band'][0], 0)}"),
                 ("Actual EOL cycle", ui.fmt(r["eol_actual"], 0) if np.isfinite(r["eol_actual"]) else "not reached", ""),
                 ("EOL error", f"{err:+.0f}" if np.isfinite(err) else "–", "cycles"),
                 ("Forecast RMSE on hidden cycles", f"{1000*r['test_metrics'].get('RMSE', np.nan):.0f}" if r["test_metrics"] else "–", "mAh"),
                 ("Capacity at the origin", f"{g.capacity_Ah.iloc[ntr-1]:.3f}", "Ah", f"{100*g.capacity_Ah.iloc[ntr-1]/2:.0f} % SOH")],
                cols=5, accent=ui.PROV["Predicted"])
    a, b = st.columns([1, 1.3], gap="large")
    with a:
        with st.container(border=True):
            st.latex(soh.FADE_MODELS[model][2])
            aic = pd.DataFrame([dict(model=m, aicc=v) for m, v in r["aicc"].items()]).sort_values("aicc")
            ui.table(aic, {"model": ("Model",), "aicc": ("AICc on the used cycles (lower = better fit)", "%.1f")})
            st.caption("Robust soft-L1 least squares with bounded parameters; band from a 60-sample residual bootstrap.")
    with b:
        bt = ui.metrics_csv("forecast_backtest.csv")
        if len(bt):
            agg = bt.assign(abs_err=bt.eol_error.abs()).groupby("model").agg(
                rmse=("rmse_hidden_mAh", "median"), eol=("abs_err", "median"), n=("origin", "size")).reindex(list(soh.FADE_MODELS)).reset_index()
            sel = bt[bt.aicc_selected]
            agg.loc[len(agg)] = ["chosen by AICc", sel.rmse_hidden_mAh.median(), sel.eol_error.abs().median(), len(sel)]
            st.markdown('<div class="g14-ch"><span class="g14-ch-t">Backtest: 4 reference cells × 4 forecast origins</span></div>',
                        unsafe_allow_html=True)
            ui.table(agg, {"model": ("Model",), "rmse": ("Median hidden-cycle RMSE (mAh)", "%.0f"),
                           "eol": ("Median |EOL error| (cycles)", "%.0f"), "n": ("Forecasts", "%d")})
    ui.note("The backtest is the honest test of a forecaster: fit on the cycles before an origin, score on the hidden cycles "
            "after it. Flexible exponential models fit the early data best (lowest AICc) but extrapolate badly; the simple "
            "linear trend gives the most reliable end-of-life estimate on these cells. Capacity regeneration after rest "
            "periods (a known NASA artefact) is down-weighted by the robust loss.")
    cfg = dict(cell=cell, n_train=ntr, eol_Ah=eol, model=model)
    summ = dict(eol_pred=r["eol_pred"], eol_actual=r["eol_actual"], test=r["test_metrics"])
    out = pd.DataFrame(dict(cycle=r["k"], forecast_Ah=r["curve"], lo_Ah=r["lo"], hi_Ah=r["hi"]))
    ui.export_bar("SOH forecast", cfg, summ, out, fig, key="fc")
ui.footer()
