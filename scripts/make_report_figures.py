"""Static, publication-quality figures for the technical report and slides.

    python scripts/make_report_figures.py        ->  outputs/figures/report/fig*.png (300 dpi)

Every figure is drawn from the processed data and the saved model metrics, using the same
provenance colours as the application.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from g14lab import data_access as da  # noqa: E402
from g14lab import ecm, estimators, paths, thermal  # noqa: E402
from g14lab import simulator as sim  # noqa: E402
from g14lab.ica import ica_curve  # noqa: E402

OUT = paths.FIGURES / "report"
OUT.mkdir(parents=True, exist_ok=True)
PROV = dict(Measured="#1E2A38", Simulated="#2456C9", Estimated="#0E8A7D", Predicted="#C46A12", Reference="#7A8794")
SERIES = ["#2456C9", "#0E8A7D", "#C46A12", "#7B3FA0", "#B3261E", "#4D7C0F"]

for f in (ROOT / "static" / "fonts" / "ttf").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": ["Barlow", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal",
    "axes.titlelocation": "left", "axes.labelcolor": "#5B6876", "axes.edgecolor": "#AEB8C2", "axes.grid": True,
    "grid.color": "#E4E9EE", "grid.linewidth": 0.8, "xtick.color": "#5B6876", "ytick.color": "#5B6876",
    "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def twin():
    p = ecm.ECMParams.load(paths.TRAINED / "ecm_reference.json")
    th = thermal.ThermalParams(**json.loads((paths.TRAINED / "thermal_reference.json").read_text()))
    return p, th


def fig1_fade(dc):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    groups = [("B0005", "24 °C, 2 A"), ("B0006", "24 °C, 2 A"), ("B0007", "24 °C, 2 A"), ("B0018", "24 °C, 2 A"),
              ("B0047", "4 °C, 1 A"), ("B0029", "43 °C, 4 A")]
    for i, (c, lab) in enumerate(groups):
        g = dc[(dc.cell_id == c) & dc.usable_for_soh]
        ax.plot(g.discharge_index, g.soh_pct, lw=1.5, color=SERIES[i % 6], label=f"{c} ({lab})")
    ax.axhline(70, ls="--", lw=1, color="#B3261E")
    ax.text(2, 71, "end of life, 70 %", color="#B3261E", fontsize=9)
    ax.set(xlabel="Discharge cycle", ylabel="SOH (% of 2.0 Ah rated)", title="Measured capacity fade, NASA PCoE cells")
    ax.legend(ncol=2, loc="lower left")
    save(fig, "fig01_capacity_fade.png")


def fig2_twin(p, th):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
    # (a) the identified OCV lies between the measured charge and discharge curves of the same cycle
    ax = axes[0, 0]
    d, c = da.cycle_dict("B0005", 10), da.charge_after("B0005", 10)
    sd = ecm.soc_reference(d["t"], d["I"], d["Q"])
    sc = ecm.soc_reference(c["t"], c["I"], c["Q"], soc0=0.0)
    ld, lc = d["I"] > 0.5, c["I"] < -0.03
    ax.plot(100 * sc[lc], c["V"][lc], color=PROV["Measured"], lw=1.6, label="measured CC-CV charge, 1.5 A")
    ax.plot(100 * sd[ld], d["V"][ld], color=PROV["Measured"], lw=1.6, ls=":", label="measured discharge, 2 A")
    s = np.linspace(0, 1, 200)
    ax.plot(100 * s, p.ocv(s), color=PROV["Simulated"], lw=2.2, label="identified OCV")
    ax.plot(100 * np.array(p.soc_knots), p.ocv_knots, "o", color=PROV["Simulated"], ms=3.5)
    ax.set(xlabel="SOC (%)", ylabel="Voltage (V)", title="(a) OCV bracketed by charge and discharge, B0005 #10")
    ax.legend(loc="lower right")
    cases = [(axes[0, 1], "discharge", 2, "(b) Training discharge, B0005 #2"),
             (axes[1, 0], "discharge", 100, "(c) Held-out discharge, B0005 #100"),
             (axes[1, 1], "charge", 50, "(d) Held-out charge after B0005 #50")]
    for ax, kind, k, tit in cases:
        cy = da.cycle_dict("B0005", k) if kind == "discharge" else da.charge_after("B0005", k)
        soc = ecm.soc_reference(cy["t"], cy["I"], cy["Q"], soc0=1.0 if kind == "discharge" else 0.0)
        vm, _ = ecm.simulate_voltage(p, cy["t"], cy["I"], soc, T_C=cy["T"])
        ax.plot(cy["t"] / 60, cy["V"], color=PROV["Measured"], lw=2, label="measured")
        ax.plot(cy["t"] / 60, vm, color=PROV["Simulated"], lw=2, ls="--", label="simulated")
        rmse = 1000 * np.sqrt(np.mean((vm - cy["V"]) ** 2))
        ax.set(xlabel="Time (min)", ylabel="Terminal voltage (V)", title=f"{tit}: RMSE {rmse:.0f} mV")
        ax.legend(loc="lower left" if kind == "discharge" else "lower right")
    fig.tight_layout()
    save(fig, "fig02_twin_identification.png")


def fig3_laws(p):
    pm = pd.read_csv(paths.METRICS / "parameter_map.csv")
    used = pm[pm.used_in_arrhenius_fit]
    tr = pd.read_parquet(paths.METRICS / "param_tracking.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    sc = axes[0].scatter(used.mean_T_C, 1000 * used.R_total_ohm, c=used.mean_current_A, cmap="viridis", s=28, vmin=1, vmax=4,
                         edgecolor="white", lw=0.5)
    T = np.linspace(5, 60, 100)
    r_ref, t_ref = p.extra.get("map_R_ref_ohm", p.R0 + p.R1), p.extra.get("map_T_ref_C", p.T_ref_C)
    law = r_ref * np.exp(p.Ea_R0 / ecm.R_GAS * (1 / (T + 273.15) - 1 / (t_ref + 273.15)))
    axes[0].plot(T, 1000 * law, "--", color=PROV["Simulated"], lw=1.8, label=f"Arrhenius at 2 A, Ea = {p.Ea_R0/1000:.1f} kJ/mol")
    axes[0].set(xlabel="Mean cell temperature (°C)", ylabel="R0 + R1 (mΩ)",
                title=f"(a) Temperature dependence, {used.cell_id.nunique()} cells")
    axes[0].legend(loc="upper right")
    cb = fig.colorbar(sc, ax=axes[0], pad=0.01)
    cb.set_label("Load (A)")
    t = tr[tr.cell_id == "B0005"].sort_values("discharge_index")
    ax = axes[1]
    ax.plot(t.discharge_index, 1000 * t.R_total_ohm.rolling(9, center=True, min_periods=3).median(), color=PROV["Simulated"],
            lw=2, label="twin R0 + R1 (tracked, rolling median)")
    ax.set(xlabel="Discharge cycle", ylabel="Twin R0 + R1 (mΩ)", title="(b) Digital shadow vs EIS, B0005")
    ax2 = ax.twinx()
    ax2.plot(t.discharge_index, 1000 * (t.Re_ohm + t.Rct_ohm), color=PROV["Measured"], lw=1.6, label="EIS Re + Rct")
    ax2.set_ylabel("EIS Re + Rct (mΩ)")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    save(fig, "fig03_physical_laws.png")


def fig4_thermal(p, th):
    crates, ambs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [4, 10, 25, 35, 45]
    Z = np.zeros((len(ambs), len(crates)))
    for i, a in enumerate(ambs):
        for j, c in enumerate(crates):
            _, s = sim.run_protocol(p, th, sim.TwinState(soc0=1, T0_C=a, T_amb_C=a),
                                    [{"type": "cc", "current_A": c * p.Q_Ah, "until_V": 2.7}], dt=2.0)
            Z[i, j] = s["dT_max_C"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    im = ax.imshow(Z, cmap="YlOrRd", aspect="auto")
    for i in range(len(ambs)):
        for j in range(len(crates)):
            ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if Z[i, j] > 0.6 * Z.max() else "#18212B")
    ax.set_xticks(range(len(crates)), [f"{c:g}C" for c in crates])
    ax.set_yticks(range(len(ambs)), [f"{a} °C" for a in ambs])
    ax.grid(False)
    ax.set(xlabel="C-rate", ylabel="Ambient", title="Simulated peak temperature rise ΔT (K), CC discharge")
    fig.colorbar(im, ax=ax, pad=0.02).set_label("ΔT (K)")
    save(fig, "fig04_thermal_stress_map.png")


def fig5_soc(p):
    c = da.cycle_dict("B0005", 60)
    ref = ecm.soc_reference(c["t"], c["I"], c["Q"])
    Im = c["I"] * 1.02 + 0.02
    cc = estimators.coulomb_counting(c["t"], Im, p.Q_Ah, 0.8)
    e = estimators.ekf(p, c["t"], Im, c["V"], 0.8, Q_Ah=p.Q_Ah, T_C=c["T"])
    ae = estimators.ekf(p, c["t"], Im, c["V"], 0.8, Q_Ah=p.Q_Ah, T_C=c["T"], adaptive=True)
    u = estimators.ukf(p, c["t"], Im, c["V"], 0.8, Q_Ah=p.Q_Ah, T_C=c["T"])
    sb = pd.read_csv(paths.METRICS / "soc_benchmark.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), gridspec_kw=dict(width_ratios=[1.5, 1]))
    x = c["t"] / 60
    axes[0].plot(x, 100 * ref, color=PROV["Reference"], lw=3, label="reference (Ah-integrated)")
    for lab, s, col, ls in (("Coulomb counting", cc, SERIES[2], "-"), ("EKF", e["soc"], PROV["Estimated"], "-"),
                            ("adaptive EKF", ae["soc"], SERIES[0], "-"), ("UKF", u["soc"], SERIES[3], "--")):
        axes[0].plot(x, 100 * s, color=col, lw=1.6, ls=ls, label=lab)
    axes[0].set(xlabel="Time (min)", ylabel="SOC (%)", title="(a) B0005 #60: 80 % initial guess, +2 % gain, +20 mA offset")
    axes[0].legend(loc="lower left", ncol=2)
    m = sb[sb.capacity_used == "twin capacity (fresh)"].groupby("estimator").MAE_pct.median().reindex(["CC", "EKF", "AEKF", "UKF"])
    axes[1].bar(["Coulomb\ncounting", "EKF", "adaptive\nEKF", "UKF"], m.values,
                color=[SERIES[2], PROV["Estimated"], SERIES[0], SERIES[3]])
    for i, v in enumerate(m.values):
        axes[1].text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
    axes[1].set(ylabel="Median MAE (percentage points)", title="(b) 10 held-out cycles")
    fig.tight_layout()
    save(fig, "fig05_soc_estimation.png")


def fig6_ica(dc):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ks = [2, 40, 80, 120, 160]
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(ks)))
    ts = da.charge_ts("B0005", tuple(ks))
    for k, col in zip(ks, cols):
        s = ts[ts.charge_index == k]
        r = ica_curve(s.t_s.to_numpy(float), s.V.to_numpy(float), -s.I.to_numpy(float))
        if r is not None:
            axes[0].plot(r[0], r[1], color=col, lw=1.8, label=f"charge {k}")
    axes[0].set(xlabel="Voltage (V)", ylabel="dQ/dV (Ah/V)", title="(a) Incremental capacity, B0005 CC charge")
    axes[0].legend(loc="upper left")
    d = dc[(dc.cell_id == "B0005") & dc.usable_for_soh & dc.ic_peak_AhV.between(0.5, 15)]
    axes[1].scatter(d.ic_peak_AhV, d.capacity_Ah, c=d.discharge_index, cmap="viridis", s=14)
    rho = d[["ic_peak_AhV", "capacity_Ah"]].corr(method="spearman").iloc[0, 1]
    axes[1].set(xlabel="ICA peak height (Ah/V)", ylabel="Discharge capacity (Ah)",
                title=f"(b) Peak height vs capacity, Spearman ρ = {rho:.2f}")
    fig.tight_layout()
    save(fig, "fig06_ica.png")


def fig7_soh():
    preds = pd.read_parquet(paths.METRICS / "soh_loco_predictions.parquet")
    bm = pd.read_csv(paths.METRICS / "soh_benchmark.csv")
    fs = "Charging + ICA + EIS impedance (Re, Rct)"
    pr = preds[(preds.feature_set == fs) & (preds.model == "Random Forest") & preds.cell_set.str.startswith("Reference")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw=dict(width_ratios=[1, 1.35]))
    for i, (c, g) in enumerate(pr.groupby("cell_id")):
        axes[0].scatter(g.capacity_Ah, g.pred_Ah, s=10, color=SERIES[i], label=f"{c} held out")
    lo, hi = pr.capacity_Ah.min(), pr.capacity_Ah.max()
    axes[0].plot([lo, hi], [lo, hi], "--", color=PROV["Reference"])
    axes[0].set(xlabel="Measured capacity (Ah)", ylabel="Predicted capacity (Ah)",
                title="(a) Random Forest, leave-one-cell-out")
    axes[0].legend(loc="upper left")
    r = bm[bm.cell_set.str.startswith("Reference")].copy()
    r["label"] = r.model.str.replace(" regression", "") + ", " + r.feature_set.str.replace(" (on-board measurable)", "", regex=False)\
        .str.replace(" impedance (Re, Rct)", "", regex=False).str.replace(" indicators", "", regex=False)
    r = r.sort_values("LOCO_RMSE_mAh")
    y = np.arange(len(r))
    axes[1].hlines(y, r.randomKFold_RMSE_mAh, r.LOCO_RMSE_mAh, color="#C9D1D9", lw=3)
    axes[1].plot(r.randomKFold_RMSE_mAh, y, "o", color=PROV["Reference"], label="random K-fold (leaky)")
    axes[1].plot(r.LOCO_RMSE_mAh, y, "o", color=PROV["Predicted"], label="leave-one-cell-out (honest)")
    axes[1].set_yticks(y, r.label, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="RMSE (mAh)", title="(b) Validation gap, reference cells")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2)
    fig.tight_layout()
    save(fig, "fig07_soh_models.png")


def fig8_vehicle():
    ss = pd.read_parquet(paths.PROCESSED / "bev_sessions.parquet")
    fc = ss[ss.kind == "fast_charge"]
    F = pd.concat([pd.read_parquet(paths.PROCESSED / "bev_ts_fast_charge.parquet", filters=[("session_id", "==", s)])
                   for s in fc.session_id], ignore_index=True)
    F["vehicle"] = F.vehicle.astype(str)
    F = F[F.i_pack_A < -5]
    F["P_kW"] = -F.v_pack_V * F.i_pack_A / 1000
    dr = ss[(ss.kind == "drive") & ss.kwh_per_100km.between(5, 60) & (ss.distance_km > 5)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    for v, col in (("BEV1", SERIES[0]), ("BEV2", SERIES[2])):
        f = F[F.vehicle == v].iloc[::4]
        axes[0].scatter(f.soc_bms_pct, f.P_kW, s=2, alpha=0.35, color=col, label=v)
        d = dr[dr.vehicle == v]
        axes[1].scatter(d.t_amb_mean_C, d.kwh_per_100km, s=6, alpha=0.35, color=col, label=v)
        b = d.groupby(pd.cut(d.t_amb_mean_C, np.arange(-10, 41, 5)), observed=True).kwh_per_100km.agg(["median", "size"])
        b = b[b["size"] >= 5]
        axes[1].plot([i.mid for i in b.index], b["median"], "-o", color=col, lw=2, ms=4)
    axes[0].set(xlabel="BMS SOC (%)", ylabel="Charging power (kW)", title=f"(a) DC fast-charge taper, {len(fc)} sessions")
    axes[0].legend(markerscale=6, loc="upper right")
    axes[1].set(xlabel="Mean ambient temperature (°C)", ylabel="kWh / 100 km", title=f"(b) Energy use vs ambient, {len(dr)} drives")
    axes[1].set_ylim(5, 45)
    fig.tight_layout()
    save(fig, "fig08_tesla_telemetry.png")


def fig9_forecast(dc):
    from g14lab import soh
    g = dc[(dc.cell_id == "B0006") & dc.usable_for_soh].sort_values("discharge_index")
    r = soh.fade_forecast(g.discharge_index.to_numpy(), g.capacity_Ah.to_numpy(), 80, eol_Ah=1.4, model="linear")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.fill_between(r["k"], r["lo"], r["hi"], color=PROV["Predicted"], alpha=0.18, label="5–95 % bootstrap band")
    ax.plot(g.discharge_index[:80], g.capacity_Ah[:80], "o", ms=3, color=PROV["Measured"], label="measured, used")
    ax.plot(g.discharge_index[80:], g.capacity_Ah[80:], "o", ms=3, mfc="none", color=PROV["Measured"], label="measured, hidden")
    ax.plot(r["k"], r["curve"], color=PROV["Predicted"], lw=2.2, label=f"forecast ({r['model']})")
    ax.axhline(1.4, ls="--", color="#B3261E", lw=1)
    ax.axvline(80, ls=":", color=PROV["Reference"])
    ax.set(xlabel="Discharge cycle", ylabel="Capacity (Ah)", xlim=(0, 260),
           ylim=(0.9, 2.1), title=f"B0006 linear fade forecast from 80 cycles: EOL predicted {r['eol_pred']:.0f}, actual {r['eol_actual']:.0f}")
    ax.legend(loc="upper right")
    save(fig, "fig09_fade_forecast.png")


if __name__ == "__main__":
    dc = da.discharge_cycles()
    p, th = twin()
    fig1_fade(dc)
    fig2_twin(p, th)
    fig3_laws(p)
    fig4_thermal(p, th)
    fig5_soc(p)
    fig6_ica(dc)
    fig7_soh()
    if (paths.PROCESSED / "bev_sessions.parquet").exists():
        fig8_vehicle()
    fig9_forecast(dc)
