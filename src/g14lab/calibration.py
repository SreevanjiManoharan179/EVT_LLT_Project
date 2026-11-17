"""Builds the calibrated reference twin and pre-computes every validation artefact.

    python -m g14lab.calibration        (run.sh does this automatically if models are missing)

Steps
  1. Identify the reference twin (OCV, R0, R1, tau1) on B0005 discharges 2-6 and the CC-CV charges
     that follow them (two passes: constant-R, then temperature-aware at 24 degC with the Arrhenius
     law from a multi-cell R(T, I) map; R0 bounded below by the cell's EIS Re), then C_th, hA and
     the entropic coefficient dU/dT on the measured temperature of both current directions.
  2. Digital-shadow tracking: re-identify R0, R1, tau1 on EVERY usable discharge of the four
     reference cells (OCV fixed, temperature-aware, R0 >= the latest EIS Re of that cell) ->
     parameter trajectories vs ageing and a resistance-growth law.
  3. Twin validation: measured-vs-simulated voltage/temperature on held-out discharges and charges
     (replayed current), CC-phase duration of a simulated CC-CV charge vs measured.
  4. SOH benchmark: 3 models x 4 indicator sets x 2 cell sets, leave-one-cell-out + leaky K-fold,
     RF tree intervals and cross-conformal intervals.
  5. SOC benchmark: CC, EKF, adaptive EKF, UKF with three sources of capacity knowledge.
Outputs: models/trained_models/*, models/model_metrics/*
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from . import data_access as da
from . import ecm, estimators, paths, soh, thermal
from . import evaluation as ev

REF_CELL = "B0005"
REF_TRAIN = [2, 3, 4, 5, 6]
MAP_CELLS = ["B0005", "B0006", "B0007", "B0018", "B0036", "B0025", "B0026", "B0027", "B0028", "B0029",
             "B0030", "B0031", "B0032", "B0033", "B0034", "B0045", "B0046", "B0047", "B0048", "B0053",
             "B0054", "B0055", "B0056", "B0042", "B0043", "B0044"]
VALIDATION_CASES = [("B0005", i) for i in (10, 25, 50, 75, 100, 125, 150, 168)] + \
    [("B0006", 10), ("B0007", 10), ("B0018", 10), ("B0036", 10), ("B0025", 5), ("B0026", 5), ("B0027", 5),
     ("B0028", 5), ("B0033", 20), ("B0034", 20), ("B0029", 5), ("B0030", 5), ("B0046", 5), ("B0047", 5),
     ("B0055", 5)]
SOC_CASES = [("B0005", 20), ("B0005", 60), ("B0005", 100), ("B0006", 30), ("B0007", 30), ("B0018", 30),
             ("B0025", 10), ("B0026", 10), ("B0027", 10), ("B0036", 30)]


# ---------------------------------------------------------------------------- 1. reference twin
def reference_cycles(cell=REF_CELL, train=REF_TRAIN):
    """Measured training data: each discharge plus the CC-CV charge that follows it. The charge is
    normalised by the capacity of the discharge before it and starts at SOC 0 (that discharge's cut-off)."""
    dis = [da.cycle_dict(cell, int(i)) for i in train]
    chg = [c for c in (da.charge_after(cell, int(i)) for i in train) if c is not None]
    return dis, chg


def arrhenius_map(p_const: ecm.ECMParams):
    """Multi-cell R(T, I) map: R0, R1, tau1 refitted per early-life cycle with the OCV fixed and constant
    resistances, then ln(R0 + R1) regressed on 1/T and ln(I) -> activation energy Ea."""
    d = da.discharge_cycles()
    rows = []
    for c in MAP_CELLS:
        g = d[(d.cell_id == c) & d.usable_for_soh & (~d.flag_capacity_mismatch)].head(12)
        for i in g.discharge_index.values[2:8:2]:
            cd = da.cycle_dict(c, int(i))
            q = ecm.identify_ecm([cd], fix_ocv=p_const)
            rows.append(dict(cell_id=c, discharge_index=int(i), ambient_C=cd["ambient"],
                             mean_current_A=float(cd["I"][cd["I"] > 0.2].mean()), mean_T_C=float(cd["T"].mean()),
                             R0_ohm=q.R0, R1_ohm=q.R1, R_total_ohm=q.R0 + q.R1, tau1_s=q.tau1, fit_rmse_mV=q.fit_rmse_mV))
    m = pd.DataFrame(rows)
    m["used_in_arrhenius_fit"] = m.fit_rmse_mV < 80
    ok = m[m.used_in_arrhenius_fit]
    X = np.column_stack([np.ones(len(ok)), 1 / (ok.mean_T_C + 273.15) - 1 / (p_const.T_ref_C + 273.15),
                         np.log(ok.mean_current_A / 2.0)])
    yl = np.log(ok.R_total_ohm)
    coef, *_ = np.linalg.lstsq(X, yl, rcond=None)
    r2 = float(1 - np.sum((yl - X @ coef) ** 2) / np.sum((yl - yl.mean()) ** 2))
    stats = dict(arrhenius_fit_R2=r2, arrhenius_n=int(len(ok)), arrhenius_cells=int(ok.cell_id.nunique()),
                 map_cells_attempted=len(MAP_CELLS), current_exponent_beta=float(coef[2]),
                 calibrated_T_range_C=[float(ok.mean_T_C.min()), float(ok.mean_T_C.max())],
                 map_R_ref_ohm=float(np.exp(coef[0])), map_T_ref_C=float(p_const.T_ref_C),
                 map_note="Ea fitted on ln(R0+R1) vs 1/T and ln(I) over early cycles with fit RMSE < 80 mV")
    return m, float(coef[1] * ecm.R_GAS), stats


def calibrate_reference(cell=REF_CELL, train=REF_TRAIN, T_ref_C=24.0, verbose=True):
    """Two-pass identification of the reference twin.
    Pass 1: OCV, R0, R1, tau1 on discharges + following charges with constant resistances; its OCV
            anchors the multi-cell R(T, I) map that yields the activation energy Ea.
    Pass 2: the same data refitted temperature-aware (Arrhenius along the measured cell temperature),
            so R0 and R1 refer to T_ref = 24 degC. R0 >= EIS Re of the same cell in both passes.
    Thermal: C_th, hA and the entropic coefficient dU/dT identified on the measured temperature of the
    training discharges and charges (the reversible heat changes sign with the current)."""
    dis, chg = reference_cycles(cell, train)
    re0 = da.eis_ohmic_resistance(cell) or 0.005
    src = f"{cell} discharges {train[0]}-{train[-1]} + {len(chg)} following CC-CV charges (24 °C)"
    p1 = ecm.identify_ecm(dis + chg, source=src, r0_min=re0)
    m, Ea, stats = arrhenius_map(p1)
    p = ecm.identify_ecm(dis + chg, source=src, Ea=Ea, T_ref_C=T_ref_C, r0_min=re0)
    p.extra.update(stats)
    p.extra.update(eis_Re_ohm=re0, identification="joint discharge + charge, temperature-aware, two-pass")
    th = thermal.identify_thermal(dis + chg, p, [c["ambient"] for c in dis + chg],
                                  source=f"{cell} discharges {train[0]}-{train[-1]} + following charges")
    m.to_csv(paths.METRICS / "parameter_map.csv", index=False)
    if verbose:
        print(f"pass 1: R0={p1.R0:.4f} R1={p1.R1:.4f} tau={p1.tau1:.1f}s at {p1.T_ref_C:.1f} °C  fit={p1.fit_rmse_mV:.1f} mV")
        print(f"ECM (T_ref {p.T_ref_C:.0f} °C): R0={p.R0:.4f} R1={p.R1:.4f} tau={p.tau1:.1f}s fit={p.fit_rmse_mV:.1f} mV "
              f"Ea={Ea:.0f} J/mol R2={stats['arrhenius_fit_R2']:.3f}  (R0 >= EIS Re {re0*1000:.1f} mΩ)")
        print(f"Thermal: C_th={th.C_th:.1f} J/K hA={th.hA:.4f} W/K dU/dT={th.dUdT*1000:+.3f} mV/K fit={th.fit_rmse_C:.2f} K")
    return p, th, m


# ---------------------------------------------------------------------------- 2. digital shadow
def track_parameters(p: ecm.ECMParams, cells=tuple(soh.REFERENCE_CELLS)) -> pd.DataFrame:
    d = da.discharge_cycles()
    rows = []
    for cell in cells:
        g = d[(d.cell_id == cell) & d.usable_for_soh & ~d.flag_capacity_mismatch]
        re_first = da.eis_ohmic_resistance(cell) or 0.005
        prev = None  # parameters identified on the previous discharge (digital-shadow state)
        for i in g.discharge_index.values:
            c = da.cycle_dict(cell, int(i))
            # R0 is bounded by the ohmic resistance of the latest EIS on this cell (first EIS before it exists)
            re_k = float(g.loc[g.discharge_index == i, "Re_ohm"].iloc[0])
            re_k = re_k if 0.01 < re_k < 0.5 else re_first
            q = ecm.identify_ecm([c], fix_ocv=p, init=(max(p.R0, 1.01 * re_k), p.R1, p.tau1), r0_min=re_k)
            soc = ecm.soc_reference(c["t"], c["I"], c["Q"])
            vm, _ = ecm.simulate_voltage(p, c["t"], c["I"], soc, T_C=c["T"])
            ahead = np.nan
            if prev is not None:
                # one-cycle-ahead PREDICTION: parameters and capacity known after discharge k-1,
                # applied to the unseen discharge k (honest validation of the tracked twin)
                soc_p = ecm.soc_reference(c["t"], c["I"], prev.Q_Ah)
                va, _ = ecm.simulate_voltage(prev, c["t"], c["I"], soc_p, T_C=c["T"])
                ahead = 1000 * ev.rmse(c["V"], va)
            r = g[g.discharge_index == i].iloc[0]
            rows.append(dict(cell_id=cell, discharge_index=int(i), capacity_Ah=c["Q_nasa"], soh_pct=float(r.soh_pct),
                             R0_ohm=q.R0, R1_ohm=q.R1, R_total_ohm=q.R0 + q.R1, tau1_s=q.tau1,
                             V_RMSE_tracked_fit_mV=q.fit_rmse_mV, V_RMSE_ahead_mV=ahead,
                             V_RMSE_fresh_mV=1000 * ev.rmse(c["V"], vm),
                             Re_ohm=float(r.Re_ohm), Rct_ohm=float(r.Rct_ohm), mean_T_C=float(c["T"].mean())))
            q.Q_Ah = c["Q"]
            prev = q
    t = pd.DataFrame(rows)
    t.to_parquet(paths.METRICS / "param_tracking.parquet", index=False)
    return t


def resistance_growth(track: pd.DataFrame, p: ecm.ECMParams, n_early: int = 10) -> dict:
    """Relative resistance growth with capacity fade, each cell normalised to its own early life:
        R_total / R_total,early = 1 + k * (SOH_early - SOH) / 100
    R_early and SOH_early are the cell's medians over its first n_early tracked cycles. The law is fitted
    through the origin (no fade, no growth) on every tracked cycle of the reference cells."""
    xs, ys = [], []
    for _, t in track.groupby("cell_id"):
        t = t.sort_values("discharge_index")
        xs.append(((t.soh_pct.iloc[:n_early].median() - t.soh_pct) / 100).to_numpy())
        ys.append((t.R_total_ohm / t.R_total_ohm.iloc[:n_early].median()).to_numpy())
    x, y = np.concatenate(xs), np.concatenate(ys)
    k = float(np.sum(x * (y - 1)) / np.sum(x * x))
    r2 = float(1 - np.sum((y - 1 - k * x) ** 2) / np.sum((y - y.mean()) ** 2))
    return dict(form="relative", k_per_fade=k, intercept=1.0, R2=r2, n=int(len(x)), n_early=n_early)


def shadow_validation(p: ecm.ECMParams, track: pd.DataFrame, step: int = 3) -> pd.DataFrame:
    """Does updating the twin's resistances with ageing improve PREDICTION of unseen discharges?
    All variants get the cycle's measured capacity, so only the resistance model differs:
      fresh      - reference twin, no ageing update
      shadow     - R0, R1, tau1 = rolling median of the previous 5 tracked cycles (one-cycle-ahead)
      growth law - reference twin with R scaled by the fitted resistance-growth law at the cycle's SOH"""
    rows = []
    for cell, t in track.groupby("cell_id"):
        t = t.sort_values("discharge_index").reset_index(drop=True)
        for col in ("R0_ohm", "R1_ohm", "tau1_s"):
            t[col + "_roll"] = t[col].shift(1).rolling(5, min_periods=1).median()
        for j in range(1, len(t), step):
            r = t.iloc[j]
            c = da.cycle_dict(cell, int(r.discharge_index))
            soc = ecm.soc_reference(c["t"], c["I"], c["Q"])
            q = ecm.ECMParams.from_dict(p.to_dict())
            q.R0, q.R1, q.tau1 = r.R0_ohm_roll, r.R1_ohm_roll, r.tau1_s_roll
            vf, _ = ecm.simulate_voltage(p, c["t"], c["I"], soc, T_C=c["T"])
            vs, _ = ecm.simulate_voltage(q, c["t"], c["I"], soc, T_C=c["T"])
            vg, _ = ecm.simulate_voltage(p, c["t"], c["I"], soc, T_C=c["T"], r_scale=p.resistance_factor(r.soh_pct))
            rows.append(dict(cell_id=cell, discharge_index=int(r.discharge_index), soh_pct=float(r.soh_pct),
                             fresh_mV=1000 * ev.rmse(c["V"], vf), shadow_mV=1000 * ev.rmse(c["V"], vs),
                             growth_mV=1000 * ev.rmse(c["V"], vg)))
    out = pd.DataFrame(rows)
    out["life_stage"] = pd.cut(out.soh_pct, [0, 75, 85, 101], labels=["late (SOH < 75 %)", "mid (75–85 %)", "early (> 85 %)"])
    out.to_csv(paths.METRICS / "shadow_validation.csv", index=False)
    return out


# ---------------------------------------------------------------------------- 3. twin validation
def evaluate_twin(p: ecm.ECMParams, th: thermal.ThermalParams, cases=VALIDATION_CASES, track: pd.DataFrame | None = None):
    d = da.discharge_cycles()
    rows = []
    for cell, i in cases:
        if not ((d.cell_id == cell) & (d.discharge_index == i)).any():
            continue
        c = da.cycle_dict(cell, i)
        soc = ecm.soc_reference(c["t"], c["I"], c["Q"])
        vm, v1 = ecm.simulate_voltage(p, c["t"], c["I"], soc, T_C=c["T"])
        Ts, _ = thermal.simulate_temperature(th, c["t"], c["I"], v1, p.r0_at(c["T"]), p.r1_at(c["T"], 1.0, c["I"]), c["ambient"], c["T"][0])
        row = dict(cell_id=cell, discharge_index=i, ambient_C=c["ambient"], mean_current_A=float(c["I"][c["I"] > 0.2].mean()),
                   capacity_Ah=c["Q"], V_RMSE_mV=1000 * ev.rmse(c["V"], vm), V_MAE_mV=1000 * ev.mae(c["V"], vm),
                   T_RMSE_C=ev.rmse(c["T"], Ts), T_MAE_C=ev.mae(c["T"], Ts), Tmax_meas_C=float(c["T"].max()),
                   Tmax_sim_C=float(Ts.max()),
                   split="training cycle" if (cell == REF_CELL and i in REF_TRAIN) else "held-out")
        rows.append(row)
    return pd.DataFrame(rows)


CHARGE_CASES = [("B0005", i) for i in (10, 50, 100, 150)] + \
    [("B0006", 10), ("B0007", 10), ("B0018", 10), ("B0036", 10), ("B0029", 5), ("B0047", 5)]


def evaluate_charges(p: ecm.ECMParams, th: thermal.ThermalParams, cases=CHARGE_CASES) -> pd.DataFrame:
    """Charge-side validation on held-out cycles (each case = the CC-CV charge after that discharge):
      replay   - measured charge current through the twin -> voltage and temperature RMSE
      protocol - CC 1.5 A to 4.2 V then CV to 20 mA simulated from SOC 0 with the cycle's capacity and the
                 measured ambient; CC-phase duration and charge compared with the measured CC phase
                 (fresh resistances, and resistances scaled by the growth law at the cycle's SOH)."""
    from . import simulator as sim
    ch = da.charge_cycles()
    rows = []
    for cell, i in cases:
        c = da.charge_after(cell, i)
        if c is None:
            continue
        k, q = c["idx"], c["Q"]
        soc = ecm.soc_reference(c["t"], c["I"], c["Q"], soc0=0.0)
        vm, v1 = ecm.simulate_voltage(p, c["t"], c["I"], soc, T_C=c["T"])
        Ts, _ = thermal.simulate_temperature(th, c["t"], c["I"], v1, p.r0_at(c["T"]), p.r1_at(c["T"], 1.0, c["I"]), c["ambient"], c["T"][0])
        r = ch[(ch.cell_id == cell) & (ch.charge_index == k)].iloc[0]
        soh_pct = 100 * q / ecm.RATED_AH
        steps = [{"type": "cc", "current_A": -1.5, "until_V": 4.2}, {"type": "cv", "voltage_V": 4.2, "until_A": 0.02}]
        sims = {}
        for tag, rs in (("fresh", 1.0), ("aged", p.resistance_factor(soh_pct))):
            st_ = sim.TwinState(soc0=0.0, T0_C=float(c["T"][0]), T_amb_C=c["ambient"], soh=q / p.Q_Ah, r_scale=rs)
            df, _ = sim.run_protocol(p, th, st_, steps, dt=2.0)
            ccp = df[df.step == 1]
            sims[tag] = (float(ccp.t_s.iloc[-1] - ccp.t_s.iloc[0]) / 60, float(-ccp.Ah_out.iloc[-1]))
        rows.append(dict(cell_id=cell, discharge_index=int(i), charge_index=int(k), ambient_C=c["ambient"], capacity_Ah=q,
                         soh_pct=soh_pct, V_RMSE_mV=1000 * ev.rmse(c["V"], vm), V_MAE_mV=1000 * ev.mae(c["V"], vm),
                         T_RMSE_C=ev.rmse(c["T"], Ts), t_cc_meas_min=float(r.t_cc_s) / 60, t_cc_sim_min=sims["fresh"][0],
                         t_cc_sim_aged_min=sims["aged"][0], q_cc_meas_Ah=float(r.q_cc_Ah), q_cc_sim_Ah=sims["fresh"][1],
                         q_cc_sim_aged_Ah=sims["aged"][1]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- 4. SOH
def validate_soh():
    d = da.discharge_cycles()
    rows, preds = [], []
    for fs_name, feats in soh.FEATURE_SETS.items():
        for cell_set, cells, cond in (("Reference protocol (B0005/6/7/18)", soh.REFERENCE_CELLS, False),
                                      ("All usable cells (cross-condition)", None, True)):
            df = soh.dataset(d, feats, cells, cond)
            cols = feats + (soh.CONDITION_FEATURES if cond else [])
            for mname in soh.MODELS:
                nest = 300 if cells else 120
                out, per_cell, pooled = soh.loco(df, cols, mname, n_estimators=nest)
                leaky = soh.random_kfold(df, cols, mname, n_estimators=nest)
                rows.append(dict(feature_set=fs_name, cell_set=cell_set, model=mname, n_rows=len(df),
                                 n_cells=df.cell_id.nunique(), LOCO_MAE_mAh=1000 * pooled["MAE"],
                                 LOCO_RMSE_mAh=1000 * pooled["RMSE"], LOCO_R2=pooled["R2"],
                                 LOCO_RMSE_SOH_pct=100 * pooled["RMSE"] / soh.RATED_AH,
                                 RF_interval_coverage=pooled.get("interval_coverage_90", np.nan),
                                 conformal_coverage=pooled["conformal_coverage_90"],
                                 conformal_halfwidth_mAh=pooled["conformal_halfwidth_mAh"],
                                 randomKFold_RMSE_mAh=1000 * leaky["RMSE"], randomKFold_R2=leaky["R2"]))
                preds.append(out.assign(feature_set=fs_name, cell_set=cell_set, model=mname))
    res = pd.DataFrame(rows)
    res.to_csv(paths.METRICS / "soh_benchmark.csv", index=False)
    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(paths.METRICS / "soh_loco_predictions.parquet", index=False)
    # Final Group 14 baseline: Random Forest on the reference protocol, best indicator set by LOCO RMSE
    rf = res[(res.model == "Random Forest") & (res.cell_set.str.startswith("Reference"))]
    best = rf.sort_values("LOCO_RMSE_mAh").iloc[0].feature_set
    feats = soh.FEATURE_SETS[best]
    df = soh.dataset(d, feats, soh.REFERENCE_CELLS, False)
    joblib.dump(dict(model=soh.fit_full(df, feats, "Random Forest"), features=feats, feature_set=best,
                     trained_on=soh.REFERENCE_CELLS, target="capacity_Ah"), paths.TRAINED / "rf_soh_reference.joblib")
    return res, allp, best


# ---------------------------------------------------------------------------- 5. SOC
def rf_capacity_for(cell, idx, allp: pd.DataFrame, best_fs: str):
    """Capacity the SOH model would report for this cycle WITHOUT having seen the cell
    (LOCO prediction for reference cells; full model for other cells)."""
    m = allp[(allp.feature_set == best_fs) & (allp.model == "Random Forest") &
             allp.cell_set.str.startswith("Reference") & (allp.cell_id == cell) & (allp.discharge_index == idx)]
    if len(m):
        return float(m.pred_Ah.iloc[0]), "LOCO (cell unseen)"
    b = joblib.load(paths.TRAINED / "rf_soh_reference.joblib")
    row = da.discharge_cycles().query("cell_id == @cell and discharge_index == @idx")
    x = row[b["features"]].to_numpy(float)
    if len(x) and np.isfinite(x).all():
        return float(b["model"].predict(x)[0]), "full model (cell not in training set)"
    return np.nan, "unavailable"


def validate_soc(p, allp, best_fs):
    rows = []
    for cell, i in SOC_CASES:
        c = da.cycle_dict(cell, i)
        ref = ecm.soc_reference(c["t"], c["I"], c["Q"])
        Im = c["I"] * 1.02 + 0.02
        q_rf, _ = rf_capacity_for(cell, i, allp, best_fs)
        modes = [("twin capacity (fresh)", p.Q_Ah), ("RF-predicted capacity", q_rf), ("measured capacity (oracle)", c["Q"])]
        for qmode, Qe in modes:
            if not np.isfinite(Qe):
                continue
            res = dict(CC=estimators.coulomb_counting(c["t"], Im, Qe, 0.8),
                       EKF=estimators.ekf(p, c["t"], Im, c["V"], 0.8, Q_Ah=Qe, T_C=c["T"])["soc"],
                       AEKF=estimators.ekf(p, c["t"], Im, c["V"], 0.8, Q_Ah=Qe, T_C=c["T"], adaptive=True)["soc"],
                       UKF=estimators.ukf(p, c["t"], Im, c["V"], 0.8, Q_Ah=Qe, T_C=c["T"])["soc"])
            for name, s in res.items():
                mtr = ev.summary(ref, s, 100)
                rows.append(dict(cell_id=cell, discharge_index=i, capacity_used=qmode, capacity_Ah=Qe, estimator=name,
                                 MAE_pct=mtr["MAE"], RMSE_pct=mtr["RMSE"], final_error_pct=100 * (s[-1] - ref[-1]),
                                 convergence_s=estimators.convergence_time(c["t"], s - ref)))
    out = pd.DataFrame(rows)
    out.to_csv(paths.METRICS / "soc_benchmark.csv", index=False)
    return out


def main():
    t0 = time.time()
    p, th, _ = calibrate_reference()
    track = track_parameters(p)
    growth = resistance_growth(track, p)
    p.extra["resistance_growth"] = growth
    p.save(paths.TRAINED / "ecm_reference.json")
    (paths.TRAINED / "thermal_reference.json").write_text(json.dumps(th.to_dict(), indent=2))
    print("resistance growth law:", growth)
    tv = evaluate_twin(p, th)
    tv.to_csv(paths.METRICS / "twin_validation.csv", index=False)
    print(tv[["cell_id", "discharge_index", "V_RMSE_mV", "T_RMSE_C"]].round(1).to_string(index=False))
    cv = evaluate_charges(p, th)
    cv.to_csv(paths.METRICS / "charge_validation.csv", index=False)
    print(cv[["cell_id", "discharge_index", "V_RMSE_mV", "T_RMSE_C", "t_cc_meas_min", "t_cc_sim_min",
              "t_cc_sim_aged_min"]].round(1).to_string(index=False))
    sv_ = shadow_validation(p, track)
    print(sv_.groupby("life_stage", observed=True)[["fresh_mV", "shadow_mV", "growth_mV"]].median().round(1))
    sh, allp, best = validate_soh()
    print("best RF feature set:", best)
    print(sh[["feature_set", "cell_set", "model", "LOCO_RMSE_mAh", "LOCO_R2", "conformal_coverage",
              "randomKFold_RMSE_mAh"]].round(3).to_string())
    bt = soh.forecast_backtest(da.discharge_cycles())
    bt.to_csv(paths.METRICS / "forecast_backtest.csv", index=False)
    print(bt.assign(abs_err=bt.eol_error.abs()).groupby("model")[["rmse_hidden_mAh", "abs_err"]].median().round(1))
    sv = validate_soc(p, allp, best)
    print(sv.groupby(["capacity_used", "estimator"])[["MAE_pct", "RMSE_pct", "final_error_pct"]].median().round(2))
    info = dict(seconds=round(time.time() - t0, 1), best_rf_feature_set=best,
                built=pd.Timestamp.now().isoformat(timespec="seconds"))
    (paths.METRICS / "build_info.json").write_text(json.dumps(info, indent=2))
    print(f"calibration + validation done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
