"""NASA PCoE Li-ion Battery Aging Data Set -> tidy parquet tables.

Source archive: "5. Battery Data Set.zip" (NASA Ames Prognostics Center of Excellence,
B. Saha and K. Goebel, 2007).  The archive contains six inner zips with 34 unique
18650 cells (B0025-B0028 appear twice with byte-identical files).

Outputs (data/processed):
    cells.parquet            one row per cell (test conditions + summary)
    discharge_cycles.parquet one row per discharge (capacity, SOH, health indicators,
                             preceding-charge features, last EIS Re/Rct)
    charge_cycles.parquet    one row per charge (CC/CV durations, Ah, temperatures)
    impedance.parquet        one row per EIS test (Re, Rct)
    eis_spectra.parquet      Nyquist points (Battery_impedance) for every EIS test
    ts_discharge.parquet     sample-level discharge time series (all cells)
    ts_charge.parquet        sample-level charge time series (all cells)

Sign convention used everywhere downstream: current I > 0 means DISCHARGE.
NASA stores discharge current as negative, so I = -Current_measured.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from . import paths
from .ica import ica_curve, ica_features

RATED_CAPACITY_AH = 2.0  # NASA README: EOL defined as fade "from 2Ahr to 1.4Ahr"

# Test conditions transcribed from the README files shipped inside each archive.
CELL_CONDITIONS: dict[str, dict] = {}


def _add(cells, archive, amb, profile, i_dis, cutoffs, eol, note=""):
    for c, v in zip(cells, cutoffs):
        CELL_CONDITIONS[c] = dict(
            archive=archive, ambient_nominal_C=amb, discharge_profile=profile,
            discharge_current_A=i_dis, cutoff_V=v, eol_criterion=eol, note=note,
        )


_add(["B0005", "B0006", "B0007", "B0018"], "1. BatteryAgingARC-FY08Q4", "24", "CC 2 A", 2.0,
     [2.7, 2.5, 2.2, 2.5], "30% fade (2.0 -> 1.4 Ah)")
_add(["B0025", "B0026", "B0027", "B0028"], "3. BatteryAgingARC_25-44", "24",
     "Square wave 4 A, 0.05 Hz, 50% duty", 4.0, [2.0, 2.2, 2.5, 2.7], "not stated (short test)")
_add(["B0029", "B0030", "B0031", "B0032"], "3. BatteryAgingARC_25-44", "43", "CC 4 A", 4.0,
     [2.0, 2.2, 2.5, 2.7], "not stated (short test)")
_add(["B0033", "B0034"], "3. BatteryAgingARC_25-44", "24", "CC 4 A", 4.0, [2.0, 2.2],
     "20% fade (1.6 Ah)")
_add(["B0036"], "3. BatteryAgingARC_25-44", "24", "CC 2 A", 2.0, [2.7], "20% fade (1.6 Ah)")
_add(["B0038", "B0039", "B0040"], "3. BatteryAgingARC_25-44", "24 & 44", "Multiple loads 1/2/4 A",
     np.nan, [2.2, 2.5, 2.7], "20% fade (1.6 Ah)")
_add(["B0041", "B0042", "B0043", "B0044"], "3. BatteryAgingARC_25-44", "4", "CC 4 A and 1 A", np.nan,
     [2.0, 2.2, 2.5, 2.7], "30% fade (1.4 Ah)",
     "README: several discharge runs with very low capacity, not analysed by NASA")
_add(["B0045", "B0046", "B0047", "B0048"], "4. BatteryAgingARC_45_46_47_48", "4", "CC 1 A", 1.0,
     [2.0, 2.2, 2.5, 2.7], "30% fade (1.4 Ah)",
     "README: several discharge runs with very low capacity, not analysed by NASA")
_add(["B0049", "B0050", "B0051", "B0052"], "5. BatteryAgingARC_49_50_51_52", "4", "CC 2 A", 2.0,
     [2.0, 2.2, 2.5, 2.7], "until control software crashed",
     "README: software crash; several runs with very low capacity and voltage")
_add(["B0053", "B0054", "B0055", "B0056"], "6. BatteryAgingARC_53_54_55_56", "4", "CC 2 A", 2.0,
     [2.0, 2.2, 2.5, 2.7], "30% fade (1.4 Ah)",
     "README: several discharge runs with very low capacity, not analysed by NASA")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def extract_archive(zip_path: Path = paths.NASA_ZIP, out: Path = paths.NASA_EXTRACT) -> Path:
    """Extract outer and inner zips (idempotent). Returns the folder holding the 6 groups."""
    root = out / "5. Battery Data Set"
    if not root.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
    for inner in sorted(root.glob("*.zip")):
        target = root / inner.stem
        if not target.exists():
            with zipfile.ZipFile(inner) as z:
                z.extractall(target)
    return root


def _ts(vec) -> pd.Timestamp:
    y, mo, d, h, mi, s = [float(x) for x in np.ravel(vec)]
    sec = int(s)
    return pd.Timestamp(int(y), int(mo), int(d), int(h), int(mi), sec, int(round((s - sec) * 1e6)) % 1000000)


def _first_crossing(t, v, level):
    idx = np.where(v <= level)[0]
    return float(t[idx[0]]) if idx.size else np.nan


def _discharge_features(t, V, I, T, cutoff_V):
    """Health indicators from one discharge (I > 0 = discharge)."""
    loaded = I > 0.2
    f = {}
    if loaded.sum() < 3:
        return None
    i0 = int(np.argmax(loaded))
    t_load0 = t[i0]
    # rest voltage just before load and first loaded sample -> DC-resistance proxy
    v_rest = V[i0 - 1] if i0 > 0 else V[0]
    f["v_rest_start"] = float(v_rest)
    f["r_dc_first_sample_ohm"] = float((v_rest - V[i0]) / max(I[i0], 1e-3))
    # integrate current up to the cutoff (or to the end of loading)
    q = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]).clip(min=0) * np.diff(t))]) / 3600.0
    e = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] * V[1:] + I[:-1] * V[:-1]).clip(min=0) * np.diff(t))]) / 3600.0
    last_loaded = int(np.where(loaded)[0][-1])
    f["ah_integrated"] = float(q[last_loaded])
    f["energy_Wh"] = float(e[last_loaded])
    f["duration_load_s"] = float(t[last_loaded] - t_load0)
    f["mean_current_A"] = float(I[loaded].mean())
    f["max_current_A"] = float(I.max())
    f["mean_voltage_loaded_V"] = float(V[loaded].mean())
    f["min_voltage_V"] = float(V.min())
    tl = t - t_load0
    for lvl in (3.8, 3.6, 3.4):
        tt = _first_crossing(tl[i0:], V[i0:], lvl)
        f[f"t_to_{str(lvl).replace('.', 'p')}V_s"] = tt
    f["T_start_C"] = float(T[max(i0 - 1, 0)])
    f["T_max_C"] = float(T.max())
    f["T_rise_C"] = float(T.max() - T[max(i0 - 1, 0)])
    f["t_to_Tmax_s"] = float(t[int(np.argmax(T))] - t_load0)
    # voltage recovery after load removal (relaxation) -> overpotential proxy
    if last_loaded < len(V) - 1:
        f["v_recovery_V"] = float(V[-1] - V[last_loaded])
    else:
        f["v_recovery_V"] = np.nan
    f["n_samples"] = int(len(t))
    f["median_dt_s"] = float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    return f


def _charge_features(t, V, I, T):
    """Charging-phase health indicators (CC-CV, I > 0 here = charging)."""
    f = {}
    good = np.arange(len(t)) >= 2  # first two samples hold a known switching transient
    tt, vv, ii, TT = t[good], V[good], I[good], T[good]
    if len(tt) < 10:
        return None
    cc = ii > 1.2
    if cc.sum() < 2:
        return None
    k0 = int(np.argmax(cc))
    reach = np.where((vv >= 4.195) & (np.arange(len(vv)) >= k0))[0]
    k1 = int(reach[0]) if reach.size else int(np.where(cc)[0][-1])
    dq = np.concatenate([[0.0], np.cumsum(0.5 * (ii[1:] + ii[:-1]).clip(min=0) * np.diff(tt))]) / 3600.0
    f["t_cc_s"] = float(tt[k1] - tt[k0])
    f["q_cc_Ah"] = float(dq[k1] - dq[k0])
    end = np.where((ii < 0.05) & (np.arange(len(ii)) > k1))[0]
    k2 = int(end[0]) if end.size else len(tt) - 1
    f["t_cv_s"] = float(tt[k2] - tt[k1])
    f["q_cv_Ah"] = float(dq[k2] - dq[k1])
    f["q_charge_Ah"] = float(dq[k2] - dq[k0])
    f["v_start_charge_V"] = float(vv[k0])
    f["T_max_charge_C"] = float(TT.max())
    f["T_rise_charge_C"] = float(TT.max() - TT[k0])
    f["charge_duration_s"] = float(tt[-1] - tt[0])
    r = ica_curve(tt, vv, ii)                       # incremental-capacity indicators
    f.update(ica_features(*(r if r is not None else (None, None))))
    return f


def parse_cell(mat_path: Path, cell: str):
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    cycles = np.atleast_1d(m[cell].cycle)
    cond = CELL_CONDITIONS[cell]
    dis_rows, chg_rows, imp_rows, spec_rows = [], [], [], []
    ts_d, ts_c = [], []
    n_d = n_c = n_i = 0
    for op_index, cyc in enumerate(cycles):
        typ = str(cyc.type)
        start = _ts(cyc.time)
        amb = float(cyc.ambient_temperature)
        d = cyc.data
        if typ == "discharge":
            t = np.asarray(d.Time, float)
            if t.size < 5:
                continue
            V = np.asarray(d.Voltage_measured, float)
            I = -np.asarray(d.Current_measured, float)
            T = np.asarray(d.Temperature_measured, float)
            feats = _discharge_features(t, V, I, T, cond["cutoff_V"])
            if feats is None:
                continue
            n_d += 1
            cap = float(np.asarray(d.Capacity, float).ravel()[0]) if np.size(d.Capacity) else np.nan
            row = dict(cell_id=cell, discharge_index=n_d, op_index=op_index, start_time=start,
                       ambient_C=amb, capacity_Ah=cap, soh_pct=100.0 * cap / RATED_CAPACITY_AH, **feats)
            dis_rows.append(row)
            ts_d.append(pd.DataFrame(dict(
                cell_id=cell, discharge_index=np.int16(n_d), t_s=t.astype(np.float32),
                V=V.astype(np.float32), I=I.astype(np.float32), T=T.astype(np.float32),
                I_load=np.asarray(d.Current_load, float).astype(np.float32),
                V_load=np.asarray(d.Voltage_load, float).astype(np.float32), ambient_C=np.float32(amb))))
        elif typ == "charge":
            t = np.asarray(d.Time, float)
            if t.size < 10:
                continue
            V = np.asarray(d.Voltage_measured, float)
            Ic = np.asarray(d.Current_measured, float)  # positive = charging in NASA files
            T = np.asarray(d.Temperature_measured, float)
            feats = _charge_features(t, V, Ic, T)
            n_c += 1
            row = dict(cell_id=cell, charge_index=n_c, op_index=op_index, start_time=start, ambient_C=amb)
            row.update(feats or {})
            row["valid"] = feats is not None
            chg_rows.append(row)
            ts_c.append(pd.DataFrame(dict(
                cell_id=cell, charge_index=np.int16(n_c), t_s=t.astype(np.float32),
                V=V.astype(np.float32), I=(-Ic).astype(np.float32), T=T.astype(np.float32),
                ambient_C=np.float32(amb))))
        elif typ == "impedance":
            n_i += 1
            re = float(np.real(np.ravel(d.Re)[0])) if np.size(d.Re) else np.nan
            rct = float(np.real(np.ravel(d.Rct)[0])) if np.size(d.Rct) else np.nan
            imp_rows.append(dict(cell_id=cell, impedance_index=n_i, op_index=op_index, start_time=start,
                                 ambient_C=amb, Re_ohm=re, Rct_ohm=rct))
            z = np.atleast_1d(np.asarray(d.Battery_impedance))
            if z.size:
                spec_rows.append(pd.DataFrame(dict(
                    cell_id=cell, impedance_index=np.int16(n_i), k=np.arange(z.size, dtype=np.int16),
                    Z_real_ohm=np.real(z).astype(np.float32), Z_imag_ohm=np.imag(z).astype(np.float32))))
    return (pd.DataFrame(dis_rows), pd.DataFrame(chg_rows), pd.DataFrame(imp_rows),
            pd.concat(spec_rows, ignore_index=True) if spec_rows else pd.DataFrame(),
            pd.concat(ts_d, ignore_index=True) if ts_d else pd.DataFrame(),
            pd.concat(ts_c, ignore_index=True) if ts_c else pd.DataFrame())


def _attach_context(dis: pd.DataFrame, chg: pd.DataFrame, imp: pd.DataFrame) -> pd.DataFrame:
    """For every discharge attach (a) the immediately preceding charge features and
    (b) the most recent plausible EIS result (Re, Rct) measured BEFORE the discharge."""
    out = []
    chg_cols = ["t_cc_s", "q_cc_Ah", "t_cv_s", "q_cv_Ah", "q_charge_Ah", "v_start_charge_V",
                "T_max_charge_C", "T_rise_charge_C", "ic_peak_AhV", "ic_peak_V", "ic_area_Ah"]
    for cell, g in dis.groupby("cell_id", sort=False):
        g = g.sort_values("op_index").copy()
        c = chg[(chg.cell_id == cell) & chg["valid"]].sort_values("op_index")
        i = imp[(imp.cell_id == cell)].sort_values("op_index")
        i = i[(i.Re_ohm > 0) & (i.Re_ohm < 0.5) & (i.Rct_ohm > 0) & (i.Rct_ohm < 1.0)]
        if len(c):
            g = pd.merge_asof(g, c[["op_index"] + chg_cols].rename(columns={"op_index": "charge_op_index"}),
                              left_on="op_index", right_on="charge_op_index", direction="backward")
        else:
            for col in chg_cols + ["charge_op_index"]:
                g[col] = np.nan
        if len(i):
            g = pd.merge_asof(g, i[["op_index", "Re_ohm", "Rct_ohm"]].rename(columns={"op_index": "eis_op_index"}),
                              left_on="op_index", right_on="eis_op_index", direction="backward")
        else:
            g["Re_ohm"] = np.nan
            g["Rct_ohm"] = np.nan
            g["eis_op_index"] = np.nan
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build(verbose: bool = True) -> dict:
    root = extract_archive()
    files = {}
    for mat in sorted(root.rglob("B00*.mat")):
        cell = mat.stem
        if cell in files:  # B0025-B0028 duplicated across archives 2 and 3
            if sha256(mat) != sha256(files[cell]):
                raise RuntimeError(f"Duplicate {cell} differs between archives")
            continue
        files[cell] = mat
    parts = {k: [] for k in ("dis", "chg", "imp", "spec", "ts_d", "ts_c")}
    for cell, mat in sorted(files.items()):
        out = parse_cell(mat, cell)
        if verbose:
            print(f"{cell}: {len(out[0]):4d} discharges, {len(out[1]):4d} charges, {len(out[2]):4d} EIS   ({mat.parent.name})")
        for k, v in zip(parts, out):
            parts[k].append(v)
    D, C, I, S, TD, TC = (parts[k] for k in parts)
    dis = pd.concat(D, ignore_index=True)
    chg = pd.concat(C, ignore_index=True)
    imp = pd.concat(I, ignore_index=True)
    spec = pd.concat([s for s in S if len(s)], ignore_index=True)
    tsd = pd.concat(TD, ignore_index=True)
    tsc = pd.concat(TC, ignore_index=True)
    dis = _attach_context(dis, chg, imp)

    # Data-quality flags (kept, not deleted: the GUI lets users include/exclude them)
    dis["flag_low_capacity"] = dis["capacity_Ah"] < 0.5
    dis["flag_capacity_mismatch"] = (dis["ah_integrated"] - dis["capacity_Ah"]).abs() > 0.15
    dis["usable_for_soh"] = (~dis["flag_low_capacity"]) & dis["capacity_Ah"].between(0.5, 2.3)

    # Cell table
    rows = []
    for cell, cond in sorted(CELL_CONDITIONS.items()):
        g = dis[dis.cell_id == cell]
        if g.empty:
            continue
        ok = g[g.usable_for_soh]
        rows.append(dict(
            cell_id=cell, **cond,
            n_discharge=int(len(g)), n_charge=int((chg.cell_id == cell).sum()),
            n_impedance=int((imp.cell_id == cell).sum()),
            ambient_values_C=", ".join(str(int(x)) for x in sorted(g.ambient_C.unique())),
            mean_load_current_A=float(g.mean_current_A.median()),
            first_capacity_Ah=float(ok.capacity_Ah.iloc[0]) if len(ok) else np.nan,
            last_capacity_Ah=float(ok.capacity_Ah.iloc[-1]) if len(ok) else np.nan,
            min_soh_pct=float(ok.soh_pct.min()) if len(ok) else np.nan,
            n_flagged_low_capacity=int(g.flag_low_capacity.sum()),
            first_test=str(g.start_time.min()), last_test=str(g.start_time.max()),
            source_file=str(files[cell].relative_to(paths.NASA_EXTRACT)),
        ))
    cells = pd.DataFrame(rows)
    cells["discharge_current_A"] = cells["discharge_current_A"].astype(float)

    out = paths.PROCESSED
    for col in ("cell_id",):
        for df in (tsd, tsc, spec):
            df[col] = df[col].astype("category")
    cells.to_parquet(out / "cells.parquet", index=False)
    dis.to_parquet(out / "discharge_cycles.parquet", index=False)
    chg.to_parquet(out / "charge_cycles.parquet", index=False)
    imp.to_parquet(out / "impedance.parquet", index=False)
    spec.to_parquet(out / "eis_spectra.parquet", index=False)
    tsd.to_parquet(out / "ts_discharge.parquet", index=False, compression="zstd")
    tsc.to_parquet(out / "ts_charge.parquet", index=False, compression="zstd")

    # Small CSV sample for quick inspection without Python
    tsd[tsd.cell_id == "B0005"].query("discharge_index <= 2").to_csv(paths.SAMPLES / "B0005_discharge_1-2.csv", index=False)
    dis[dis.cell_id == "B0005"].to_csv(paths.SAMPLES / "B0005_discharge_cycles.csv", index=False)

    summary = dict(
        built_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        cells=int(cells.shape[0]), discharges=int(len(dis)), charges=int(len(chg)), eis_tests=int(len(imp)),
        ts_discharge_rows=int(len(tsd)), ts_charge_rows=int(len(tsc)), eis_points=int(len(spec)),
        archive_sha256=sha256(paths.NASA_ZIP) if paths.NASA_ZIP.exists() else None,
    )
    (paths.METADATA / "build_summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
