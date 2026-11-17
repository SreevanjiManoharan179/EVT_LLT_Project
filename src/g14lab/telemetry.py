"""KU Leuven "BEV energy dynamics dataset" (doi:10.48804/8KPDTW, CC BY 4.0) -> parquet.

Two Tesla Model Y vehicles (BEV1, BEV2), 1 s CAN-bus data decoded by the dataset authors.
We use the fast-charging, slow-charging and driving session CSVs (parking sessions and the
raw .MF4 example stay in the raw zip, unprocessed).

Kept signals (renamed):            source column
    soc_bms_pct   SOCave292          BMS average SOC (%), BMS-REPORTED, not measured truth
    v_pack_V      BattVoltage132     pack terminal voltage
    i_pack_A      RawBattCurrent132  pack current, >0 = discharge (dataset convention)
    t_min_C       BMSminPackTemperature
    t_max_C       BMSmaxPackTemperature
    t_amb_C       VCFRONT_tempAmbient
    speed_kph     DI_uiSpeed
    p_ac_kW       ChargeLinePower264 (AC on-board charger line power)
    fc_i_A, fc_v_V FC_dcCurrent, FC_dcVoltage (DC fast charger side)
    bms_diss_kW   BMSdissipation312
    p_dis_lim_kW, p_regen_lim_kW  BMS_maxDischargePower, BMS_maxRegenPower
GPS columns are deliberately dropped (not needed; avoids redistributing location traces).
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from . import paths

_ZIP_NAME = "BEV energy dynamic data_V2.zip"
# Preferred location inside the project; falls back to the parent folder where it was downloaded.
ZIP = next((p for p in (paths.RAW / "kuleuven_bev" / _ZIP_NAME, paths.ROOT.parent / _ZIP_NAME) if p.exists()),
           paths.RAW / "kuleuven_bev" / _ZIP_NAME)
COLMAP = {
    "Timestamp": "timestamp", "SOCave292": "soc_bms_pct", "BattVoltage132": "v_pack_V",
    "RawBattCurrent132": "i_pack_A", "BMSminPackTemperature": "t_min_C",
    "BMSmaxPackTemperature": "t_max_C", "VCFRONT_tempAmbient": "t_amb_C", "DI_uiSpeed": "speed_kph",
    "ChargeLinePower264": "p_ac_kW", "FC_dcCurrent": "fc_i_A", "FC_dcVoltage": "fc_v_V",
    "BMSdissipation312": "bms_diss_kW", "BMS_maxDischargePower": "p_dis_lim_kW",
    "BMS_maxRegenPower": "p_regen_lim_kW", "UI_Range": "ui_range_km",
}
FOLDERS = {"fast charging sessions": "fast_charge", "slow charging sessions": "slow_charge",
           "driving sessions": "drive"}


def _read(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    head = pd.read_csv(io.BytesIO(z.read(name)), nrows=0).columns
    use = [c for c in COLMAP if c in head]
    df = pd.read_csv(io.BytesIO(z.read(name)), usecols=use)
    df = df.rename(columns=COLMAP)
    for c in COLMAP.values():
        if c not in df.columns:
            df[c] = np.nan
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    num = [c for c in COLMAP.values() if c != "timestamp"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").ffill().bfill()
    return df


def _summ(df: pd.DataFrame, sid, veh, kind, fname) -> dict:
    t = (df.timestamp - df.timestamp.iloc[0]).dt.total_seconds().to_numpy()
    i = df.i_pack_A.to_numpy(float)
    v = df.v_pack_V.to_numpy(float)
    p = v * i / 1000.0
    dt_raw = np.diff(t, prepend=t[0])
    dt = np.clip(dt_raw, 0, 10)  # ignore logging gaps > 10 s in integrals
    coverage = float(dt.sum() / max(t[-1], 1.0))  # share of the session actually logged
    ah = float(np.nansum(i * dt) / 3600.0)
    kwh = float(np.nansum(p * dt) / 3600.0)
    dist = float(np.nansum(df.speed_kph.fillna(0).to_numpy() * dt) / 3600.0)
    soc0, soc1 = float(df.soc_bms_pct.iloc[0]), float(df.soc_bms_pct.iloc[-1])
    return dict(session_id=sid, vehicle=veh, kind=kind, file=fname, start=df.timestamp.iloc[0],
                duration_min=float(t[-1] / 60.0), n_rows=int(len(df)),
                median_dt_s=float(np.median(np.diff(t))) if len(t) > 1 else np.nan,
                soc_start_pct=soc0, soc_end_pct=soc1, d_soc_pct=soc1 - soc0,
                ah_net_discharge=ah, kwh_net_discharge=kwh, distance_km=dist,
                i_max_charge_A=float(-np.nanmin(i)), i_max_discharge_A=float(np.nanmax(i)),
                p_max_charge_kW=float(-np.nanmin(p)), v_min_V=float(np.nanmin(v)), v_max_V=float(np.nanmax(v)),
                t_pack_start_C=float(df.t_max_C.iloc[0]), t_pack_max_C=float(df.t_max_C.max()),
                t_amb_mean_C=float(df.t_amb_C.mean()),
                # Pack capacity implied by BMS: |Ah| / |dSOC|. A PROXY that inherits BMS SOC error.
                logged_fraction=coverage,
                q_implied_Ah=float(abs(ah) / (abs(soc1 - soc0) / 100.0)) if (abs(soc1 - soc0) >= 20 and coverage > 0.9) else np.nan,
                kwh_per_100km=float(100 * kwh / dist) if (kind == "drive" and dist > 5) else np.nan)


def build(verbose=True, max_drive_sessions: int | None = None) -> dict:
    z = zipfile.ZipFile(ZIP)
    names = [i.filename for i in z.infolist() if i.filename.endswith(".csv")]
    ts_parts = {k: [] for k in FOLDERS.values()}
    summ = []
    for folder, kind in FOLDERS.items():
        files = sorted(n for n in names if f"/{folder}/" in n)
        if kind == "drive" and max_drive_sessions:
            files = files[:max_drive_sessions]
        for n in files:
            fname = n.split("/")[-1]
            sid = fname[:-4]
            veh = fname.split("_")[0]
            try:
                df = _read(z, n)
            except Exception as exc:  # pragma: no cover - logged, not fatal
                print("skip", fname, exc)
                continue
            if len(df) < 30:
                continue
            summ.append(_summ(df, sid, veh, kind, fname))
            df.insert(0, "session_id", sid)
            df.insert(1, "vehicle", veh)
            df["t_s"] = (df.timestamp - df.timestamp.iloc[0]).dt.total_seconds().astype("float32")
            for c in df.columns:
                if df[c].dtype == "float64":
                    df[c] = df[c].astype("float32")
            ts_parts[kind].append(df)
        if verbose:
            print(f"{kind}: {len(ts_parts[kind])} sessions")
    out = paths.PROCESSED
    for kind, parts in ts_parts.items():
        if parts:
            d = pd.concat(parts, ignore_index=True)
            d["session_id"] = d["session_id"].astype("category")
            d["vehicle"] = d["vehicle"].astype("category")
            d.to_parquet(out / f"bev_ts_{kind}.parquet", index=False, compression="zstd")
    s = pd.DataFrame(summ)
    s.to_parquet(out / "bev_sessions.parquet", index=False)
    info = dict(sessions=int(len(s)), by_kind=s.kind.value_counts().to_dict(),
                by_vehicle=s.vehicle.value_counts().to_dict(),
                rows={k: int(sum(len(p) for p in v)) for k, v in ts_parts.items()})
    if verbose:
        print(info)
    return info


if __name__ == "__main__":
    build()
