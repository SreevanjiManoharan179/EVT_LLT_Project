"""Loaders for the processed tables (pure pandas; the GUI adds Streamlit caching)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from . import paths


@lru_cache(maxsize=None)
def cells() -> pd.DataFrame:
    return pd.read_parquet(paths.PROCESSED / "cells.parquet")


@lru_cache(maxsize=None)
def discharge_cycles() -> pd.DataFrame:
    return pd.read_parquet(paths.PROCESSED / "discharge_cycles.parquet")


@lru_cache(maxsize=None)
def charge_cycles() -> pd.DataFrame:
    return pd.read_parquet(paths.PROCESSED / "charge_cycles.parquet")


@lru_cache(maxsize=None)
def impedance() -> pd.DataFrame:
    return pd.read_parquet(paths.PROCESSED / "impedance.parquet")


def eis_spectra(cell: str) -> pd.DataFrame:
    return pd.read_parquet(paths.PROCESSED / "eis_spectra.parquet", filters=[("cell_id", "==", cell)])


def _ts(name: str, cell: str, idx_col: str, idx=None) -> pd.DataFrame:
    filt = ds.field("cell_id") == cell
    if idx is not None:
        if np.isscalar(idx):
            filt = filt & (ds.field(idx_col) == int(idx))
        else:
            filt = filt & ds.field(idx_col).isin([int(i) for i in idx])
    t = ds.dataset(paths.PROCESSED / name, format="parquet").to_table(filter=filt)
    df = t.to_pandas()
    df["cell_id"] = df["cell_id"].astype(str)
    return df


def discharge_ts(cell: str, idx=None) -> pd.DataFrame:
    return _ts("ts_discharge.parquet", cell, "discharge_index", idx)


def charge_ts(cell: str, idx=None) -> pd.DataFrame:
    return _ts("ts_charge.parquet", cell, "charge_index", idx)


def cycle_dict(cell: str, discharge_index: int) -> dict:
    """Arrays for one measured discharge, ready for identification/estimation."""
    ts = discharge_ts(cell, discharge_index)
    row = discharge_cycles().query("cell_id == @cell and discharge_index == @discharge_index").iloc[0]
    return dict(cell=cell, idx=int(discharge_index), t=ts.t_s.to_numpy(float), V=ts.V.to_numpy(float),
                I=ts.I.to_numpy(float), T=ts["T"].to_numpy(float),
                # Q used to normalise SOC = Ah integrated from the measured current up to the end
                # of the load (consistent with the samples); NASA's own field kept as Q_nasa.
                Q=float(row.ah_integrated), Q_nasa=float(row.capacity_Ah),
                ambient=float(row.ambient_C))


def charge_cycle_dict(cell: str, charge_index: int, Q_Ah: float, n_target: int = 400) -> dict:
    """Arrays for one measured CC-CV charge (I < 0), starting from SOC 0 = the cut-off of the
    preceding discharge whose integrated capacity is ``Q_Ah``. The first three samples hold a
    tester switching transient and are dropped; long records are decimated to ~n_target samples."""
    ts = charge_ts(cell, charge_index).iloc[3:]
    step = max(1, len(ts) // n_target)
    ts = ts.iloc[::step]
    row = charge_cycles().query("cell_id == @cell and charge_index == @charge_index").iloc[0]
    t = ts.t_s.to_numpy(float)
    return dict(cell=cell, idx=int(charge_index), kind="charge", t=t - t[0], V=ts.V.to_numpy(float),
                I=ts.I.to_numpy(float), T=ts["T"].to_numpy(float), Q=float(Q_Ah), Q_nasa=float(Q_Ah),
                soc0=0.0, ambient=float(row.ambient_C))


def following_charge_index(cell: str, discharge_index: int):
    """charge_index of the charge that directly follows a discharge (None if not found)."""
    d = discharge_cycles()
    c = charge_cycles()
    r = d[(d.cell_id == cell) & (d.discharge_index == discharge_index)]
    if r.empty:
        return None
    op = int(r.op_index.iloc[0])
    nxt = c[(c.cell_id == cell) & (c.op_index > op) & c["valid"]].sort_values("op_index")
    nd = d[(d.cell_id == cell) & (d.op_index > op)].op_index.min()
    if nxt.empty or (np.isfinite(nd) and nxt.op_index.iloc[0] > nd):
        return None
    return int(nxt.charge_index.iloc[0])


def eis_ohmic_resistance(cell: str, n_first: int = 3):
    """Median EIS high-frequency (ohmic) resistance Re of the cell's first EIS measurements, or None."""
    imp = impedance()
    im = imp[(imp.cell_id == cell) & imp.Re_ohm.between(0.01, 0.5)].sort_values("op_index")
    return float(im.Re_ohm.iloc[:n_first].median()) if len(im) else None


def charge_after(cell: str, discharge_index: int):
    """The CC-CV charge that follows a discharge as a cycle dict (SOC 0 = that discharge's cut-off), or None."""
    k = following_charge_index(cell, int(discharge_index))
    if k is None:
        return None
    return charge_cycle_dict(cell, k, cycle_dict(cell, int(discharge_index))["Q"])
