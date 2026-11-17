"""Incremental capacity analysis (ICA, dQ/dV) on the CC phase of NASA charges.

ICA turns a voltage plateau of the full-cell OCV into a peak of dQ/dV. As the cell ages,
loss of lithium inventory and loss of active material lower the peak, and rising polarisation
shifts it to higher voltage. The CC charge at 1.5 A (~C/1.3) smears the peaks, so the curves are
smoothed (Savitzky-Golay) on a uniform voltage grid; the resulting peak height, peak voltage
and window charge are used as physically interpretable health indicators.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

V_GRID = np.round(np.arange(3.70, 4.1951, 0.005), 4)
WINDOW = (3.80, 4.10)


def ica_curve(t, V, I_charge, v_grid=V_GRID, window=11, poly=2):
    """dQ/dV (Ah/V) on ``v_grid`` from one CC-CV charge.  ``I_charge`` > 0 while charging.
    Returns (v_grid, dqdv) or None if the CC segment is too short."""
    t = np.asarray(t, float)
    V = np.asarray(V, float)
    I = np.asarray(I_charge, float)
    idx = np.where(I > 1.2)[0]
    if idx.size < 20:
        return None
    k0, k1 = idx[0], idx[-1]
    tt, vv, ii = t[k0:k1 + 1], V[k0:k1 + 1], I[k0:k1 + 1]
    q = np.concatenate([[0.0], np.cumsum(0.5 * (ii[1:] + ii[:-1]) * np.diff(tt))]) / 3600.0
    vm = np.maximum.accumulate(vv)                     # voltage is monotone in CC charge
    keep = np.concatenate([[True], np.diff(vm) > 1e-5])
    vm, qm = vm[keep], q[keep]
    if vm.size < 10:
        return None
    qg = np.interp(v_grid, vm, qm, left=np.nan, right=np.nan)
    dq = np.gradient(qg, v_grid)
    valid = np.isfinite(dq)
    if valid.sum() < window + 2:
        return None
    out = np.full_like(dq, np.nan)
    out[valid] = savgol_filter(dq[valid], window, poly)
    return v_grid, out


def ica_features(v, dq) -> dict:
    """Peak height (Ah/V), peak voltage (V) and charge inside the 3.80-4.10 V window (Ah)."""
    if v is None or dq is None or not np.isfinite(dq).any():
        return dict(ic_peak_AhV=np.nan, ic_peak_V=np.nan, ic_area_Ah=np.nan)
    j = int(np.nanargmax(dq))
    w = (v >= WINDOW[0]) & (v <= WINDOW[1]) & np.isfinite(dq)
    area = float(np.trapezoid(dq[w], v[w])) if w.sum() > 2 else np.nan
    return dict(ic_peak_AhV=float(dq[j]), ic_peak_V=float(v[j]), ic_area_Ah=area)
