"""First-order lumped thermal model of a cylindrical cell.

    C_th * dT/dt = I^2*R0(T) + V1^2/R1(T) - I*(T + 273.15)*dU/dT - hA*(T - T_amb)

Heat generation counts the irreversible loss in R0 and in the RC-branch resistor (V1^2/R1)
and the reversible (entropic) heat -I*T*dU/dT (I > 0 = discharge). dU/dT is an effective,
SOC-averaged entropic coefficient: it is only identifiable when the data contain both current
directions, because the reversible term changes sign with the current while I^2R does not.
C_th [J/K], hA [W/K] and dU/dT [V/K] are identified from the measured NASA cell temperature.
This is a thermal-RESPONSE model for relative stress comparison, not a thermal-runaway
or safety model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

from .ecm import ECMParams, simulate_voltage, soc_reference


@dataclass
class ThermalParams:
    C_th: float = 40.0   # J/K  (~45 g x ~900 J/kgK)
    hA: float = 0.035    # W/K
    source: str = "literature prior"
    fit_rmse_C: float = float("nan")
    dUdT: float = 0.0    # V/K, effective entropic coefficient (0 = reversible heat neglected)

    @property
    def tau_s(self):
        return self.C_th / self.hA

    def to_dict(self):
        return asdict(self)


def simulate_temperature(th: ThermalParams, t, I, v1, R0, R1, T_amb, T0):
    """Explicit-Euler integration of the lumped model for given electrical states.
    Returns the temperature and the net heat generation (irreversible + reversible, W)."""
    t = np.asarray(t, float)
    I = np.asarray(I, float)
    n = len(t)
    T = np.empty(n)
    T[0] = T0
    q_irr = I ** 2 * R0 + v1 ** 2 / np.maximum(R1, 1e-6)
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        # sub-step for stability when dt is large relative to tau
        nsub = max(1, int(np.ceil(dt / 2.0)))
        h = dt / nsub
        Tk = T[k - 1]
        for _ in range(nsub):
            Tk = Tk + h * (q_irr[k - 1] - I[k - 1] * (Tk + 273.15) * th.dUdT - th.hA * (Tk - T_amb)) / th.C_th
        T[k] = Tk
    return T, q_irr - I * (T + 273.15) * th.dUdT


def identify_thermal(cycles: list[dict], ecm: ECMParams, T_amb_list, source="") -> ThermalParams:
    """Fit C_th and hA (and dU/dT when both current directions are present) to measured cell temperature."""
    prepared = []
    for c, Tamb in zip(cycles, T_amb_list):
        soc = soc_reference(c["t"], c["I"], c["Q"], soc0=c.get("soc0", 1.0))
        T_meas = np.asarray(c["T"], float)
        _, v1 = simulate_voltage(ecm, c["t"], c["I"], soc, T_C=T_meas)
        R0 = ecm.r0_at(T_meas)
        R1 = ecm.r1_at(T_meas, 1.0, c["I"])
        prepared.append((np.asarray(c["t"], float), np.asarray(c["I"], float), v1, R0, R1, float(Tamb), T_meas))
    I_all = np.concatenate([p[1] for p in prepared])
    entropic = bool((I_all > 0.2).any() and (I_all < -0.2).any())

    def make(x):
        return ThermalParams(C_th=np.exp(x[0]), hA=np.exp(x[1]), dUdT=x[2] if entropic else 0.0)

    def res(x):
        th = make(x)
        out = []
        for t, I, v1, R0, R1, Tamb, Tm in prepared:
            Ts, _ = simulate_temperature(th, t, I, v1, R0, R1, Tamb, Tm[0])
            out.append(Ts - Tm)
        return np.concatenate(out)

    x0 = [np.log(40.0), np.log(0.035)] + ([0.0] if entropic else [])
    lb = [np.log(2.0), np.log(1e-3)] + ([-2e-3] if entropic else [])
    ub = [np.log(500.0), np.log(2.0)] + ([2e-3] if entropic else [])
    sol = least_squares(res, x0, bounds=(lb, ub), max_nfev=200)
    th = make(sol.x)
    th.source = source
    th.fit_rmse_C = float(np.sqrt(np.mean(sol.fun ** 2)))
    th.C_th, th.hA, th.dUdT = float(th.C_th), float(th.hA), float(th.dUdT)
    return th
