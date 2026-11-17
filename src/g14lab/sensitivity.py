"""One-at-a-time local sensitivity of experiment outputs to twin parameters (tornado analysis)."""
from __future__ import annotations

import copy

import pandas as pd

from .ecm import ECMParams
from .simulator import TwinState, run_protocol
from .thermal import ThermalParams

PARAMS = {
    "R0": ("ecm", "R0", "Ohmic resistance R0"),
    "R1": ("ecm", "R1", "Polarisation resistance R1 (discharge)"),
    "R1_chg": ("ecm", "R1_chg", "Polarisation resistance R1 (charge)"),
    "tau1": ("ecm", "tau1", "Time constant τ1"),
    "Q_Ah": ("ecm", "Q_Ah", "Capacity Q"),
    "Ea": ("ecm", "Ea", "Activation energy Ea"),
    "C_th": ("thermal", "C_th", "Heat capacity C_th"),
    "hA": ("thermal", "hA", "Heat transfer hA"),
    "dUdT": ("thermal", "dUdT", "Entropic coefficient dU/dT"),
}
OUTPUTS = {"Ah_discharged": "Charge moved (Ah)", "T_max_C": "Peak temperature (°C)", "Wh_net": "Net energy (Wh)",
           "duration_min": "Duration (min)"}


def tornado(p: ECMParams, th: ThermalParams, state: TwinState, steps: list[dict], rel: float = 0.2, dt: float = 2.0):
    """Run the protocol with each parameter at (1 - rel) and (1 + rel); returns a tidy DataFrame."""
    _, base = run_protocol(p, th, state, steps, dt=dt)
    rows = []
    for key, (where, attr, label) in PARAMS.items():
        for sign in (-1, 1):
            pp, tt = copy.deepcopy(p), copy.deepcopy(th)
            f = 1 + sign * rel
            if key == "Ea":
                pp.Ea_R0 *= f
                pp.Ea_R1 *= f
            elif key == "R1_chg" and pp.R1_chg is None:
                continue                                   # symmetric twin: no separate charge resistance
            elif where == "ecm":
                setattr(pp, attr, getattr(pp, attr) * f)
                pp._pchip = None
            else:
                setattr(tt, attr, getattr(tt, attr) * f)
            _, s = run_protocol(pp, tt, state, copy.deepcopy(steps), dt=dt)
            for out in OUTPUTS:
                rows.append(dict(parameter=label, change=f"{sign * rel * 100:+.0f} %", output=out,
                                 delta=float(s[out] - base[out]), base=float(base[out])))
    return pd.DataFrame(rows), base
