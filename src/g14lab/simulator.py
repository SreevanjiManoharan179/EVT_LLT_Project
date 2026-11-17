"""Coupled electro-thermal digital-twin simulator (1-RC ECM + lumped thermal model).

The simulator runs a PROTOCOL (list of steps) on a calibrated cell twin:
    {"type": "cc",   "current_A": +2.0, "until_V": 2.7, "max_s": 7200}   (I>0 discharge)
    {"type": "cc",   "current_A": -1.5, "until_V": 4.2, "max_s": 7200}   (charge)
    {"type": "cv",   "voltage_V": 4.2, "until_A": 0.02, "max_s": 7200}
    {"type": "rest", "duration_s": 600}
    {"type": "pulse","high_A": 4.0, "low_A": 0.0, "period_s": 20, "duty": 0.5, "until_V": 2.5, "max_s": 7200}
    {"type": "profile", "t_s": [...], "current_A": [...]}               (replay any current)

Electro-thermal coupling: R0 and R1 are re-evaluated every step at the simulated cell
temperature (Arrhenius); the heat they dissipate plus the reversible (entropic) heat
-I*T*dU/dT drives the temperature.
Ageing: Q = Q_fresh * SOH, and R0/R1 scaled by r_scale (resistance growth factor).
Everything produced here is SIMULATED.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ecm import ECMParams
from .thermal import ThermalParams


@dataclass
class TwinState:
    soc0: float = 1.0
    T0_C: float = 25.0
    T_amb_C: float = 25.0
    soh: float = 1.0          # capacity ratio (0-1) relative to the calibrated capacity
    r_scale: float = 1.0      # resistance growth factor
    v_max: float = 4.2
    v_min: float = 2.7


def run_protocol(ecm: ECMParams, th: ThermalParams, state: TwinState, steps: list[dict],
                 dt: float = 1.0, max_total_s: float = 6 * 3600) -> tuple[pd.DataFrame, dict]:
    Q = ecm.Q_Ah * state.soh
    soc, v1, T = state.soc0, 0.0, state.T0_C
    t = 0.0
    rec = {k: [] for k in ("t_s", "step", "I_A", "V", "SOC", "V1", "T_C", "R0", "Qgen_W", "Qrev_W", "P_W", "OCV")}
    events = []

    def advance(I, step_no):
        nonlocal soc, v1, T, t
        R0 = float(ecm.r0_at(T, state.r_scale))
        R1 = float(ecm.r1_at(T, state.r_scale, I))
        a = np.exp(-dt / ecm.tau1)
        eta = ecm.eta if I < 0 else 1.0
        soc_n = soc - eta * I * dt / (3600.0 * Q)
        v1_n = a * v1 + R1 * (1 - a) * I
        qrev = -I * (T + 273.15) * th.dUdT          # reversible (entropic) heat, I > 0 = discharge
        qgen = I * I * R0 + v1_n * v1_n / R1 + qrev
        T_n = T + dt * (qgen - th.hA * (T - state.T_amb_C)) / th.C_th
        ocv = float(ecm.ocv(soc_n))
        V = ocv - I * R0 - v1_n
        soc, v1, T, t = soc_n, v1_n, T_n, t + dt
        for k, val in zip(rec, (t, step_no, I, V, soc, v1, T, R0, qgen, qrev, V * I, ocv)):
            rec[k].append(val)
        return V

    # initial point
    ocv0 = float(ecm.ocv(soc))
    for k, val in zip(rec, (0.0, 0, 0.0, ocv0, soc, 0.0, T, float(ecm.r0_at(T, state.r_scale)), 0.0, 0.0, 0.0, ocv0)):
        rec[k].append(val)

    for n, st in enumerate(steps, start=1):
        typ = st["type"]
        t_start = t
        reason = "max time"
        if typ == "cc":
            I = float(st["current_A"])
            lim = st.get("until_V", state.v_min if I > 0 else state.v_max)
            while t - t_start < st.get("max_s", 4 * 3600) and t < max_total_s:
                V = advance(I, n)
                if (I > 0 and V <= lim) or (I < 0 and V >= lim):
                    reason = f"voltage limit {lim:.2f} V"
                    break
                if soc < -0.08 or soc > 1.08:
                    reason = "SOC bound"
                    break
        elif typ == "cv":
            Vset = float(st.get("voltage_V", state.v_max))
            i_cut = float(st.get("until_A", 0.02))
            i_lim = -abs(float(st.get("max_current_A", 10.0)))
            lag = max(1, int(round(600.0 / dt)))      # settle check window: 10 min
            hist = []
            while t - t_start < st.get("max_s", 4 * 3600) and t < max_total_s:
                R0 = float(ecm.r0_at(T, state.r_scale))
                ocv_n = float(ecm.ocv(soc))
                R1 = float(ecm.r1_at(T, state.r_scale, -1.0 if Vset >= ocv_n else 1.0))
                a = np.exp(-dt / ecm.tau1)
                # solve V_{k+1} = Vset for the current (I<0 when charging)
                I = (ocv_n - a * v1 - Vset) / (R0 + R1 * (1 - a))
                I = float(np.clip(I, i_lim, 0.0))
                advance(I, n)
                if abs(I) <= i_cut:
                    reason = f"taper current {i_cut:.3f} A"
                    break
                # the 1-RC twin's OCV saturates a few mV below the CV set-point near full charge, so its
                # current can settle above the cut-off; stop once it has stopped decaying (< 0.5 % in 10 min)
                hist.append(abs(I))
                if len(hist) > lag and hist[-1 - lag] - hist[-1] < 0.005 * hist[-1] and hist[-1] < 3 * i_cut:
                    reason = f"current settled at {abs(I):.3f} A"
                    break
        elif typ == "rest":
            dur = float(st.get("duration_s", 600))
            while t - t_start < dur and t < max_total_s:
                advance(0.0, n)
            reason = "duration"
        elif typ == "pulse":
            hi, lo = float(st["high_A"]), float(st.get("low_A", 0.0))
            per, duty = float(st.get("period_s", 20.0)), float(st.get("duty", 0.5))
            lim = st.get("until_V", state.v_min)
            while t - t_start < st.get("max_s", 4 * 3600) and t < max_total_s:
                phase = ((t - t_start) % per) / per
                I = hi if phase < duty else lo
                V = advance(I, n)
                if V <= lim:
                    reason = f"voltage limit {lim:.2f} V"
                    break
        elif typ == "profile":
            tp = np.asarray(st["t_s"], float)
            ip = np.asarray(st["current_A"], float)
            lim_lo = st.get("until_V", None)
            while t - t_start < tp[-1] and t < max_total_s:
                I = float(np.interp(t - t_start, tp, ip))
                V = advance(I, n)
                if lim_lo is not None and V <= lim_lo:
                    reason = f"voltage limit {lim_lo:.2f} V"
                    break
            else:
                reason = "profile end"
        events.append(dict(step=n, type=typ, t_start_s=t_start, t_end_s=t, end_reason=reason))
        if t >= max_total_s:
            break

    df = pd.DataFrame(rec)
    df["Ah_out"] = np.concatenate([[0.0], np.cumsum(0.5 * (df.I_A.values[1:] + df.I_A.values[:-1]) * np.diff(df.t_s.values))]) / 3600.0
    df["Wh_out"] = np.concatenate([[0.0], np.cumsum(0.5 * (df.P_W.values[1:] + df.P_W.values[:-1]) * np.diff(df.t_s.values))]) / 3600.0
    df["C_rate"] = df.I_A / Q
    summary = dict(
        duration_min=float(df.t_s.iloc[-1] / 60.0), Q_twin_Ah=Q,
        Ah_discharged=float(df.Ah_out.clip(lower=None).iloc[-1]),
        Wh_net=float(df.Wh_out.iloc[-1]), soc_final_pct=float(100 * df.SOC.iloc[-1]),
        V_min=float(df.V.min()), V_max=float(df.V.max()),
        T_max_C=float(df.T_C.max()), dT_max_C=float(df.T_C.max() - state.T0_C),
        heat_J=float(np.trapezoid(df.Qgen_W, df.t_s)), mean_heat_W=float(df.Qgen_W.mean()),
        end_reason="; ".join(f"step {e['step']} ({e['type']}): {e['end_reason']}" for e in events),
    )
    return df, summary


def thermal_stress_index(T_max_C: float, dT_C: float) -> tuple[str, str]:
    """Relative, ILLUSTRATIVE stress band (not a safety prediction)."""
    if T_max_C >= 60:
        return "High", "exceeds the typical 60 °C upper operating limit of 18650 datasheets"
    if T_max_C >= 45 or dT_C >= 20:
        return "Elevated", "above ~45 °C or >20 K rise: accelerated ageing region"
    return "Normal", "within typical operating window"
