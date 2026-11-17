"""First-order Thevenin (1-RC) equivalent-circuit model and its identification from
measured NASA discharge data.

Model (I > 0 = discharge):
    V_t   = OCV(SOC) - I*R0 - V1
    V1[k+1] = a*V1[k] + R1(sign I)*(1-a)*I[k],      a = exp(-dt/tau1)
    SOC[k+1] = SOC[k] - eta*I[k]*dt/(3600*Q)
The ohmic resistance R0 is direction-independent; the polarisation resistance takes a separate
value R1_chg for charging current (as HPPC testing separates discharge and regen/charge pulse
resistance). Without R1_chg the model is symmetric.

Temperature dependence (identified across the 4 / 24 / 43 degC NASA cells):
    R(T) = R_ref * exp( Ea/Rg * (1/T - 1/T_ref) )            (Arrhenius)

Identification approach (docs/Technical_Report_Group14.md, section 4):
  * SOC reference = coulomb counting of the measured current, normalised by the measured
    capacity of the discharge: discharges start full (SOC 1), the CC-CV charge that follows a
    discharge starts at SOC 0, the cut-off point at the reference rate.
  * OCV(SOC) is a monotone PCHIP curve through 12 knots; knots, R0, R1 and tau1 are fitted
    jointly by non-linear least squares to the measured terminal voltage of discharges AND
    charges. The two current directions bracket the OCV, and the up-weighted rest samples
    separate it from the resistive drop.
  * Temperature-aware mode: with a known activation energy, R0 and R1 follow the Arrhenius law
    along the measured cell temperature during the fit and refer to T_ref (24 degC).
  * R0 is bounded below by the cell's EIS ohmic resistance Re: the instantaneous resistance
    seen at a 2.5-19 s sampling interval cannot be smaller than the high-frequency resistance.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

R_GAS = 8.314462618  # J/(mol K)
N_KNOTS = 12
RATED_AH = 2.0       # NASA 18650 rating; SOH (%) = 100 * Q / RATED_AH
I_DIR = 0.01         # A: a sample counts as charging when I < -I_DIR (dead band for current-sensor noise at rest)


@dataclass
class ECMParams:
    soc_knots: list
    ocv_knots: list
    R0: float            # ohm at T_ref
    R1: float            # ohm at T_ref
    tau1: float          # s
    Q_Ah: float          # capacity used to normalise SOC
    T_ref_C: float = 24.0
    Ea_R0: float = 0.0   # J/mol (0 -> temperature-independent)
    Ea_R1: float = 0.0
    eta: float = 1.0     # coulombic efficiency (charge)
    ocv_shift: float = 0.0   # ageing-aware OCV: OCV_aged(z) = OCV_ref(shift + scale*z)
    ocv_scale: float = 1.0   # (electrode-window drift; 0/1 = fresh reference curve)
    source: str = ""
    fit_rmse_mV: float = float("nan")
    extra: dict = field(default_factory=dict)
    R1_chg: float | None = None   # ohm at T_ref, polarisation resistance for charging current (None = symmetric)

    # ---- OCV ---------------------------------------------------------------
    def _interp(self):
        if not hasattr(self, "_pchip") or self._pchip is None:
            self._pchip = PchipInterpolator(np.asarray(self.soc_knots), np.asarray(self.ocv_knots), extrapolate=False)
            self._dpchip = self._pchip.derivative()
        return self._pchip, self._dpchip

    def ocv(self, soc):
        p, dp = self._interp()
        s = self.ocv_shift + self.ocv_scale * np.asarray(soc, float)
        lo, hi = self.soc_knots[0], self.soc_knots[-1]
        sc = np.clip(s, lo, hi)
        v = p(sc)
        # linear extrapolation outside the identified range using end slopes
        v = np.where(s < lo, p(lo) + dp(lo) * (s - lo), v)
        v = np.where(s > hi, p(hi) + dp(hi) * (s - hi), v)
        return v

    def docv(self, soc):
        p, dp = self._interp()
        lo, hi = self.soc_knots[0], self.soc_knots[-1]
        s = self.ocv_shift + self.ocv_scale * np.asarray(soc, float)
        return self.ocv_scale * dp(np.clip(s, lo, hi))

    # ---- temperature dependence -------------------------------------------
    def arrhenius(self, T_C, Ea):
        T = np.asarray(T_C, float) + 273.15
        Tr = self.T_ref_C + 273.15
        return np.exp(Ea / R_GAS * (1.0 / T - 1.0 / Tr))

    def r0_at(self, T_C, r_scale=1.0):
        return self.R0 * r_scale * self.arrhenius(T_C, self.Ea_R0)

    def r1_at(self, T_C, r_scale=1.0, I=None):
        """Polarisation resistance at temperature T_C; with the current I given, charging samples (I < -I_DIR)
        use R1_chg when the model is direction-dependent. Rest samples keep the discharge value, so the
        relaxation of a discharge polarisation never dissipates through the much smaller charge resistance."""
        base = self.R1 if (I is None or self.R1_chg is None) else np.where(np.asarray(I, float) < -I_DIR, self.R1_chg, self.R1)
        return base * r_scale * self.arrhenius(T_C, self.Ea_R1)

    @property
    def C1(self):
        return self.tau1 / self.R1

    # ---- ageing ------------------------------------------------------------
    @property
    def soh_pct(self):
        """SOH of the twin itself: its normalising capacity as % of the rating."""
        return 100.0 * self.Q_Ah / RATED_AH

    def resistance_factor(self, soh_pct: float) -> float:
        """Resistance multiplier at a given SOH (% of rating) relative to this twin's own SOH, from the growth
        law fitted on the digital-shadow tracking, R/R_early = 1 + k (SOH_early - SOH) (1.0 when none is stored)."""
        g = self.extra.get("resistance_growth")
        if not g:
            return 1.0
        if g.get("form") == "relative":
            return float(max(0.5, 1 + g["k_per_fade"] * (self.soh_pct - soh_pct) / 100.0))
        f = lambda s: g["intercept"] + g["k_per_fade"] * (1 - s / 100.0)  # noqa: E731  (legacy pooled form)
        return float(max(0.5, f(soh_pct) / f(self.soh_pct)))

    def to_dict(self):
        d = asdict(self)
        d.pop("_pchip", None)
        d.pop("_dpchip", None)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# --------------------------------------------------------------------------------------
def soc_reference(t, I, Q_Ah, soc0=1.0):
    """Coulomb-counted SOC from measured current (trapezoidal), I > 0 = discharge."""
    t = np.asarray(t, float)
    I = np.asarray(I, float)
    q = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t))]) / 3600.0
    return soc0 - q / Q_Ah


def simulate_voltage(params: ECMParams, t, I, soc, T_C=None, r_scale=1.0, v1_0=0.0):
    """Terminal voltage for a GIVEN current and SOC trajectory (used for identification and
    measured-vs-simulated validation). Zero-order hold on current between samples."""
    t = np.asarray(t, float)
    I = np.asarray(I, float)
    n = len(t)
    if T_C is None:
        T_C = np.full(n, params.T_ref_C)
    R0 = params.r0_at(T_C, r_scale)
    R1 = params.r1_at(T_C, r_scale, I) * np.ones(n)
    v1 = np.empty(n)
    v1[0] = v1_0
    dt = np.diff(t, prepend=t[0])
    for k in range(1, n):
        a = np.exp(-dt[k] / params.tau1)
        v1[k] = a * v1[k - 1] + R1[k - 1] * (1 - a) * I[k - 1]
    return params.ocv(soc) - I * R0 - v1, v1


def _knots_from_theta(theta):
    v0 = theta[0]
    inc = np.exp(theta[1:N_KNOTS])
    return np.concatenate([[v0], v0 + np.cumsum(inc)])


def identify_ecm(cycles: list[dict], source="", fix_ocv: ECMParams | None = None,
                 init=(0.08, 0.04, 40.0), rest_weight: float = 8.0, ocv_remap: bool = False,
                 Ea: float | None = None, T_ref_C: float = 24.0, r0_min: float = 0.005) -> ECMParams:
    """Identify OCV knots + R0, R1, tau1 from measured discharges and (optionally) CC-CV charges.

    Each cycle dict needs arrays t, V, I (I > 0 discharge, I < 0 charge), T, scalar Q (Ah) and,
    for charges, soc0 (start SOC). Fitting both current directions constrains the OCV between the
    charge and discharge curves, which separates it from the overpotential.
    If ``fix_ocv`` is given only R0, R1, tau1 are fitted (OCV shape and SOC grid reused).
    When the data contain charging current, R1_chg (polarisation resistance while charging) is fitted
    as well; R0 is shared by both directions.
    ``Ea`` (J/mol) makes the identification temperature-aware: R0 and R1 then follow the Arrhenius
    law along the MEASURED cell temperature and the fitted values refer to ``T_ref_C``. Without it
    the resistances are constant and refer to the mean cell temperature of the data.
    ``r0_min`` is a hard lower bound on R0 (e.g. the EIS ohmic resistance).
    """
    soc_grid = np.linspace(0.0, 1.0, N_KNOTS)
    prepared = []
    for c in cycles:
        # discharges start full (SOC 1); charges start at the previous cut-off (SOC 0, key soc0)
        soc = soc_reference(c["t"], c["I"], c["Q"], soc0=c.get("soc0", 1.0))
        prepared.append((np.asarray(c["t"], float), np.asarray(c["I"], float),
                         np.asarray(c["V"], float), soc, np.asarray(c["T"], float)))

    # charging data present -> the charge-direction polarisation resistance R1_chg is identified too
    has_chg = any((p_[1] < -0.2).any() for p_ in prepared)
    n_r = 4 if has_chg else 3

    def unpack(theta):
        rest = theta[N_KNOTS:] if fix_ocv is None else theta
        knots = _knots_from_theta(theta[:N_KNOTS]) if fix_ocv is None else np.asarray(fix_ocv.ocv_knots)
        r0, r1, tau = np.exp(rest[:3])
        r1c = float(np.exp(rest[3])) if has_chg else None
        a, b = (rest[n_r], np.exp(rest[n_r + 1])) if (fix_ocv is not None and ocv_remap) else (0.0, 1.0)
        return knots, r0, r1, tau, r1c, a, b

    # Rest samples (I ~ 0: before the load and the relaxation tail) carry the information that
    # separates the OCV level from the resistive drop, so they are up-weighted.
    weights = [np.where(np.abs(I) < 0.05, rest_weight, 1.0) for (t, I, V, soc, T) in prepared]

    if Ea is None and fix_ocv is not None and fix_ocv.Ea_R0 > 0:
        # re-identification against a temperature-aware reference stays temperature-aware, so the
        # fitted R0/R1 refer to the reference temperature and are comparable between cycles
        Ea, T_ref_C = fix_ocv.Ea_R0, fix_ocv.T_ref_C
    t_aware = Ea is not None and Ea > 0
    T_mean = float(np.mean(np.concatenate([p_[4] for p_ in prepared])))
    T_ref = T_ref_C if t_aware else T_mean

    def residuals(theta):
        knots, r0, r1, tau, r1c, a, b = unpack(theta)
        p = ECMParams(list(soc_grid), list(knots), r0, r1, tau, Q_Ah=1.0, ocv_shift=a, ocv_scale=b,
                      T_ref_C=T_ref, Ea_R0=Ea or 0.0, Ea_R1=Ea or 0.0, R1_chg=r1c)
        res = []
        for (t, I, V, soc, T), w in zip(prepared, weights):
            vm, _ = simulate_voltage(p, t, I, soc, T_C=T if t_aware else None)
            res.append((vm - V) * w)
        return np.concatenate(res)

    x_r = np.log(np.asarray(tuple(init) + ((init[1],) if has_chg else ()), float))
    r_lb = [r0_min, 0.001, 2.0] + ([0.001] if has_chg else [])
    if fix_ocv is None:
        # initial OCV guess: measured V + I*(R0+R1) mapped onto the SOC grid (first discharge)
        first = next((p_ for p_, c in zip(prepared, cycles) if c.get("soc0", 1.0) == 1.0), prepared[0])
        t, I, V, soc, T = first
        loaded = I > 0.2
        guess = np.interp(soc_grid, soc[loaded][::-1], (V + I * (init[0] + init[1]))[loaded][::-1])
        guess = np.maximum.accumulate(np.clip(guess, 2.5, 4.3))
        guess[-1] = max(guess[-1], 4.18)
        inc = np.maximum(np.diff(guess), 1e-3)
        x0 = np.concatenate([[guess[0]], np.log(inc), x_r])
        lb = np.concatenate([[2.0], np.full(N_KNOTS - 1, np.log(1e-5)), np.log(r_lb)])
        ub = np.concatenate([[3.9], np.full(N_KNOTS - 1, np.log(1.5)), np.log([0.6, 0.6, 2000.0] + [0.6] * (n_r - 3))])
    else:
        x0 = x_r
        lb = np.log(r_lb)
        ub = np.log([0.8, 0.8, 2000.0] + [0.8] * (n_r - 3))
        if ocv_remap:
            x0 = np.concatenate([x0, [0.0, 0.0]])
            lb = np.concatenate([lb, [-0.35, np.log(0.6)]])
            ub = np.concatenate([ub, [0.35, np.log(1.5)]])
    x0 = np.clip(x0, lb + 1e-9, ub - 1e-9)
    sol = least_squares(residuals, x0, bounds=(lb, ub), x_scale="jac", max_nfev=400)
    knots, r0, r1, tau, r1c, a, b = unpack(sol.x)
    if r1c is None and fix_ocv is not None and fix_ocv.R1_chg is not None:
        r1c = fix_ocv.R1_chg * r1 / fix_ocv.R1   # discharge-only refit: charge polarisation scales with R1
    w_all = np.concatenate(weights)
    fit_rmse = float(np.sqrt(np.mean((sol.fun / w_all) ** 2)) * 1000)
    Q_dis = [c["Q"] for c in cycles if c.get("soc0", 1.0) == 1.0] or [c["Q"] for c in cycles]
    n_chg = sum(1 for c in cycles if c.get("soc0", 1.0) != 1.0)
    return ECMParams(
        soc_knots=list(map(float, soc_grid)), ocv_knots=list(map(float, knots)), R0=float(r0), R1=float(r1),
        tau1=float(tau), Q_Ah=float(np.mean(Q_dis)), T_ref_C=float(T_ref), Ea_R0=float(Ea) if t_aware else 0.0,
        Ea_R1=float(Ea) if t_aware else 0.0, ocv_shift=float(a), ocv_scale=float(b),
        R1_chg=None if r1c is None else float(r1c), source=source, fit_rmse_mV=fit_rmse,
        extra=dict(n_cycles=len(cycles), n_discharges=len(cycles) - n_chg, n_charges=n_chg, T_mean_C=T_mean,
                   temperature_aware=bool(t_aware), r0_lower_bound_ohm=float(r0_min)),
    )


def fit_arrhenius(T_C, R, T_ref_C=24.0):
    """Least-squares fit of ln R = ln R_ref + Ea/Rg (1/T - 1/T_ref). Returns (R_ref, Ea)."""
    T = np.asarray(T_C, float) + 273.15
    x = 1.0 / T - 1.0 / (T_ref_C + 273.15)
    y = np.log(np.asarray(R, float))
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.exp(coef[0])), float(coef[1] * R_GAS)
