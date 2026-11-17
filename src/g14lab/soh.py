"""SOH / degradation analysis.

* Capacity-based SOH:  SOH_Q(%) = Q_measured / Q_reference * 100, Q_reference = 2.0 Ah rated
  (NASA EOL definition: "30% fade in rated capacity, 2 Ah -> 1.4 Ah").
* ML estimation of the CURRENT capacity from health indicators (HIs) of the same cycle.
  Models: Ridge (linear baseline), Random Forest (Group 14 baseline), gradient boosting.
  Validation: leave-one-cell-out (LOCO) - every prediction is for a cell the model never saw.
  A random K-fold score is also computed ONLY to show how row-level leakage inflates accuracy.
* Degradation-trend forecasting (remaining capacity / cycles to EOL) from the early part of a
  cell's own fade curve: double-exponential empirical model with bootstrap bands.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import evaluation as ev

SEED = 14
RATED_AH = 2.0
REFERENCE_CELLS = ["B0005", "B0006", "B0007", "B0018"]

FEATURE_SETS = {
    "Charging-phase indicators (on-board measurable)": ["t_cc_s", "q_cc_Ah", "t_cv_s", "T_rise_charge_C"],
    "Incremental-capacity (ICA) indicators": ["ic_peak_AhV", "ic_peak_V", "ic_area_Ah"],
    "Charging + ICA + EIS impedance (Re, Rct)": ["t_cc_s", "q_cc_Ah", "t_cv_s", "T_rise_charge_C",
                                                "ic_peak_AhV", "ic_peak_V", "ic_area_Ah", "Re_ohm", "Rct_ohm"],
    "Discharge-curve indicators": ["t_to_3p8V_s", "t_to_3p6V_s", "T_rise_C", "mean_voltage_loaded_V",
                                   "r_dc_first_sample_ohm"],
}
DEFAULT_FEATURE_SET = "Charging + ICA + EIS impedance (Re, Rct)"
CONDITION_FEATURES = ["ambient_C", "mean_current_A"]
FEATURE_LABELS = {
    "t_cc_s": "CC-charge duration (s)", "q_cc_Ah": "Charge in CC phase (Ah)", "t_cv_s": "CV-charge duration (s)",
    "T_rise_charge_C": "Temperature rise during charge (K)", "Re_ohm": "EIS electrolyte resistance Re (Ω)",
    "Rct_ohm": "EIS charge-transfer resistance Rct (Ω)", "t_to_3p8V_s": "Time to 3.8 V on discharge (s)",
    "t_to_3p6V_s": "Time to 3.6 V on discharge (s)", "T_rise_C": "Temperature rise on discharge (K)",
    "mean_voltage_loaded_V": "Mean loaded voltage (V)", "r_dc_first_sample_ohm": "DC resistance proxy (Ω)",
    "ambient_C": "Ambient temperature (°C)", "mean_current_A": "Mean discharge current (A)",
    "ic_peak_AhV": "ICA peak height (Ah/V)", "ic_peak_V": "ICA peak voltage (V)",
    "ic_area_Ah": "ICA charge in 3.80–4.10 V (Ah)",
}
MODELS = ["Ridge regression", "Random Forest", "Gradient boosting"]


def make_model(name: str, n_estimators: int = 300):
    if name == "Ridge regression":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=2, max_features=0.8,
                                     random_state=SEED, n_jobs=-1)
    if name == "Gradient boosting":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=10, random_state=SEED)
    raise ValueError(name)


def dataset(dis: pd.DataFrame, features: list[str], cells: list[str] | None, with_conditions: bool) -> pd.DataFrame:
    df = dis[dis.usable_for_soh & ~dis.flag_capacity_mismatch].copy()
    if cells:
        df = df[df.cell_id.isin(cells)]
    cols = list(features) + (CONDITION_FEATURES if with_conditions else [])
    df = df.dropna(subset=cols + ["capacity_Ah"])
    # discard physically implausible feature rows (logging glitches, partial charges)
    if "t_cc_s" in df.columns:
        df = df[(df.t_cc_s > 60) | df.t_cc_s.isna()]
    if "v_start_charge_V" in df.columns:          # drop partial charges (CC phase starting above 3.9 V)
        df = df[(df.v_start_charge_V < 3.9) | df.v_start_charge_V.isna()]
    if "ic_peak_AhV" in cols:
        df = df[df.ic_peak_AhV.between(0.5, 15)]
    return df


def loco(df: pd.DataFrame, cols: list[str], model_name: str, n_estimators: int = 300):
    """Leave-one-cell-out predictions with (for RF) per-tree 5-95 % intervals."""
    X, y, g = df[cols].to_numpy(float), df["capacity_Ah"].to_numpy(float), df["cell_id"].to_numpy()
    pred = np.full(len(df), np.nan)
    lo = np.full(len(df), np.nan)
    hi = np.full(len(df), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = make_model(model_name, n_estimators)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        if model_name == "Random Forest":
            trees = np.stack([t.predict(X[te]) for t in m.estimators_])
            lo[te], hi[te] = np.percentile(trees, 5, axis=0), np.percentile(trees, 95, axis=0)
    out = df[["cell_id", "discharge_index", "capacity_Ah", "soh_pct"]].copy()
    out["pred_Ah"], out["lo_Ah"], out["hi_Ah"] = pred, lo, hi
    out["pred_soh_pct"] = 100 * pred / RATED_AH
    # Cross-conformal (leave-one-cell-out) 90 % intervals: the half-width for a test cell is the
    # finite-sample 90 % quantile of |LOCO residuals| of the OTHER cells -> no information leak.
    res = np.abs(out.capacity_Ah - out.pred_Ah).to_numpy()
    half = np.full(len(out), np.nan)
    for c in np.unique(g):
        other = res[g != c]
        n = other.size
        lvl = min(1.0, np.ceil((n + 1) * 0.9) / n)
        half[g == c] = np.quantile(other, lvl)
    out["conf_lo_Ah"], out["conf_hi_Ah"] = out.pred_Ah - half, out.pred_Ah + half
    per_cell = (out.groupby("cell_id")
                .apply(lambda d: pd.Series(ev.summary(d.capacity_Ah, d.pred_Ah)), include_groups=False)
                .reset_index())
    pooled = ev.summary(out.capacity_Ah, out.pred_Ah)
    if model_name == "Random Forest":
        inside = (out.capacity_Ah >= out.lo_Ah) & (out.capacity_Ah <= out.hi_Ah)
        pooled["interval_coverage_90"] = float(inside.mean())
    cin = (out.capacity_Ah >= out.conf_lo_Ah) & (out.capacity_Ah <= out.conf_hi_Ah)
    pooled["conformal_coverage_90"] = float(cin.mean())
    pooled["conformal_halfwidth_mAh"] = float(1000 * np.nanmedian(out.conf_hi_Ah - out.pred_Ah))
    return out, per_cell, pooled


def random_kfold(df: pd.DataFrame, cols: list[str], model_name: str, k: int = 5, n_estimators: int = 300) -> dict:
    """Row-level random K-fold (LEAKY: neighbouring cycles of the same cell end up in train and
    test). Reported only to quantify the optimism it causes."""
    X, y = df[cols].to_numpy(float), df["capacity_Ah"].to_numpy(float)
    pred = np.full(len(df), np.nan)
    for tr, te in KFold(k, shuffle=True, random_state=SEED).split(X):
        m = make_model(model_name, n_estimators)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return ev.summary(y, pred)


def fit_full(df: pd.DataFrame, cols: list[str], model_name: str, n_estimators: int = 300):
    m = make_model(model_name, n_estimators)
    m.fit(df[cols].to_numpy(float), df["capacity_Ah"].to_numpy(float))
    return m


def permutation_importance_loco(df, cols, model_name, test_cell, n_repeats=5):
    from sklearn.inspection import permutation_importance
    tr, te = df[df.cell_id != test_cell], df[df.cell_id == test_cell]
    m = fit_full(tr, cols, model_name, n_estimators=200)
    r = permutation_importance(m, te[cols].to_numpy(float), te.capacity_Ah.to_numpy(float),
                               n_repeats=n_repeats, random_state=SEED, scoring="neg_root_mean_squared_error")
    return pd.DataFrame(dict(feature=[FEATURE_LABELS.get(c, c) for c in cols],
                             importance_mAh=1000 * r.importances_mean, std_mAh=1000 * r.importances_std)
                        ).sort_values("importance_mAh", ascending=False)


# ----------------------------------------------------------------------------- forecasting
def _init_lin(k, y):
    return [y[0], -1e-3], [-np.inf, -0.05], [np.inf, 0.0]


def _init_pow(k, y):
    return [y[0], 1e-3, 1.0], [0, 0, 0.3], [3, 1, 3.0]


def _init_exp(k, y):
    return [y[0] + 0.01, 0.01, 0.01], [0, 0, 1e-4], [3, 2, 0.08]


def _init_dexp(k, y):
    return [y[0], -1e-3, -1e-3, 0.02], [0, -0.05, -1, 0], [3, 0, 0, 0.05]


FADE_MODELS = {
    "linear": (lambda k, p: p[0] + p[1] * k, _init_lin, r"Q(k) = a + b\,k"),
    "power law": (lambda k, p: p[0] - p[1] * np.power(k, p[2]), _init_pow, r"Q(k) = a - b\,k^{c}"),
    "exponential": (lambda k, p: p[0] - p[1] * np.exp(p[2] * k), _init_exp, r"Q(k) = a - b\,e^{c k}"),
    "double exponential": (lambda k, p: p[0] * np.exp(p[1] * k) + p[2] * np.exp(p[3] * k), _init_dexp,
                           r"Q(k) = a\,e^{b k} + c\,e^{d k}"),
}


def fit_fade(name: str, k, y):
    """Robust (soft-L1) bounded least-squares fit; returns (callable, params, AICc)."""
    from scipy.optimize import least_squares
    f, init, _ = FADE_MODELS[name]
    x0, lb, ub = init(k, y)
    x0 = np.clip(x0, np.array(lb, float) + 1e-9, np.array(ub, float) - 1e-9)
    r = least_squares(lambda p: f(k, p) - y, x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.02, max_nfev=4000)
    n, m = len(y), len(x0)
    rss = float(np.sum((f(k, r.x) - y) ** 2))
    aicc = n * np.log(rss / n) + 2 * m + 2 * m * (m + 1) / max(n - m - 1, 1)
    return (lambda z, _p=r.x: f(z, _p)), r.x, aicc


def fade_forecast(k, cap, n_train: int, eol_Ah: float = 1.4, model: str = "linear", horizon: int | None = None,
                  n_boot: int = 60):
    """Fit a capacity-fade trend on the first n_train cycles and extrapolate, with a residual-bootstrap band.
    Returns forecast curve, band, predicted / actual EOL cycle, hidden-cycle error and the AICc of every model."""
    k = np.asarray(k, float)
    cap = np.asarray(cap, float)
    kt, ct = k[:n_train], cap[:n_train]
    horizon = horizon or int(max(k.max() * 1.6, n_train + 50))
    kk = np.arange(1, horizon + 1, dtype=float)
    f, params, _ = fit_fade(model, kt, ct)
    aicc = {}
    for m in FADE_MODELS:
        try:
            aicc[m] = fit_fade(m, kt, ct)[2]
        except Exception:
            aicc[m] = np.nan
    curve = f(kk)
    resid = ct - f(kt)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        cb = f(kt) + rng.choice(resid, size=len(resid), replace=True)
        try:
            fb, _, _ = fit_fade(model, kt, cb)
            boots.append(fb(kk) + rng.choice(resid, size=len(kk)))
        except Exception:
            continue
    boots = np.array(boots) if boots else np.array([curve])
    lo, hi = np.percentile(boots, 5, axis=0), np.percentile(boots, 95, axis=0)

    def first_below(arr):
        idx = np.where(arr <= eol_Ah)[0]
        return float(kk[idx[0]]) if idx.size else np.nan

    act = np.where(cap <= eol_Ah)[0]
    eol_actual = float(k[act[0]]) if act.size else np.nan
    eol_pred = first_below(curve)
    test = k > kt.max()
    return dict(model=model, k=kk, curve=curve, lo=lo, hi=hi, eol_pred=eol_pred, eol_band=(first_below(hi), first_below(lo)),
                eol_actual=eol_actual, rul_pred=eol_pred - kt.max() if np.isfinite(eol_pred) else np.nan,
                rul_actual=eol_actual - kt.max() if np.isfinite(eol_actual) else np.nan,
                test_metrics=ev.summary(cap[test], f(k[test])) if test.any() else {}, n_train=n_train, aicc=aicc, params=params)


def forecast_backtest(dis: pd.DataFrame, cells=REFERENCE_CELLS, origins=(40, 60, 80, 100), eol_Ah: float = 1.4):
    """Rolling-origin backtest of every fade model on the reference cells (hidden-cycle RMSE, EOL error)."""
    rows = []
    for cell in cells:
        g = dis[(dis.cell_id == cell) & dis.usable_for_soh].sort_values("discharge_index")
        k, y = g.discharge_index.to_numpy(float), g.capacity_Ah.to_numpy(float)
        act = np.where(y <= eol_Ah)[0]
        eol_act = float(k[act[0]]) if act.size else np.nan
        kk = np.arange(1, 400, dtype=float)
        for n0 in origins:
            if n0 >= len(k) - 5:
                continue
            fits = {m: fit_fade(m, k[:n0], y[:n0]) for m in FADE_MODELS}
            best = min(fits, key=lambda m: fits[m][2])
            for m, (f, _, a) in fits.items():
                cur = f(kk)
                idx = np.where(cur <= eol_Ah)[0]
                eol = float(kk[idx[0]]) if idx.size else np.nan
                rows.append(dict(cell_id=cell, origin=n0, model=m, aicc=a, aicc_selected=(m == best), eol_pred=eol,
                                 eol_actual=eol_act, eol_error=eol - eol_act,
                                 rmse_hidden_mAh=1000 * float(np.sqrt(np.mean((f(k[n0:]) - y[n0:]) ** 2)))))
    return pd.DataFrame(rows)
