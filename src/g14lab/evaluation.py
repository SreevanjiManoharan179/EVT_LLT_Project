"""Error metrics shared by all subsystems (all computed from actual model outputs)."""
from __future__ import annotations

import numpy as np


def _clean(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    return y[m], yhat[m]


def mae(y, yhat):
    y, yhat = _clean(y, yhat)
    return float(np.mean(np.abs(yhat - y))) if y.size else np.nan


def rmse(y, yhat):
    y, yhat = _clean(y, yhat)
    return float(np.sqrt(np.mean((yhat - y) ** 2))) if y.size else np.nan


def max_abs(y, yhat):
    y, yhat = _clean(y, yhat)
    return float(np.max(np.abs(yhat - y))) if y.size else np.nan


def r2(y, yhat):
    y, yhat = _clean(y, yhat)
    if y.size < 2:
        return np.nan
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def summary(y, yhat, scale: float = 1.0) -> dict:
    """MAE/RMSE/max error (multiplied by ``scale``, e.g. 100 for SOC in % points) and R^2."""
    return dict(MAE=mae(y, yhat) * scale, RMSE=rmse(y, yhat) * scale,
                MaxAbs=max_abs(y, yhat) * scale, R2=r2(y, yhat), n=int(_clean(y, yhat)[0].size))
