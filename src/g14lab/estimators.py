"""SOC estimators: Coulomb counting, Extended Kalman Filter, Unscented Kalman Filter.

State x = [SOC, V1]  (1-RC ECM), input u = I (>0 discharge), measurement y = V_t.
    x_{k+1} = [SOC_k - I_k dt/(3600 Q);  a V1_k + R1(sign I_k) (1-a) I_k]
    y_k     = OCV(SOC_k) - I_k R0 - V1_k
EKF Jacobians: F = diag(1, a);  H = [dOCV/dSOC, -1].
The estimators only see the (optionally corrupted) measured current and voltage; the
reference SOC is never fed back into them.
"""
from __future__ import annotations

import numpy as np

from .ecm import ECMParams


def coulomb_counting(t, I, Q_Ah, soc0, eta=1.0):
    t = np.asarray(t, float)
    I = np.asarray(I, float)
    soc = np.empty_like(t)
    soc[0] = soc0
    for k in range(1, len(t)):
        e = eta if I[k - 1] < 0 else 1.0
        soc[k] = soc[k - 1] - e * I[k - 1] * (t[k] - t[k - 1]) / (3600.0 * Q_Ah)
    return soc


def ekf(params: ECMParams, t, I, V, soc0, Q_Ah=None, T_C=None, p0=(0.1, 0.01), q=(1e-7, 1e-6), r=1e-3,
        adaptive: bool = False, window: int = 25, r_bounds=(1e-5, 5e-2)):
    """Extended Kalman filter. With ``adaptive=True`` the measurement-noise variance R is
    re-estimated from the innovation sequence (Sage-Husa / innovation-based adaptive EKF):
        R_k = mean(e^2 over the last N steps) - H P^- H^T, clipped to r_bounds.
    Returns dict(soc, v1, sigma_soc, innovation, v_pred, r_trace)."""
    t = np.asarray(t, float)
    I = np.asarray(I, float)
    V = np.asarray(V, float)
    n = len(t)
    Q_Ah = Q_Ah or params.Q_Ah
    T_C = np.full(n, params.T_ref_C) if T_C is None else np.asarray(T_C, float)
    x = np.array([soc0, 0.0])
    P = np.diag(p0).astype(float)
    Qn = np.diag(q)
    out_soc, out_v1, out_sig, out_inn, out_vp, out_r = (np.empty(n) for _ in range(6))
    hist = []
    r_k = r
    for k in range(n):
        if k > 0:
            dt = t[k] - t[k - 1]
            a = np.exp(-dt / params.tau1)
            R1 = float(params.r1_at(T_C[k - 1], 1.0, I[k - 1]))
            x = np.array([x[0] - I[k - 1] * dt / (3600.0 * Q_Ah), a * x[1] + R1 * (1 - a) * I[k - 1]])
            F = np.diag([1.0, a])
            P = F @ P @ F.T + Qn * max(dt, 1.0)
        R0 = float(params.r0_at(T_C[k]))
        v_pred = float(params.ocv(x[0])) - I[k] * R0 - x[1]
        H = np.array([float(params.docv(x[0])), -1.0])
        inn = V[k] - v_pred
        if adaptive:
            hist.append(inn * inn)
            if len(hist) > window:
                hist.pop(0)
            if len(hist) >= 5:
                r_k = float(np.clip(np.mean(hist) - H @ P @ H, *r_bounds))
        S = H @ P @ H + r_k
        K = P @ H / S
        x = x + K * inn
        x[0] = np.clip(x[0], -0.1, 1.1)
        P = (np.eye(2) - np.outer(K, H)) @ P
        out_soc[k], out_v1[k], out_sig[k], out_inn[k], out_vp[k], out_r[k] = (
            x[0], x[1], np.sqrt(max(P[0, 0], 0)), inn, v_pred, r_k)
    return dict(soc=out_soc, v1=out_v1, sigma_soc=out_sig, innovation=out_inn, v_pred=out_vp, r_trace=out_r)


def ukf(params: ECMParams, t, I, V, soc0, Q_Ah=None, T_C=None, p0=(0.1, 0.01), q=(1e-7, 1e-6), r=1e-3,
        alpha=0.3, beta=2.0, kappa=0.0):
    """Unscented KF (scaled sigma points) on the same model; handles OCV curvature without Jacobians."""
    t = np.asarray(t, float)
    I = np.asarray(I, float)
    V = np.asarray(V, float)
    n = len(t)
    L = 2
    Q_Ah = Q_Ah or params.Q_Ah
    T_C = np.full(n, params.T_ref_C) if T_C is None else np.asarray(T_C, float)
    lam = alpha ** 2 * (L + kappa) - L
    Wm = np.full(2 * L + 1, 1.0 / (2 * (L + lam)))
    Wc = Wm.copy()
    Wm[0] = lam / (L + lam)
    Wc[0] = lam / (L + lam) + (1 - alpha ** 2 + beta)
    x = np.array([soc0, 0.0])
    P = np.diag(p0).astype(float)
    Qn = np.diag(q)
    out_soc, out_sig, out_vp = np.empty(n), np.empty(n), np.empty(n)

    def sigma(x, P):
        S = np.linalg.cholesky((L + lam) * (P + 1e-12 * np.eye(L)))
        return np.vstack([x, x + S.T, x - S.T])

    for k in range(n):
        if k > 0:
            dt = t[k] - t[k - 1]
            a = np.exp(-dt / params.tau1)
            R1 = float(params.r1_at(T_C[k - 1], 1.0, I[k - 1]))
            X = sigma(x, P)
            X = np.column_stack([X[:, 0] - I[k - 1] * dt / (3600.0 * Q_Ah), a * X[:, 1] + R1 * (1 - a) * I[k - 1]])
            x = Wm @ X
            d = X - x
            P = (Wc[:, None] * d).T @ d + Qn * max(dt, 1.0)
        X = sigma(x, P)
        R0 = float(params.r0_at(T_C[k]))
        Y = params.ocv(X[:, 0]) - I[k] * R0 - X[:, 1]
        y = Wm @ Y
        dy = Y - y
        dx = X - x
        Pyy = Wc @ (dy * dy) + r
        Pxy = (Wc[:, None] * dx).T @ dy
        K = Pxy / Pyy
        x = x + K * (V[k] - y)
        x[0] = np.clip(x[0], -0.1, 1.1)
        P = P - np.outer(K, K) * Pyy
        P = 0.5 * (P + P.T)
        out_soc[k], out_sig[k], out_vp[k] = x[0], np.sqrt(max(P[0, 0], 0)), y
    return dict(soc=out_soc, sigma_soc=out_sig, v_pred=out_vp)


def convergence_time(t, err, band=0.02):
    """First time after which |error| stays inside the band (s); NaN if never."""
    err = np.abs(np.asarray(err))
    outside = np.where(err > band)[0]
    if outside.size == 0:
        return 0.0
    k = outside[-1] + 1
    return float(t[k] - t[0]) if k < len(t) else np.nan
