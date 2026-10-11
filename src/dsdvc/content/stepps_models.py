"""STEPPS functional-form comparison (spec §5).

Tests, rather than assumes, how STEPPS relates to sharing:

* additive:        V = b0 + sum_k b_k X_k
* multiplicative:  V = exp(b0) prod_k X_k^{b_k}     (log-log regression)
* interaction:     V = b0 + b X + sum_{j<k} g_jk X_j X_k  (ridge-regularised)

All three predict the same target (share rate) and are compared on K-fold
out-of-sample error on the *original* scale, plus AIC on the fitting scale.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from scipy import stats

from dsdvc.content.stepps import DIMENSIONS
from dsdvc.core.literature import cites

EPS = 1e-3


def _design(X: np.ndarray, form: str) -> np.ndarray:
    if form == "additive":
        return X
    if form == "multiplicative":
        return np.log(np.clip(X, EPS, None))
    if form == "interaction":
        pairs = [X[:, j] * X[:, k] for j, k in itertools.combinations(range(X.shape[1]), 2)]
        return np.column_stack([X] + pairs)
    raise ValueError(form)


def _target(y: np.ndarray, form: str) -> np.ndarray:
    return np.log(np.clip(y, EPS * 1e-2, None)) if form == "multiplicative" else y


def _inverse(yhat: np.ndarray, form: str, resid_var: float = 0.0) -> np.ndarray:
    return np.exp(yhat + resid_var / 2) if form == "multiplicative" else yhat


@cites("berger2013", "berger_milkman2012")
def compare_stepps_forms(X: np.ndarray | pd.DataFrame, y: np.ndarray, k_folds: int = 5, seed: int = 0) -> dict:
    """Return a comparison table and the best form by out-of-sample RMSE."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rows = []
    coefs = {}
    for form in ("additive", "multiplicative", "interaction"):
        D, T = _design(X, form), _target(y, form)
        model_cls = (lambda: RidgeCV(alphas=np.logspace(-4, 2, 20))) if form == "interaction" else LinearRegression
        errs, spear = [], []
        for tr, te in KFold(k_folds, shuffle=True, random_state=seed).split(D):
            m = model_cls().fit(D[tr], T[tr])
            rv = float(np.var(T[tr] - m.predict(D[tr])))
            pred = _inverse(m.predict(D[te]), form, rv)
            errs.append(np.sqrt(np.mean((pred - y[te]) ** 2)))
            spear.append(stats.spearmanr(pred, y[te]).statistic)
        m = model_cls().fit(D, T)
        resid = T - m.predict(D)
        n, k = D.shape[0], D.shape[1] + 1
        sigma2 = max(float(np.mean(resid ** 2)), 1e-12)
        ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
        # AIC on different target scales is not comparable; we report the Jacobian-adjusted AIC
        if form == "multiplicative":
            ll -= float(np.sum(np.log(np.clip(y, EPS * 1e-2, None))))
        rows.append({"form": form, "cv_rmse": float(np.mean(errs)), "cv_rmse_sd": float(np.std(errs)),
                     "cv_spearman": float(np.nanmean(spear)), "aic": float(2 * k - 2 * ll), "n_params": k})
        names = list(DIMENSIONS) if X.shape[1] == 6 else [f"x{i}" for i in range(X.shape[1])]
        if form == "interaction":
            names = names + [f"{names[j]}*{names[k]}" for j, k in itertools.combinations(range(X.shape[1]), 2)]
        coefs[form] = dict(zip(names, np.round(m.coef_, 4).tolist()))
    table = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    best = table.iloc[0]["form"]
    return {"table": table, "best_form": best, "coefficients": coefs,
            "note": "Model chosen by out-of-sample RMSE on share rate. STEPPS is a hypothesis; a weak or "
                    "negative coefficient is a legitimate finding."}
