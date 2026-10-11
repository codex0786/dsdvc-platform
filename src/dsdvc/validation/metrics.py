"""Evaluation metrics (spec §34): MAE, RMSE, MAPE, R^2, log loss, Brier, AUC, calibration
curve, precision/recall, lift and cumulative gain."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def regression_report(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    nz = np.abs(y) > 1e-9
    return {
        "mae": float(np.mean(np.abs(y - yhat))),
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        "mape": float(np.mean(np.abs((y[nz] - yhat[nz]) / y[nz]))) if nz.any() else float("nan"),
        "r2": float(skm.r2_score(y, yhat)) if len(y) > 1 else float("nan"),
    }


def lift_at(y, p, frac: float = 0.1) -> float:
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-p)[:k]
    base = y.mean()
    return float(y[top].mean() / base) if base > 0 else float("nan")


def cumulative_gain(y, p, points: int = 10) -> list[tuple[float, float]]:
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    cum = np.cumsum(y[order]) / max(y.sum(), 1)
    out = []
    for q in np.linspace(0.1, 1.0, points):
        k = max(1, int(round(q * len(y))))
        out.append((float(q), float(cum[k - 1])))
    return out


def calibration_curve(y, p, bins: int = 10) -> list[dict]:
    y, p = np.asarray(y), np.asarray(p)
    qs = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    out = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (p >= lo) & (p <= hi)
        if m.any():
            out.append({"bin_low": float(lo), "bin_high": float(hi), "mean_pred": float(p[m].mean()),
                        "observed": float(y[m].mean()), "n": int(m.sum())})
    return out


def expected_calibration_error(y, p, bins: int = 10) -> float:
    cc = calibration_curve(y, p, bins)
    n = sum(c["n"] for c in cc)
    return float(sum(c["n"] * abs(c["mean_pred"] - c["observed"]) for c in cc) / max(n, 1))


def classification_report(y, p, top_frac: float = 0.05) -> dict:
    y, p = np.asarray(y).astype(int), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = {"log_loss": float(skm.log_loss(y, p, labels=[0, 1])), "brier": float(skm.brier_score_loss(y, p)),
           "ece": expected_calibration_error(y, p)}
    if 0 < y.sum() < len(y):
        out["auc"] = float(skm.roc_auc_score(y, p))
        out["pr_auc"] = float(skm.average_precision_score(y, p))
    else:
        out["auc"] = out["pr_auc"] = float("nan")
    k = max(1, int(round(top_frac * len(y))))
    pred = np.zeros_like(y)
    pred[np.argsort(-p)[:k]] = 1
    out["precision_at_top"] = float(skm.precision_score(y, pred, zero_division=0))
    out["recall_at_top"] = float(skm.recall_score(y, pred, zero_division=0))
    out["lift_top_decile"] = lift_at(y, p, 0.1)
    return out
