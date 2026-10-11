"""Small, dependency-free statistical estimators with standard errors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class LogitResult:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    loglik: float
    n: int
    l2: float

    @property
    def aic(self) -> float:
        return float(2 * len(self.coef) - 2 * self.loglik)

    def table(self) -> pd.DataFrame:
        z = self.coef / np.where(self.se > 0, self.se, np.nan)
        return pd.DataFrame({"coef": self.coef, "se": self.se, "z": z,
                             "p_value": 2 * (1 - stats.norm.cdf(np.abs(z))),
                             "ci_low": self.coef - 1.645 * self.se, "ci_high": self.coef + 1.645 * self.se},
                            index=self.names)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(np.column_stack([np.ones(len(X)), X]) @ self.coef)


def fit_logit(X: np.ndarray | pd.DataFrame, y: np.ndarray, l2: float = 1e-3, names: list[str] | None = None,
              weights: np.ndarray | None = None) -> LogitResult:
    """Ridge-stabilised logistic regression (intercept unpenalised) with Hessian-based SEs."""
    if isinstance(X, pd.DataFrame):
        names = names or list(X.columns)
        X = X.to_numpy(dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    Z = np.column_stack([np.ones(len(X)), X])
    pen = np.full(Z.shape[1], l2)
    pen[0] = 0.0

    def nll(b):
        eta = Z @ b
        return -np.sum(w * (y * eta - np.logaddexp(0, eta))) + 0.5 * np.sum(pen * b * b)

    def grad(b):
        p = sigmoid(Z @ b)
        return -Z.T @ (w * (y - p)) + pen * b

    r = optimize.minimize(nll, np.zeros(Z.shape[1]), jac=grad, method="L-BFGS-B")
    b = r.x
    p = sigmoid(Z @ b)
    H = (Z * (w * p * (1 - p))[:, None]).T @ Z + np.diag(pen)
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv(H)), 0, None))
    ll = float(np.sum(w * (y * (Z @ b) - np.logaddexp(0, Z @ b))))
    return LogitResult(["intercept"] + list(names or [f"x{i}" for i in range(X.shape[1])]), b, se, ll, len(y), l2)
