"""Conversion model with platform-specific coefficients (spec §25, §60).

    P(Conversion) = sigma(b0 + b1 Awareness + b2 SocialProof + b3 Trust + b4 PriceFit
                          + b5 STEPPS + b6 CommunityFit + b7 ProductFit)

Platform heterogeneity: b_k^{(p)} = b_k + d_k^{(p)} (deviation coded, ridge-shrunk toward the
pooled coefficient), so b_IG, b_FB, b_WA are estimated - not assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from dsdvc.core.prediction import Explanation, Prediction, confidence_label
from dsdvc.core.stats import fit_logit, sigmoid
from dsdvc.validation import metrics as VM

FEATURES = ["awareness", "social_proof", "trust", "price_fit", "stepps", "community_fit", "product_fit"]
PLATFORMS = ["instagram", "facebook", "whatsapp"]


def _design(df: pd.DataFrame, features: list[str], platform_specific: bool) -> pd.DataFrame:
    X = df[features].astype(float).copy()
    if platform_specific:
        for p in PLATFORMS[1:]:  # instagram is reference
            d = (df["platform"] == p).astype(float)
            X[f"is_{p}"] = d
            for f in features:
                X[f"{f}:{p}"] = df[f].astype(float) * d
    return X


class ConversionModel:
    def __init__(self, features: list[str] | None = None, platform_specific: bool = True, l2: float = 1.0):
        self.features = features or FEATURES
        self.platform_specific = platform_specific
        self.l2 = l2

    def fit(self, df: pd.DataFrame, target: str = "converted") -> "ConversionModel":
        X = _design(df, self.features, self.platform_specific)
        self.columns = list(X.columns)
        self.res = fit_logit(X, df[target].to_numpy(), l2=self.l2)
        return self

    def coefficients_by_platform(self) -> pd.DataFrame:
        t = self.res.table()
        rows = []
        for p in PLATFORMS:
            for f in self.features:
                b, se = t.loc[f, "coef"], t.loc[f, "se"]
                if self.platform_specific and p != "instagram":
                    k = f"{f}:{p}"
                    b = b + t.loc[k, "coef"]
                    se = float(np.sqrt(se ** 2 + t.loc[k, "se"] ** 2))  # conservative (ignores covariance)
                rows.append({"platform": p, "feature": f, "beta": b, "se": se,
                             "ci_low": b - 1.645 * se, "ci_high": b + 1.645 * se})
        return pd.DataFrame(rows)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = _design(df, self.features, self.platform_specific).reindex(columns=self.columns, fill_value=0.0)
        return self.res.predict(X.to_numpy())

    def predict(self, row: pd.DataFrame, data_quality: float = 0.7, draws: int = 2000, seed: int = 0) -> Prediction:
        X = _design(row, self.features, self.platform_specific).reindex(columns=self.columns, fill_value=0.0)
        z = np.concatenate([[1.0], X.to_numpy()[0]])
        rng = np.random.default_rng(seed)
        bs = rng.normal(self.res.coef, self.res.se, (draws, len(z)))
        ps = sigmoid(bs @ z)
        p = float(sigmoid(z @ self.res.coef))
        lo, hi = np.quantile(ps, [0.05, 0.95])
        contrib = dict(zip(self.res.names[1:], (self.res.coef[1:] * z[1:]).tolist()))
        return Prediction(target="p_conversion", prediction=p, lower=float(lo), upper=float(hi),
                          confidence=confidence_label(hi - lo, data_quality, 1.0), data_quality=data_quality,
                          model_agreement=1.0, explanation=Explanation.from_contributions(contrib, "logit b*x"))


def benchmark_conversion_models(df: pd.DataFrame, target: str = "converted", seed: int = 0) -> pd.DataFrame:
    """Logistic (pooled) vs logistic (platform-specific, the DSDVC spec) vs gradient boosting,
    5-fold CV (spec §60). The complex model must earn its complexity."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = df[target].astype(int).to_numpy()
    res = {"logistic_pooled": [], "dsdvc_platform_specific": [], "tree": []}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y):
        a, b = df.iloc[tr], df.iloc[te]
        res["logistic_pooled"].append(ConversionModel(platform_specific=False).fit(a, target).predict_proba(b))
        res["dsdvc_platform_specific"].append(ConversionModel(platform_specific=True).fit(a, target).predict_proba(b))
        Xa = _design(a, FEATURES, True)
        Xb = _design(b, FEATURES, True)
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed).fit(Xa, y[tr])
        res["tree"].append(clf.predict_proba(Xb)[:, 1])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y))
    y_all = np.concatenate([y[te] for _, te in folds])
    rows = []
    for k, preds in res.items():
        rows.append({"model": k, **VM.classification_report(y_all, np.concatenate(preds), top_frac=0.1)})
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
