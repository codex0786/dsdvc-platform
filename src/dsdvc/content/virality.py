"""Virality prediction (spec §19, §20, §35, §58).

"Viral" is defined *operationally*: top ``top_pct`` (default 5%) of comparable content
by a chosen outcome (default shares), where comparability = benchmark groups
(platform x creator-size tier x format by default; coarser fallback when a group is
too small). No universal view threshold is used.

Models (all compared out-of-sample, spec §60):
  naive      - group base rate
  logistic   - L2 logistic regression on static features
  tree       - LightGBM (sklearn HistGradientBoosting fallback)
  temporal   - gradient boosting on static + early-velocity features (horizon h)
  ensemble   - convex combination fitted on a temporal validation fold, then isotonic
               calibration. Weights are learned, not assumed.

Prediction output includes interval, confidence, model agreement and top drivers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsdvc import MODEL_VERSION
from dsdvc.content.features import extract_text_features, novelty_scores
from dsdvc.content.stepps import DIMENSIONS, STEPPSAnalyzer
from dsdvc.content.velocity import early_features
from dsdvc.core.literature import cites
from dsdvc.core.prediction import Explanation, Prediction, confidence_label
from dsdvc.core.runtime import available_cpus
from dsdvc.validation import metrics as VM

SIZE_TIERS = [0, 10_000, 100_000, 1_000_000, np.inf]
SIZE_LABELS = ["nano", "micro", "macro", "mega"]


def creator_tier(followers: pd.Series) -> pd.Series:
    return pd.cut(followers.fillna(0), SIZE_TIERS, labels=SIZE_LABELS, right=False).astype(str)


def label_viral(df: pd.DataFrame, outcome: str = "shares", top_pct: float = 0.05,
                keys: tuple[str, ...] = ("platform", "creator_tier", "media_type"), min_group: int = 40) -> pd.Series:
    """Percentile rank within the finest benchmark group having >= ``min_group`` items."""
    d = df.copy()
    if "creator_tier" in keys and "creator_tier" not in d:
        d["creator_tier"] = creator_tier(d["followers"])
    pct = pd.Series(np.nan, index=d.index)
    for depth in range(len(keys), -1, -1):
        ks = list(keys[:depth])
        if ks:
            sizes = d.groupby(ks)[outcome].transform("size")
            ranks = d.groupby(ks)[outcome].rank(pct=True, method="average")
        else:
            sizes = pd.Series(len(d), index=d.index)
            ranks = d[outcome].rank(pct=True, method="average")
        fill = pct.isna() & (sizes >= min_group)
        pct[fill] = ranks[fill]
    pct = pct.fillna(d[outcome].rank(pct=True))
    return (pct > 1 - top_pct).astype(int)


@dataclass
class FeatureBuilder:
    analyzer: STEPPSAnalyzer = field(default_factory=STEPPSAnalyzer)
    reference_texts: list[str] | None = None
    columns_: list[str] | None = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for text, mt in zip(df["text"].fillna(""), df.get("media_type", pd.Series("image", index=df.index))):
            prof = self.analyzer.analyze(text, media_type=mt)
            f = prof.features
            r = {f"stepps_{d}": prof.dimensions[d].score for d in DIMENSIONS}
            r.update({"sentiment": f.sentiment, "arousal": f.arousal, "readability": f.readability,
                      "informativeness": f.informativeness, "cta_strength": f.cta_strength,
                      "controversy": f.controversy, "n_hashtags": len(f.hashtags), "n_tokens": f.n_tokens})
            rows.append(r)
        X = pd.DataFrame(rows, index=df.index)
        X["novelty"] = novelty_scores(df["text"].fillna("").tolist(), self.reference_texts)
        X["log_followers"] = np.log1p(df.get("followers", pd.Series(0, index=df.index)).fillna(0))
        if "creator_engagement_rate" in df:
            X["creator_engagement_rate"] = df["creator_engagement_rate"].fillna(df["creator_engagement_rate"].median())
        ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df else None
        if ts is not None:
            X["hour"] = ts.dt.hour
            X["weekday"] = ts.dt.weekday
        for p in ("instagram", "facebook", "whatsapp"):
            X[f"platform_{p}"] = (df["platform"] == p).astype(float)
        for m in ("reel", "image", "carousel", "video", "text"):
            X[f"media_{m}"] = (df.get("media_type", "") == m).astype(float)
        if self.columns_ is None:
            self.columns_ = list(X.columns)
        return X.reindex(columns=self.columns_, fill_value=0.0)


def _tree_model(seed: int):
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=20,
                                  n_jobs=available_cpus(),
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                  random_state=seed, verbose=-1), "lightgbm"
    except Exception:  # noqa: BLE001
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0,
                                              random_state=seed), "sklearn_hgb"


def _regressor(seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, l2_regularization=1.0, random_state=seed)


@cites("cheng2014", "goel2016", "berger_milkman2012", "lundberg2017")
class ViralityPredictor:
    def __init__(self, outcome: str = "shares", top_pct: float = 0.05, horizon_h: int = 3, seed: int = 0,
                 benchmark_keys: tuple[str, ...] = ("platform", "creator_tier", "media_type")):
        self.outcome, self.top_pct, self.horizon_h, self.seed = outcome, top_pct, horizon_h, seed
        self.benchmark_keys = benchmark_keys
        self.fb = FeatureBuilder()
        self.fitted = False
        self.report: dict = {}

    # ------------------------------------------------------------------
    def _split(self, df):
        order = np.argsort(pd.to_datetime(df["timestamp"]).values) if "timestamp" in df else np.arange(len(df))
        n = len(df)
        tr, va = order[: int(0.6 * n)], order[int(0.6 * n): int(0.8 * n)]
        te = order[int(0.8 * n):]
        return tr, va, te

    def fit(self, df: pd.DataFrame, data_version: str = "unversioned") -> dict:
        df = df.reset_index(drop=True).copy()
        self.data_version = data_version
        df["creator_tier"] = creator_tier(df["followers"])
        y = label_viral(df, self.outcome, self.top_pct, self.benchmark_keys).to_numpy()
        tr, va, te = self._split(df)
        self.fb.reference_texts = df.loc[tr, "text"].fillna("").tolist()
        X = self.fb.transform(df)
        has_early = any(f"{self.outcome}_{h}h" in df.columns for h in (1, 3))
        Xe = pd.concat([X, early_features(df, self.horizon_h, self.outcome)], axis=1) if has_early else None
        self.temporal_cols = list(Xe.columns) if Xe is not None else None

        # naive: base rate per benchmark group on train
        self.base_rate = float(y[tr].mean())
        self.logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(X.iloc[tr], y[tr])
        self.tree, self.tree_kind = _tree_model(self.seed)
        self.tree.fit(X.iloc[tr], y[tr])
        if Xe is not None:
            self.temporal, _ = _tree_model(self.seed + 1)
            self.temporal.fit(Xe.iloc[tr], y[tr])
        else:
            self.temporal = None

        def member_preds(idx):
            P = {"naive": np.full(len(idx), self.base_rate),
                 "logistic": self.logit.predict_proba(X.iloc[idx])[:, 1],
                 "tree": self.tree.predict_proba(X.iloc[idx])[:, 1]}
            if self.temporal is not None:
                P["temporal"] = self.temporal.predict_proba(Xe.iloc[idx])[:, 1]
            return P

        Pva = member_preds(va)
        names = [k for k in Pva if k != "naive"]
        M = np.column_stack([Pva[k] for k in names])

        def ll(w):
            w = np.abs(w) / np.abs(w).sum()
            p = np.clip(M @ w, 1e-6, 1 - 1e-6)
            return -np.mean(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p))

        r = optimize.minimize(ll, x0=np.ones(len(names)) / len(names), method="Nelder-Mead")
        w = np.abs(r.x) / np.abs(r.x).sum()
        self.weights = dict(zip(names, w.round(4).tolist()))
        ens_va = M @ w
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(ens_va, y[va])
        self._va_pairs = np.column_stack([self.calibrator.predict(ens_va), y[va]])

        # ---- test-set evaluation of every model (benchmarks, spec §60)
        Pte = member_preds(te)
        Mte = np.column_stack([Pte[k] for k in names])
        Pte["ensemble_calibrated"] = self.calibrator.predict(Mte @ w)
        rows = []
        for k, p in Pte.items():
            rows.append({"model": k, **VM.classification_report(y[te], p, top_frac=self.top_pct)})
        self.report = {"benchmarks": pd.DataFrame(rows), "weights": self.weights, "tree_backend": self.tree_kind,
                       "n_train": len(tr), "n_val": len(va), "n_test": len(te), "positives_train": int(y[tr].sum()),
                       "viral_definition": f"top {self.top_pct:.0%} of '{self.outcome}' within benchmark group {self.benchmark_keys}",
                       "calibration_curve": VM.calibration_curve(y[te], Pte["ensemble_calibrated"], 8)}
        # ---- auxiliary outcome regressors (reach, shares, conversion, retention)
        self.regressors = {}
        for target in ("views", "shares", "conversions"):
            if target in df:
                reg = _regressor(self.seed).fit(X.iloc[tr], np.log1p(df.loc[tr, target]))
                resid = np.log1p(df.loc[va, target]) - reg.predict(X.iloc[va])
                self.regressors[target] = (reg, float(np.quantile(resid, 0.05)), float(np.quantile(resid, 0.95)))
        if {"conversions", "views"} <= set(df.columns):
            rate = df["conversions"] / df["views"].clip(lower=1)
            reg = _regressor(self.seed).fit(X.iloc[tr], rate.iloc[tr])
            res = rate.iloc[va] - reg.predict(X.iloc[va])
            self.regressors["conversion_rate"] = (reg, float(np.quantile(res, 0.05)), float(np.quantile(res, 0.95)))
        if {"retained_90d", "conversions"} <= set(df.columns):
            ok = df["conversions"] > 0
            rr = (df["retained_90d"] / df["conversions"].clip(lower=1))
            idx = [i for i in tr if ok.iloc[i]]
            if len(idx) > 50:
                reg = _regressor(self.seed).fit(X.iloc[idx], rr.iloc[idx])
                self.regressors["retention_rate"] = (reg, -0.15, 0.15)
        self._X_train_mean = X.iloc[tr].mean()
        self._feature_names = list(X.columns)
        self._shap = None
        self.fitted = True
        return self.report

    # ------------------------------------------------------------------
    def _local_interval(self, p: float, k: int = 60, level: float = 0.9) -> tuple[float, float, int]:
        """Beta posterior on hit rate among the k validation items with the closest calibrated score."""
        pairs = self._va_pairs
        idx = np.argsort(np.abs(pairs[:, 0] - p))[:k]
        pos = pairs[idx, 1].sum()
        a = (1 - level) / 2
        lo, hi = stats.beta.ppf([a, 1 - a], 0.5 + pos, 0.5 + len(idx) - pos)
        return float(lo), float(hi), int(pairs[:, 1].sum())

    def _contributions(self, X: pd.DataFrame) -> tuple[np.ndarray, str]:
        """Per-row feature contributions for all rows at once (SHAP if installed, else logistic)."""
        try:
            import shap
            if self._shap is None:
                self._shap = shap.TreeExplainer(self.tree)
            sv = self._shap.shap_values(X)
            sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
            if sv.ndim == 3:
                sv = sv[..., 1]
            return sv, "SHAP (tree model, log-odds)"
        except Exception:  # noqa: BLE001
            lr, sc = self.logit[-1], self.logit[0]
            Z = (X.to_numpy(dtype=float) - sc.mean_) / sc.scale_
            return Z * lr.coef_[0][None, :], "logistic coefficient x standardized value"

    def _explain(self, X: pd.DataFrame) -> Explanation:
        C, method = self._contributions(X.iloc[[0]])
        return Explanation.from_contributions(dict(zip(self._feature_names, C[0].tolist())), method)

    def predict(self, df: pd.DataFrame, data_quality: float = 0.7) -> list[dict]:
        if not self.fitted:
            raise RuntimeError("fit() first")
        df = df.reset_index(drop=True).copy()
        X = self.fb.transform(df)
        out = []
        has_early = self.temporal is not None and all(c in df for c in [f"{self.outcome}_1h"])
        Xe = pd.concat([X, early_features(df, self.horizon_h, self.outcome)], axis=1)[self.temporal_cols] if has_early else None
        # batched model calls (one call per model, not per row)
        members_all = {"logistic": self.logit.predict_proba(X)[:, 1], "tree": self.tree.predict_proba(X)[:, 1]}
        if "temporal" in self.weights and Xe is not None:
            members_all["temporal"] = self.temporal.predict_proba(Xe)[:, 1]
        ws = {k: self.weights[k] for k in members_all}
        tot = sum(ws.values()) or 1
        raw_all = sum(members_all[k] * ws[k] / tot for k in members_all)
        p_all = self.calibrator.predict(raw_all)
        contrib_all, expl_method = self._contributions(X)
        reg_all = {t: reg.predict(X) for t, (reg, _, _) in self.regressors.items()}
        for i in range(len(df)):
            members = {k: float(v[i]) for k, v in members_all.items()}
            p = float(p_all[i])
            lo, hi, _ = self._local_interval(p)
            spread = float(np.std(list(members.values())))
            agreement = float(np.clip(1 - spread / 0.25, 0, 1))
            expl = Explanation.from_contributions(dict(zip(self._feature_names, contrib_all[i].tolist())), expl_method)
            conf = confidence_label(hi - lo, data_quality, agreement)
            pred = Prediction(target="p_viral_top_pct", prediction=p, lower=min(lo, p), upper=max(hi, p),
                              confidence=conf, data_quality=data_quality, model_agreement=agreement,
                              explanation=expl, data_version=getattr(self, "data_version", "unversioned"),
                              conditional_on=f"historical benchmark ({self.report['viral_definition']}); "
                                             f"{'early velocity at ' + str(self.horizon_h) + 'h' if 'temporal' in members else 'pre-launch features only'}",
                              main_uncertainty="limited positive examples near this score" if hi - lo > 0.3 else None,
                              extra={"members": members, "used_temporal": "temporal" in members})
            expected = {}
            for t, (reg, lo_r, hi_r) in self.regressors.items():
                mu = float(reg_all[t][i])
                if t in ("views", "shares", "conversions"):
                    expected[f"expected_{t}"] = {"prediction": float(np.expm1(mu)), "lower": float(np.expm1(mu + lo_r)),
                                                 "upper": float(np.expm1(mu + hi_r))}
                else:
                    expected[f"expected_{t}"] = {"prediction": float(np.clip(mu, 0, 1)),
                                                 "lower": float(np.clip(mu + lo_r, 0, 1)),
                                                 "upper": float(np.clip(mu + hi_r, 0, 1))}
            expected["expected_cascade_depth"] = {"status": "DATA_UNAVAILABLE",
                                                  "reason": "requires reshare-tree data; use CascadeEngine on authorised share events or simulation"}
            out.append({"content_id": df.get("content_id", pd.Series(range(len(df)))).iloc[i],
                        "viral": pred.model_dump(mode="json"), "summary": pred.summary(),
                        "drivers_text": expl.text(), **expected, "model_version": MODEL_VERSION})
        return out
