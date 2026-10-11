"""Cross-platform transmission matrix (spec §23).

P[a, b] = probability that a share/exposure originating on platform a produces a
transmission onto platform b. Estimated, not hard-coded: each row has a Dirichlet prior
(weakly informative, documented) updated with observed transfer counts. With no data
the posterior equals the prior and the uncertainty is reported as such.

Row outcome categories: [IG, FB, WA, none] - "none" captures shares that did not
transfer, so rows sum to 1 including the no-transfer mass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLAT = ["instagram", "facebook", "whatsapp"]
CATS = PLAT + ["none"]

# Weak prior: pseudo-counts (total 20 per row). Rationale documented in CALIBRATION.md:
# same-platform resharing dominates; IG->WA forwarding of Reels links is plausible; values are
# NOT measured estimates.
DEFAULT_PRIOR = np.array([
    [6.0, 0.6, 1.4, 12.0],
    [0.6, 6.0, 1.4, 12.0],
    [0.6, 0.8, 7.6, 11.0],
])


class TransmissionMatrix:
    def __init__(self, prior: np.ndarray | None = None):
        self.prior = np.array(DEFAULT_PRIOR if prior is None else prior, dtype=float)
        self.counts = np.zeros_like(self.prior)

    def update(self, events: pd.DataFrame) -> "TransmissionMatrix":
        """events: columns source_platform, target_platform (None/'none' for no transfer)."""
        for r in events.itertuples(index=False):
            a = PLAT.index(r.source_platform)
            tgt = r.target_platform if isinstance(r.target_platform, str) and r.target_platform in PLAT else "none"
            self.counts[a, CATS.index(tgt)] += 1
        return self

    def update_counts(self, counts: np.ndarray) -> "TransmissionMatrix":
        self.counts += np.asarray(counts, dtype=float)
        return self

    @property
    def alpha(self) -> np.ndarray:
        return self.prior + self.counts

    def mean(self) -> pd.DataFrame:
        a = self.alpha
        return pd.DataFrame(a / a.sum(axis=1, keepdims=True), index=PLAT, columns=CATS)

    def interval(self, level: float = 0.9, draws: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        lo, hi = np.zeros_like(self.alpha), np.zeros_like(self.alpha)
        q = (1 - level) / 2
        for i, row in enumerate(self.alpha):
            s = rng.dirichlet(row, draws)
            lo[i], hi[i] = np.quantile(s, q, axis=0), np.quantile(s, 1 - q, axis=0)
        return pd.DataFrame(lo, index=PLAT, columns=CATS), pd.DataFrame(hi, index=PLAT, columns=CATS)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.vstack([rng.dirichlet(r) for r in self.alpha])

    def data_weight(self) -> pd.Series:
        """Share of each row's posterior mass that comes from data (0 = prior only)."""
        return pd.Series(self.counts.sum(1) / self.alpha.sum(1), index=PLAT)

    def transfer_matrix(self) -> np.ndarray:
        """3x3 matrix of P(transfer a->b) for b != 'none' (simulator input)."""
        return self.mean().to_numpy()[:, :3]
