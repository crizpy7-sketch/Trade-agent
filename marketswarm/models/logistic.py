"""L2-regularised logistic regression, numpy only.

Deliberately not a gradient-boosted forest. With a few hundred thousand noisy,
highly correlated financial samples and a signal-to-noise ratio near zero, a
linear model in a well-chosen feature space is hard to beat and — more
importantly — its coefficients are readable. When this model says the gap is
worth -0.3 in log-odds, that is a claim you can argue with. A forest's answer
is not.

It also serialises to a small JSON blob, so the VPS agent loads a trained model
with no extra dependency and no pickle-compatibility landmines.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class StandardScaler:
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, float)
        self.mean = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)
        # A constant feature has no information; scaling it by ~0 would
        # manufacture enormous values from rounding noise.
        self.scale = np.where(sd < 1e-9, 1.0, sd)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if self.mean is None:
            return X
        return (X - self.mean) / self.scale

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist() if self.mean is not None else None,
                "scale": self.scale.tolist() if self.scale is not None else None}

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler":
        s = cls()
        if d.get("mean") is not None:
            s.mean = np.array(d["mean"], float)
            s.scale = np.array(d["scale"], float)
        return s


@dataclass
class LogisticModel:
    """Binary logistic regression trained by L-BFGS-free gradient descent with
    momentum. Small enough to be obvious, robust enough for this data."""

    feature_names: list[str] = field(default_factory=list)
    coef: np.ndarray | None = None
    intercept: float = 0.0
    scaler: StandardScaler = field(default_factory=StandardScaler)
    l2: float = 1.0
    n_train: int = 0
    train_loss: float = float("nan")

    # ---------- training ----------

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 400,
            lr: float = 0.1, momentum: float = 0.9, class_weight: bool = True,
            verbose: bool = False) -> "LogisticModel":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n, p = X.shape
        if n == 0:
            return self

        Z = self.scaler.fit_transform(X)
        w = np.zeros(p)
        b = 0.0
        vw = np.zeros(p)
        vb = 0.0

        # Reweight classes so a 45/55 base rate does not become "always predict
        # the majority" — which scores well on accuracy and is useless.
        if class_weight:
            pos = max(y.sum(), 1.0)
            neg = max(n - y.sum(), 1.0)
            sw = np.where(y > 0.5, n / (2 * pos), n / (2 * neg))
        else:
            sw = np.ones(n)
        sw_sum = sw.sum()

        for epoch in range(epochs):
            z = Z @ w + b
            pred = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            err = (pred - y) * sw

            grad_w = Z.T @ err / sw_sum + self.l2 * w / n
            grad_b = err.sum() / sw_sum

            vw = momentum * vw - lr * grad_w
            vb = momentum * vb - lr * grad_b
            w += vw
            b += vb

            if verbose and epoch % 100 == 0:
                loss = self._loss(Z, y, w, b, sw, sw_sum)
                print(f"  epoch {epoch:4d}  loss {loss:.5f}")

        self.coef = w
        self.intercept = float(b)
        self.n_train = n
        self.train_loss = self._loss(Z, y, w, b, sw, sw_sum)
        return self

    def _loss(self, Z, y, w, b, sw, sw_sum) -> float:
        z = np.clip(Z @ w + b, -30, 30)
        pred = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-9
        ll = -(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps))
        return float((ll * sw).sum() / sw_sum + 0.5 * self.l2 * float(w @ w) / max(len(y), 1))

    # ---------- inference ----------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef is None:
            return np.full(len(np.atleast_2d(X)), 0.5)
        Z = self.scaler.transform(np.atleast_2d(np.asarray(X, float)))
        z = np.clip(Z @ self.coef + self.intercept, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_one(self, x: np.ndarray) -> float:
        return float(self.predict_proba(np.atleast_2d(x))[0])

    # ---------- interpretation ----------

    def importances(self, top: int = 12) -> list[tuple[str, float]]:
        """Coefficients on standardised features — directly comparable, and in
        log-odds per standard deviation, which is a unit you can reason about."""
        if self.coef is None:
            return []
        names = self.feature_names or [f"f{i}" for i in range(len(self.coef))]
        pairs = sorted(zip(names, self.coef.tolist()), key=lambda kv: -abs(kv[1]))
        return pairs[:top]

    def explain(self, x: np.ndarray, top: int = 5) -> list[tuple[str, float]]:
        """Per-prediction contributions, so a published idea can say which
        features actually drove its probability."""
        if self.coef is None:
            return []
        z = self.scaler.transform(np.atleast_2d(np.asarray(x, float)))[0]
        contrib = z * self.coef
        names = self.feature_names or [f"f{i}" for i in range(len(contrib))]
        return sorted(zip(names, contrib.tolist()), key=lambda kv: -abs(kv[1]))[:top]

    # ---------- persistence ----------

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "coef": self.coef.tolist() if self.coef is not None else None,
            "intercept": self.intercept,
            "scaler": self.scaler.to_dict(),
            "l2": self.l2,
            "n_train": self.n_train,
            "train_loss": self.train_loss,
        }

    def save(self, path: Path | str) -> None:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticModel":
        m = cls(
            feature_names=d.get("feature_names", []),
            intercept=float(d.get("intercept", 0.0)),
            l2=float(d.get("l2", 1.0)),
            n_train=int(d.get("n_train", 0)),
            train_loss=float(d.get("train_loss", float("nan"))),
        )
        if d.get("coef") is not None:
            m.coef = np.array(d["coef"], float)
        m.scaler = StandardScaler.from_dict(d.get("scaler", {}))
        return m

    @classmethod
    def load(cls, path: Path | str) -> "LogisticModel | None":
        p = Path(path).expanduser()
        if not p.exists():
            return None
        try:
            return cls.from_dict(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError, KeyError):
            return None
