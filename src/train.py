"""Stage 3: temporal split, fit, evaluate, write metrics.

Two baselines and two models, on purpose.

  baseline_mean         predict the 2011 average for every day
  baseline_persistence  predict yesterday's count (cnt_lag_1)
  ridge                 standardised linear model on the audited features
  hgb                   gradient-boosted trees on the same features

The mean baseline shows what "the distribution moved" costs. Persistence
is the baseline that actually matters: ridge's features are dominated by
lagged demand, so if it cannot beat "yesterday again" it has learned
nothing worth reporting. The booster is there because it fails, and the
reason it fails is the headline result.

Run: python -m src.train

Exercise 4 (never part of `make all`; reports/metrics.json stays untransformed):
  python -m src.train --target-transform ratio --anchor cnt_max_7
fits ridge and hgb on a re-expressed target and scores validation only,
next to both baselines and the anchor alone. --test scores test as well; it
is run once, for the transform chosen on validation.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import METRICS, PROCESSED, TARGET, TRAIN_END, VALID_END
from src.features import feature_columns


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Predict a single lagged column unchanged. Nothing is fit."""

    def __init__(self, column: str = "cnt_lag_1"):
        self.column = column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"persistence baseline needs column {self.column!r}")
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


class RelativeTarget(BaseEstimator, RegressorMixin):
    """Fit `model` on a transformed target, predict on the original scale.

      log:    learn log(cnt)              predict exp(f(X))
      ratio:  learn cnt / anchor          predict anchor * f(X)
      diff:   learn cnt - anchor          predict anchor + f(X)

    The anchor is a lag column (built with shift(1)), so it is known at
    prediction time. A tree predicts only values it saw in training; with an
    anchor it predicts a relative move and the anchor carries the level.
    """

    def __init__(self, model=None, kind: str = "ratio", anchor: str = "cnt_max_7"):
        self.model = model
        self.kind = kind
        self.anchor = anchor

    def _anchor(self, X):
        return X[self.anchor].to_numpy(dtype=float)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        z = {"log": lambda: np.log(y),
             "ratio": lambda: y / self._anchor(X),
             "diff": lambda: y - self._anchor(X)}[self.kind]()
        self.model_ = clone(self.model).fit(X, z)
        return self

    def predict(self, X):
        z = self.model_.predict(X)
        if self.kind == "log":
            return np.exp(z)
        return self._anchor(X) * z if self.kind == "ratio" else self._anchor(X) + z


MODELS = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "baseline_persistence": PersistenceBaseline("cnt_lag_1"),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "hgb": HistGradientBoostingRegressor(random_state=0, max_iter=300),
}


def temporal_split(df: pd.DataFrame):
    """Split by date, never at random. A random split on a time series
    puts tomorrow in train and today in test."""
    train = df[df.dteday <= TRAIN_END]
    valid = df[(df.dteday > TRAIN_END) & (df.dteday <= VALID_END)]
    test = df[df.dteday > VALID_END]
    return train, valid, test


def fit_and_score(models: dict = MODELS, splits=("valid", "test")) -> dict:
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)

    results = {
        "n_features": len(cols),
        "features": cols,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
    }
    for name, model in models.items():
        model.fit(train[cols], train[TARGET])
        results[name] = {
            split: {
                "r2": round(float(r2_score(part[TARGET], model.predict(part[cols]))), 4),
                "mae": round(float(mean_absolute_error(part[TARGET], model.predict(part[cols]))), 1),
            }
            for split, part in [("valid", valid), ("test", test)]
            if split in splits
        }
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-transform", choices=["log", "ratio", "diff"])
    ap.add_argument("--anchor", choices=["cnt_lag_1", "cnt_roll_7", "cnt_max_7"], default="cnt_max_7")
    ap.add_argument("--test", action="store_true", help="also score test (Exercise 4: once, at the end)")
    ns = ap.parse_args()
    METRICS.parent.mkdir(parents=True, exist_ok=True)

    if ns.target_transform is None:
        res = fit_and_score()
        out = METRICS
    else:
        tag = ns.target_transform if ns.target_transform == "log" else f"{ns.target_transform}_{ns.anchor}"
        models = {name: MODELS[name] for name in ("baseline_mean", "baseline_persistence")}
        if ns.target_transform != "log":
            models[f"anchor_only_{ns.anchor}"] = PersistenceBaseline(ns.anchor)
        for name in ("ridge", "hgb"):
            models[f"{name}_{tag}"] = RelativeTarget(MODELS[name], ns.target_transform, ns.anchor)
        res = fit_and_score(models, ("valid", "test") if ns.test else ("valid",))
        out = METRICS.parent / f"target_{tag}{'_test' if ns.test else ''}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=2))
