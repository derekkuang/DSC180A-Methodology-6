"""Drop features, refit, and report the validation delta. Never scores test.

Run: python -m src.ablation cnt_max_7
     python -m src.ablation temp atemp hum windspeed weather_severity
"""
import json
import sys

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import PROCESSED, TARGET
from src.features import feature_columns
from src.train import MODELS, temporal_split

REFIT = ("ridge", "hgb")


def ablate(drop: list[str]) -> dict:
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    unknown = set(drop) - set(cols)
    if unknown:
        raise ValueError(f"not in the feature set: {sorted(unknown)}; features are {cols}")
    kept = [c for c in cols if c not in drop]
    train, valid, _ = temporal_split(df)

    def score(name, features):
        model = clone(MODELS[name]).fit(train[features], train[TARGET])
        pred = model.predict(valid[features])
        return round(float(r2_score(valid[TARGET], pred)), 4), round(float(mean_absolute_error(valid[TARGET], pred)), 1)

    out = {"dropped": drop, "n_features": [len(cols), len(kept)], "split": "valid", "n_valid": len(valid)}
    # The baselines do not depend on the dropped columns unless cnt_lag_1 is
    # among them, so they are scored once on the full set for reference.
    for name in ("baseline_mean", "baseline_persistence"):
        out[name] = dict(zip(("r2", "mae"), score(name, cols)))
    for name in REFIT:
        (r2_all, mae_all), (r2_drop, mae_drop) = score(name, cols), score(name, kept)
        out[name] = {"r2_all": r2_all, "r2_without": r2_drop, "r2_delta": round(r2_drop - r2_all, 4),
                     "mae_all": mae_all, "mae_without": mae_drop}
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.ablation <feature> [<feature> ...]")
    res = ablate(sys.argv[1:])
    print(json.dumps(res, indent=2))
    deltas = ", ".join(f"{m} {res[m]['r2_delta']:+.4f}" for m in REFIT)
    print(f"OK ablation: without {', '.join(sys.argv[1:])}: valid R^2 {deltas}")
