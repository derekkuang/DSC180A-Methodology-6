"""The tests that make the agent honest.

test_no_forbidden_features is the single most valuable test in this repo:
casual + registered == cnt exactly, so any model handed those columns
scores R^2 = 1.0 and looks brilliant while having learned nothing.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import FORBIDDEN, LEAKAGE_COLS, TARGET
from src.features import build_features, feature_columns
from src.ingest import load_raw


@pytest.fixture(scope="module")
def feats():
    return build_features(load_raw())


def test_leakage_identity_holds(feats):
    """Documents *why* casual and registered are forbidden."""
    assert (feats.casual + feats.registered == feats[TARGET]).all()


def test_no_forbidden_features(feats):
    """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
    construction, so this cannot discover a *new* leaky column. What it
    catches is someone editing feature_columns() or FORBIDDEN -- which is
    exactly the fault the Step 7 hook demonstrates."""
    cols = set(feature_columns(feats))
    assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"


def test_no_feature_restates_a_leakage_column(feats):
    """Check values, not names. test_no_forbidden_features only sees names,
    so `out["riders"] = out.casual + out.registered` passes it. Here:

      * no single feature may rank-correlate with cnt, casual or registered
        at |rho| > 0.999 (catches monotone restatements such as log(cnt));
      * cnt, casual and registered must each be unrecoverable as a linear
        combination of *all* features at once: the least-squares fit must
        miss by at least half a ride on some day. They are integers, so a
        fit inside 0.5 everywhere recovers them exactly by rounding.
    """
    cols = feature_columns(feats)
    X = np.column_stack([feats[cols].to_numpy(dtype=float), np.ones(len(feats))])
    for target in sorted(LEAKAGE_COLS):
        y = feats[target].astype(float)
        rho = feats[cols].corrwith(y, method="spearman").abs()
        assert not (rho > 0.999).any(), f"{list(rho[rho > 0.999].index)} restate {target!r}"
        coef, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
        worst = float(np.abs(X @ coef - y.to_numpy()).max())
        assert worst >= 0.5, f"features jointly restate {target!r} (max residual {worst:.2g})"


def test_lags_are_true_calendar_lags(feats):
    """cnt_lag_1 for date d must equal cnt on d-1, in calendar days."""
    by_date = feats.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_lag_1.items():
        prev = date - pd.Timedelta(days=1)
        if prev in by_date.index:
            assert lag == pytest.approx(by_date[prev]), f"bad lag at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified"


def test_lag_is_not_the_target(feats):
    assert not (feats.cnt_lag_1 == feats[TARGET]).all()


def test_rolling_mean_excludes_today(feats):
    """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
    by_date = feats.set_index("dteday")[TARGET]
    date = feats.dteday.iloc[200]
    window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
    if all(d in by_date.index for d in window):
        expected = sum(by_date[d] for d in window) / 7
        assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)


def test_rolling_max_excludes_today(feats):
    """cnt_max_7 on date d must be max(cnt) over d-7 .. d-1 in calendar days.

    Checked on every row whose full window is present, so a rows-not-days
    shift across the 2011-03-10 gap fails. The window must exclude d itself:
    we also require days where cnt[d] beats the prior 7, so a version that
    forgot .shift(1) cannot pass by coincidence.
    """
    by_date = feats.set_index("dteday")[TARGET]
    max7 = feats.set_index("dteday")["cnt_max_7"]

    # Hand-computed from data/raw/bike_daily.csv. 2011-10-29 was the
    # nor'easter (BAD, 627 rides); the 7 days before 2011-10-30 are
    # 4381, 4187, 4687, 3894, 2659, 3747, 627, so the max is 4687
    # (2011-10-25), while cnt_lag_1 is 627.
    assert max7[pd.Timestamp("2011-10-30")] == 4687
    assert feats.set_index("dteday").cnt_lag_1[pd.Timestamp("2011-10-30")] == 627

    checked = discriminating = 0
    for date, value in max7.items():
        window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
        if not all(d in by_date.index for d in window):
            continue
        expected = max(by_date[d] for d in window)
        assert value == pytest.approx(expected), f"bad cnt_max_7 at {date.date()}"
        checked += 1
        discriminating += by_date[date] > expected
    assert checked > 600, "too few rows actually verified"
    assert discriminating > 0, "no row distinguishes the window from one including today"


def test_no_missing_values(feats):
    assert feats[feature_columns(feats)].isna().sum().sum() == 0


def test_dates_are_sorted_and_unique(feats):
    assert feats.dteday.is_monotonic_increasing
    assert feats.dteday.is_unique
