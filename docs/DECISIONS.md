# Decisions

One line per methodological decision: what you chose, why, and what it changed.
Agents append here via the `/new-feature` skill; you append here by hand.

- Exclude `casual`, `registered`: they sum to `cnt` exactly. Leaky model R^2 1.000 -> honest ridge 0.583.
- Exclude `cnt_2d_bfr`: a shipped 2-day lag we cannot test the alignment of. Build our own lags instead.
- Exclude `instant`, `yr`, `days_since_2011`: time indexes let a tree predict a flat constant past 2011 and a linear model extrapolate a trend we have no grounds for.
- Split 2011 / H1 2012 / H2 2012 by date. Test set touched once, after all feature work.
- Report mean and persistence baselines with every score. Persistence (test R^2 0.510, MAE 840) is within 0.07 R^2 of ridge and beats it on MAE; that caveat goes in the report.
- 2026-11-23 Add `cnt_max_7`, the max of `cnt` over d-7..d-1 (shifted before rolling, on the calendar grid; no LAG_WARMUP change, no rows lost). Available: every input is a past day's count. Validation R^2: ridge 0.6475 -> 0.7072 (+0.060), hgb 0.2185 -> 0.1428 (-0.076). Kept: ridge is the model we report, and its gain is 6x the 0.01 noise bar; hgb's loss is the same extrapolation failure as before, not a reason to drop it. Test not read.
- 2026-11-23 Every raw column must have a role (`RAW_COLUMN_ROLES` in `src/config.py`, or FORBIDDEN, or `dteday`); an unclassified one fails `make test`. `feature_columns()` is a denylist and fails open: an appended `riders = casual + registered` column reached the model with every leakage test green.
- 2026-11-23 Leakage is judged on values, not names: no feature may rank-correlate with `cnt`/`casual`/`registered` above 0.999, and all features together must not reproduce them by least squares. Checked against three renamed leaks.
- 2026-11-23 Where a seeded fault stays green, add a test. Fault 3 (encoder `raise` removed) passed all 17 tests; `test_unseen_weather_label_raises` now catches it.
- 2026-11-23 Raw data is immutable by any route: a `cp` from Bash bypassed the deny rules and the hook, passed all tests, and moved three validation scores. Pinned the file's sha256 (checked by `make data` and `make test`) and added a PreToolUse Bash hook that blocks writes into `data/raw/`.
- 2026-11-23 Exercise 4, validation only, rule fixed before scoring: pick the transform with the best hgb validation R^2, apply the same one to ridge, and call it defensible only if hgb beats persistence and the anchor alone. Seven candidates (`--target-transform` in `src/train.py`; `make all` unaffected). hgb valid R^2 0.1428 untransformed -> log 0.0524; ratio to cnt_lag_1 / cnt_roll_7 / cnt_max_7 0.5146 / 0.7366 / 0.6556; diff 0.5627 / 0.6940 / 0.6400. Chosen: ratio to cnt_roll_7 (hgb 0.7366, beats persistence 0.4507 and the anchor alone 0.5945). Same transform cuts ridge 0.7072 -> 0.5278, so it suits trees, not models in general; ridge's best is diff to cnt_max_7 (0.7131). Choosing from seven on 182 days will shrink on test.
