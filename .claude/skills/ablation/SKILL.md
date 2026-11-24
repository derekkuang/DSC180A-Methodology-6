---
name: ablation
description: Drop one or more features, refit ridge and hgb, and report the validation R^2/MAE delta next to both baselines. Use when asked whether a feature earns its place, what a group of features contributes, or to ablate/remove/drop a predictor. Never scores the test split.
allowed-tools: Read, Bash(make features*), Bash(python -m src.ablation*)
---

# Ablation: $ARGUMENTS

Current features:

!`python -c "import pandas as pd; from src.features import feature_columns; print(', '.join(feature_columns(pd.read_parquet('data/processed/features.parquet'))))"`

1. **Check the names.** Every name in `$ARGUMENTS` must be in the list above.
   If one is not, stop and say which, instead of guessing a near match.

2. **Rebuild, then ablate.** Run `make features`, then
   `python -m src.ablation $ARGUMENTS`. Do not edit `src/` to drop the
   feature: the module refits a clone of each model on the reduced columns,
   so the pipeline and `reports/metrics.json` are left untouched.

3. **Report in this shape**, from the module's JSON and nothing else:

   | model | valid R^2 with | without | delta | valid MAE with | without |
   |---|---:|---:|---:|---:|---:|

   with one row each for ridge and hgb, then the mean and persistence
   baselines' validation R^2 underneath. Say whether ridge *without* the
   features still beats persistence: that is the comparison that matters.

4. **Judge the delta.** The validation set is 182 days. A |delta| below
   0.01 R^2 is inside the noise: say the features make no measurable
   difference. If dropping cnt_lag_1, say the persistence baseline is
   unaffected (it reads cnt_lag_1 directly, not through the model).

5. **Do not score test.** If asked for a test-set number, decline and
   point to `make all`, which is run once, at the end.

If the result changes a decision (keep or drop a feature), add one dated
line to `docs/DECISIONS.md` with the delta; otherwise record nothing.
