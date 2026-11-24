# Seeded faults (Exercise 2)

**Setup.** Claude Opus 5.5 (`claude-opus-5-5`) in Claude Code 2.1.331, in
this repo. Each fault was seeded with the agent's `Edit` tool, committed in
its broken state, and then fixed in a separate commit (`git log` shows the
pairs). Everything below is pasted output.

**How the hook was run.** The starter's PostToolUse hook
(`.claude/hooks/check_leakage.sh`) did not fire inside this session: the
`Edit` results came back with no hook output. So after every edit the agent
ran the hook script itself, piping it the same JSON payload Claude Code
sends and setting `CLAUDE_PROJECT_DIR`:

```bash
echo '{"tool_name":"Edit","tool_input":{"file_path":"<repo>/src/features.py"}}' \
  | CLAUDE_PROJECT_DIR=<repo> .claude/hooks/check_leakage.sh
```

The stderr and exit codes below are that script's real output. Exit 2 is the
code Claude Code turns into a blocking message to the agent. The hook only
runs `tests/test_leakage.py`, so its counts are lower than `make test`'s.

**How turns are counted.** One turn is one agent tool call, counted from
the hook's stderr arriving to the suite going green again. The same session
seeded each fault, so it already knew the cause. These counts are a lower
bound for an agent meeting the failure cold.

---

## Exercise 1: the feature's test, written first (commit `465b3fa`)

`test_rolling_max_excludes_today` was committed before `cnt_max_7` existed.
`make test` at that commit:

```text
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_rolling_max_excludes_today - KeyError: 'cn...
1 failed, 16 passed in 1.56s
make: *** [test] Error 1
```

The feature (commit `744758a`) made it pass: 17 passed.

---

## Fault 1: a forbidden column added to `feature_columns`

**Change** (commit `db608b5`):

```diff
diff --git a/src/features.py b/src/features.py
index 3f3f75e..e34e0ee 100644
--- a/src/features.py
+++ b/src/features.py
@@ -84,7 +84,7 @@ def build_features(df: pd.DataFrame) -> pd.DataFrame:
 def feature_columns(df: pd.DataFrame) -> list[str]:
     """Numeric columns that are safe to feed the model."""
     numeric = df.select_dtypes(include="number").columns
-    return [c for c in numeric if c not in FORBIDDEN]
+    return [c for c in numeric if c not in FORBIDDEN] + ["casual"]
 
 
 if __name__ == "__main__":
```

**Hook stderr** (exit 2):

```text
Leakage tests failed after editing /Users/derekkuang/DSC180A-Methodology-6/src/features.py:
            combination of *all* features at once: the least-squares fit must
            miss by at least half a ride on some day. They are integers, so a
            fit inside 0.5 everywhere recovers them exactly by rounding.
        """
        cols = feature_columns(feats)
        X = np.column_stack([feats[cols].to_numpy(dtype=float), np.ones(len(feats))])
        for target in sorted(LEAKAGE_COLS):
            y = feats[target].astype(float)
            rho = feats[cols].corrwith(y, method="spearman").abs()
>           assert not (rho > 0.999).any(), f"{list(rho[rho > 0.999].index)} restate {target!r}"
E           AssertionError: ['casual'] restate 'casual'
E           assert not np.True_
E            +  where np.True_ = any()
E            +    where any = temp                0.654315\natemp               0.654601\nhum                 0.089767\nwindspeed           0.175294\nda...         0.552581\ncnt_roll_7          0.646500\ncnt_max_7           0.617063\ncasual              1.000000\ndtype: float64 > 0.999.any

tests/test_leakage.py:51: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
FAILED tests/test_leakage.py::test_no_feature_restates_a_leakage_column - Ass...
2 failed, 7 passed in 1.70s
```

**`make test`:** the hook's two tests, plus my contract test, go red:

```text
E       AssertionError: raw columns used as features without role 'model': ['casual']
E       assert not {'casual'}
E       AssertionError: forbidden columns in feature set: {'casual'}
E       assert not ({'atemp', 'casual', 'cnt_lag_1', 'cnt_lag_7', 'cnt_max_7', 'cnt_roll_7', ...} & {'casual', 'cnt', 'cnt_2d_bfr', 'days_since_2011', 'instant', 'registered', ...})
E           AssertionError: ['casual'] restate 'casual'
E           assert not np.True_
=========================== short test summary info ============================
FAILED tests/test_contract.py::test_every_raw_column_is_classified - Assertio...
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
FAILED tests/test_leakage.py::test_no_feature_restates_a_leakage_column - Ass...
3 failed, 14 passed in 1.25s
make: *** [test] Error 1
```

**Fix:** 1 turn, one `Edit` deleting `+ ["casual"]` (commit `a5b90a4`).
Hook exit 0, `make test` 17 passed.

---

## Fault 2: `.shift(1)` deleted from `cnt_roll_7`

**Change** (commit `9a63574`):

```diff
diff --git a/src/features.py b/src/features.py
index 3f3f75e..cb23494 100644
--- a/src/features.py
+++ b/src/features.py
@@ -73,7 +73,7 @@ def build_features(df: pd.DataFrame) -> pd.DataFrame:
     # If you add a longer lag, raise LAG_WARMUP in src/config.py.
     out["cnt_lag_1"] = out[TARGET].shift(1)
     out["cnt_lag_7"] = out[TARGET].shift(LAG_WARMUP)
-    out["cnt_roll_7"] = out[TARGET].shift(1).rolling(LAG_WARMUP).mean()
+    out["cnt_roll_7"] = out[TARGET].rolling(LAG_WARMUP).mean()
     # Recent peak: one storm day drags cnt_lag_1 and cnt_roll_7 down, but
     # not the max of the previous week, which tracks the demand level.
     out["cnt_max_7"] = out[TARGET].shift(1).rolling(LAG_WARMUP).max()
```

**Hook stderr** (exit 2):

```text
Leakage tests failed after editing /Users/derekkuang/DSC180A-Methodology-6/src/features.py:
[714 rows x 29 columns]

    def test_rolling_mean_excludes_today(feats):
        """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
        by_date = feats.set_index("dteday")[TARGET]
        date = feats.dteday.iloc[200]
        window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
        if all(d in by_date.index for d in window):
            expected = sum(by_date[d] for d in window) / 7
>           assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)
E           assert np.float64(4389.0) == 4414.857142857143 ± 0.00441486
E             
E             comparison failed
E             Obtained: 4389.0
E             Expected: 4414.857142857143 ± 0.00441486

tests/test_leakage.py:80: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 8 passed in 0.87s
```

**`make test`:**

```text
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 16 passed in 1.22s
make: *** [test] Error 1
```

**Fix:** 1 turn, one `Edit` restoring `.shift(1)` (commit `50a6205`). Hook
exit 0, `make test` 17 passed.

The starter's `test_rolling_mean_excludes_today` caught this, but only by
checking a single date (`iloc[200]`), and inside an `if` that skips the
check when that date's window has a gap. My own feature's test checks every
row and fails rather than skipping.

---

## Fault 3: the weather encoder's `raise` removed

**Change** (commit `e8dd5d5`):

```diff
diff --git a/src/features.py b/src/features.py
index 3f3f75e..ba5117e 100644
--- a/src/features.py
+++ b/src/features.py
@@ -26,8 +26,6 @@ def encode_weather(series: pd.Series) -> pd.Series:
     """
     known = series.dropna()
     unknown = set(known.unique()) - set(WEATHER_ORDER)
-    if unknown:
-        raise ValueError(f"unmapped weather labels: {sorted(unknown)}")
     return series.map(WEATHER_ORDER).astype(float)
```

**It stayed green.** Hook exit 0, with no stderr. `make test`:

```text
python -m pytest tests/ -q
.................                                                        [100%]
17 passed in 1.19s
```

Nothing in the suite gives the encoder a label it does not know. Without
the `raise`, an unknown label maps to NaN and `dropna()` deletes the day.
If upstream renamed `BAD` to `RAIN`, the 20 bad-weather days would disappear
with every test green.

**New test** (commit `231bbfa`): `test_unseen_weather_label_raises`, in
`tests/test_leakage.py` so the hook runs it. It relabels the 20 BAD days as
`RAIN` and requires `build_features` to raise. Against the seeded fault:

**Hook stderr** (exit 2):

```text
Leakage tests failed after editing /Users/derekkuang/DSC180A-Methodology-6/src/features.py:
    def test_unseen_weather_label_raises():
        """A label the encoder does not know must stop the pipeline.
    
        The data only holds GOOD / MISTY / BAD, so nothing else exercises this
        path. Without the raise, a relabelled day maps to NaN and dropna()
        deletes it with every other test green: the 20 BAD days are exactly the
        ones that would vanish if upstream changed "BAD" to "RAIN".
        """
        raw = load_raw()
        bad = raw.weather == "BAD"
        assert bad.sum() == 20, "fixture changed: expected 20 BAD days"
        raw.loc[bad, "weather"] = "RAIN"
>       with pytest.raises(ValueError, match="unmapped weather labels"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ValueError

tests/test_leakage.py:126: Failed
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_unseen_weather_label_raises - Failed: DID ...
1 failed, 9 passed in 1.22s
```

**`make test`:**

```text
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_unseen_weather_label_raises - Failed: DID ...
1 failed, 17 passed in 1.36s
make: *** [test] Error 1
```

**Fix:** 1 turn, one `Edit` restoring the two lines (commit `95ddd6d`). Hook
exit 0, `make test` 18 passed.

---

## Fault 4: a `cp` into `data/raw/` from Bash

**What happened.** I imputed the missing 2011-03-10 (each value the mean of
03-09 and 03-11, with `cnt = casual + registered` kept exact). I wrote the
result to a scratch file and copied it over the raw data from Bash:

```text
$ cp <scratch>/bike_daily_imputed.csv data/raw/bike_daily.csv
cp exit=0
afc1f750ede328422e6c64542f00bfeed40037a172c2b187a0621d8601db78b0  data/raw/bike_daily.csv
OK ingest: 729 rows, 2011-01-03 -> 2012-12-31
OK features: 722 rows, 15 features -> features.parquet
18 passed in 1.16s
n_train 356 n_valid 182
baseline_mean valid r2 -0.9651
baseline_persistence valid r2 0.4507
ridge valid r2 0.7044
hgb valid r2 0.1133
```

Nothing stopped it:

- **Deny rules:** `Write(./data/raw/**)` and `Edit(./data/raw/**)` only
  govern those two tools, not a shell `cp`.
- **The PostToolUse hook:** it matches `Edit|Write`, so a Bash command
  never triggers it.
- **The tests:** all 18 passed. Every check reads the file as it now is.
  The row-loss bound even got *looser*, because there was no longer a
  missing day.

**The fabricated day changed the numbers.** It added 8 training rows (348 →
356): the day itself, plus the 7 days after it that had been dropped because
their lag windows contained the gap. Three of the four validation scores
moved: mean −0.9318 → −0.9651, ridge 0.7072 → 0.7044, hgb 0.1428 → 0.1133.
I restored the file with `git checkout -- data/raw/` (hash back to
`a6bed1577f6e4de5…`). The broken state was never committed.

**What would have caught it, and what I implemented** (commit `9b8a234`):

1. **A hash check (implemented).** `RAW_SHA256` in `src/config.py` pins the
   bytes. `make data` and `tests/test_contract.py::test_raw_data_is_unchanged`
   both check them. This catches a change made by *any* route: Bash, a
   Python script, a merge, an editor. Re-seeding the same `cp` in a scratch
   copy of the repo:

```text
ValueError: bike_daily.csv sha256 is afc1f750ede32842..., expected a6bed1577f6e4de5... Restore it with `git checkout -- data/raw/`; clean data in code, downstream.
make: *** [data] Error 1
        passed every other test, added 8 training rows and moved three of four
E           ValueError: bike_daily.csv sha256 is afc1f750ede32842..., expected a6bed1577f6e4de5... Restore it with `git checkout -- data/raw/`; clean data in code, downstream.
FAILED tests/test_contract.py::test_raw_data_is_unchanged - ValueError: bike_...
1 failed, 18 passed in 1.15s
```

2. **A PreToolUse hook on Bash (implemented).** `.claude/hooks/protect_raw.py`,
   registered in `.claude/settings.json`, runs *before* the command and exits
   2 if it would write into `data/raw/`. It checks for `cp`/`mv`/`rsync`
   with a destination there, a `>` redirect, `tee`, `sed -i`, `rm`, or
   `dd of=`. Reads (`head`, `shasum`, copying out) and
   `git checkout -- data/raw/` are allowed. The fault-4 command fed to it:

```text
exit=2 | cp <scratch>/bike_daily_imputed.csv data/raw/bike_daily.csv
    Blocked: this command `cp` writes into data/raw/. Raw data is immutable; do any cleaning in src/ (downstream of load_raw). To undo a change, run `git checkout -- data/raw/`.
exit=0 | head data/raw/bike_daily.csv
exit=0 | cp data/raw/bike_daily.csv /tmp/copy.csv
exit=0 | git checkout -- data/raw/
```

   The hook screens shell commands only. A Python script that opens the file
   for writing gets past it, which is why the hash check is the real guard.

3. **Not implemented:** making the file read-only on disk
   (`chmod a-w data/raw/bike_daily.csv`). Git does not preserve that across a
   clone, so it protects one checkout, not the repository.
