# Forecasting daily bikeshare demand: what the honest numbers say

Daily Capital Bikeshare rides (`cnt`), Washington DC, 2011–2012, from the
Molnar mirror of the UCI data (728 rows; 2011-03-10 missing). Train on
2011 (348 rows after the 7-day lag warm-up), validate on H1 2012 (182),
test on H2 2012 (184). Every score below is in `reports/metrics.json`,
which `make all` regenerates byte for byte from a clean checkout.

## Results

Validation and test R^2 and MAE (rides/day), before and after my Exercise 1
feature `cnt_max_7`: the highest daily count in the 7 days before *d*.
"Before" is the starter pipeline (`reports/metrics.json` at commit
`92bcaf4`); "after" is the committed final `reports/metrics.json`.

| Feature set | Model | Valid R^2 | Valid MAE | Test R^2 | Test MAE |
|---|---|---:|---:|---:|---:|
| before (starter, 14 features) | baseline_mean | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| before (starter, 14 features) | baseline_persistence | 0.4507 | 900.6 | 0.5099 | 840.1 |
| before (starter, 14 features) | ridge | 0.6475 | 832.9 | 0.5833 | 947.1 |
| before (starter, 14 features) | hgb | 0.2185 | 1296.6 | -0.4975 | 1972.6 |
| after (+ `cnt_max_7`, 15 features) | baseline_mean | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| after (+ `cnt_max_7`, 15 features) | baseline_persistence | 0.4507 | 900.6 | 0.5099 | 840.1 |
| after (+ `cnt_max_7`, 15 features) | ridge | 0.7072 | 731.9 | 0.6335 | 867.2 |
| after (+ `cnt_max_7`, 15 features) | hgb | 0.1428 | 1336.0 | -0.6976 | 2105.6 |

The feature drops no rows (714 before and after), so the baselines are
identical in both halves. I decided to keep it on validation alone (ridge
+0.0597, hgb −0.0757). The "after" test numbers were read once, from the
final `make all`, after all feature work and the Exercise 4 choice were
committed. R^2 is against the test-period mean, so 0 means "as good as a
constant nobody could have known in advance". For reference, the leaky model
(`python -m src.leakage_demo`, with `casual` and `registered`) scores test
R^2 = 1.000000.

## Headline finding

The naive evaluation gives R^2 = 1.000 and is invalid:
`casual + registered == cnt` on every row, so the model learns addition.
Under a temporal split, the picture changes:

- **Mean baseline:** −2.14 on test. The H2 2012 mean (6,043 rides/day)
  is 73% above the 2011 mean (3,493).
- **Persistence ("yesterday again"):** +0.510.
- **Ridge:** +0.634. It beats persistence by 0.12 R^2 but still loses to it
  on MAE (867 vs 840 rides/day).
- **Gradient boosting:** −0.70. It is negative and still far better than
  the mean baseline. A tree cannot predict above the largest value it saw,
  and 56.5% of test days exceed the 2011 maximum (6,043 rides, 2011-07-04).

My feature, the 7-day peak, raised ridge's validation R^2 by 0.060 and its
test R^2 from 0.583 to 0.634, with the baselines unchanged, so I kept it.
It made hgb worse (test −0.50 → −0.70): it is one more level feature that a
tree caps at its 2011 range.

**One assumption carries part of ridge's edge.** Like the starter, the
models see the day's *observed* weather (`temp`, `atemp`, `hum`,
`windspeed`, `weather_severity`). That makes this a nowcast with perfect
weather foresight. Without those five columns
(`python -m src.ablation temp atemp hum windspeed weather_severity`),
ridge falls to 0.553 on validation. That still beats persistence there
(0.451), but on test it scores 0.511 against persistence's 0.510: a tie.

## Exercise 4: target transformation

The tree fails because the target's level moved out of its training range.
So I let it predict a relative move from a lagged anchor that it already
sees as a feature, and the anchor carries the level. `RelativeTarget` in
`src/train.py` (behind `--target-transform` and `--anchor`) learns
`log(cnt)`, `cnt / anchor`, or `cnt − anchor`, then maps predictions back.
`make all` and `reports/metrics.json` are untransformed.

**Rule, fixed before scoring.** Pick the transform with the best hgb
validation R^2. Apply the same transform to ridge. Call it defensible only
if hgb then beats both persistence and the anchor alone. Validation only
(commit `2395ba5`):

| Transform | Anchor alone | Ridge | hgb | hgb MAE |
|---|---:|---:|---:|---:|
| none | — | 0.7072 | 0.1428 | 1336.0 |
| log | — | −0.3816 | 0.0524 | 1415.5 |
| ratio, `cnt_lag_1` | 0.4507 | −0.2741 | 0.5146 | 981.1 |
| **ratio, `cnt_roll_7`** | **0.5945** | **0.5278** | **0.7366** | **712.8** |
| ratio, `cnt_max_7` | 0.2008 | 0.6308 | 0.6556 | 822.2 |
| diff, `cnt_lag_1` | 0.4507 | 0.7067 | 0.5627 | 891.0 |
| diff, `cnt_roll_7` | 0.5945 | 0.7085 | 0.6940 | 767.9 |
| diff, `cnt_max_7` | 0.2008 | 0.7131 | 0.6400 | 810.3 |

A log transform alone does nothing: the tree is still capped, now in log
space. Any anchor helps. Ratio to `cnt_roll_7` won for hgb, and it clears
both bars on validation. I recorded the choice, then **touched test once**
(commit `6e9d890`, `reports/target_ratio_cnt_roll_7_test.json`):

| Test (n = 184) | R^2 | MAE |
|---|---:|---:|
| hgb, untransformed | −0.6976 | 2105.6 |
| **hgb, ratio / `cnt_roll_7`** | **0.6033** | **859.4** |
| ridge, ratio / `cnt_roll_7` | 0.1602 | 1376.0 |
| ridge, untransformed | 0.6335 | 867.2 |
| persistence | 0.5099 | 840.1 |
| anchor alone (`cnt_roll_7`) | 0.4796 | 871.6 |

**Defensible as a forecaster, not as a model comparison.** It is valid
forecasting: the anchor is a shifted, past-only mean, so no future
information enters. And on test, the tree beats its own anchor by 0.12 R^2,
so it is learning real day-to-day movement, not just echoing the anchor.
But it does not show that trees can compete with ridge:

- **It suits one model only.** The same transform wrecks ridge (0.71 →
  0.53 valid, 0.16 test), because ridge on a ratio target turns additive
  weather effects into multiplicative ones.
- **The selection cost showed up.** Choosing from seven candidates on 182
  days overstated hgb by 0.13 (0.737 valid → 0.603 test).
- **It still loses on test.** The anchored tree trails untransformed ridge
  (0.603 vs 0.634) and persistence on MAE (859 vs 840).

The honest reading: the tree's failure was a target-scale problem, and an
anchor fixes most of it. Comparing each model on its own best transform is
a comparison of transforms, not of model classes.

## Exercise 5: ridge vs persistence

| Test, n = 184 | Ridge | Persistence |
|---|---:|---:|
| R^2 / RMSE | 0.634 / 1054 | 0.510 / 1219 |
| MAE / median abs. error | 867 / 796 | 840 / 506 |
| Mean error (bias) | −418 | +16 |
| Days closer to the truth | 44% | 56% |
| MAE, 70 MISTY/BAD days | 982 | 1153 |
| MAE, 114 GOOD days | 797 | 648 |
| MAE, 17 days persistence misses by > 2,000 | 1,800 | 3,019 |

**Ridge is not worth deploying over "yesterday again" as it stands.**

- **Its edge is in the large errors only.** A paired bootstrap over test
  days (5,000 resamples, seed 0) puts ridge's R^2 edge at +0.127, 95%
  interval [+0.016, +0.262], so the gain in squared error is real. The same
  resamples put the MAE difference at +27 rides/day, interval [−85, +136]:
  on a typical day there is no measurable gain. All of the R^2 edge comes
  from the 17 days where persistence misses by more than 2,000 rides,
  weather swings and the day after a storm (ridge MAE 1,800 vs 3,019).
- **It is worse on ordinary days.** On the 114 GOOD-weather days, ridge
  is worse (797 vs 648). Its median error is 796 against 506, and it is
  closer on only 44% of days.
- **The cause is a level bias.** Ridge is fitted on 2011 and runs 418
  rides/day low on test, while persistence is unbiased (+16). The pattern
  was visible on validation before test was read (bias −318), so it is
  not an accident of the test half.
- **The weather it uses is the day's actual weather.** Without the
  same-day columns, ridge's test R^2 is 0.511, level with persistence
  (0.510). A deployed forecast would use a weather forecast, so 0.634 is
  an upper bound.

Ridge earns deployment only where errors cost quadratically, such as
staffing for storm days. Even then, the better design keeps persistence's
level and adds ridge's weather correction. These numbers read the
residuals of the committed models; nothing was refit or re-chosen after
seeing them.

## Attribution

**Model:** Claude Opus 5.5 (`claude-opus-5-5`, Anthropic-hosted, 1M-token
context), driven through Claude Code 2.1.331. This is not the local Ollama
model the lesson describes: `bikeshare-coder` was never built on this
machine, so `make model` and `make check-model` were not run. I used the
same model throughout and did not switch. It ran:

- the main session;
- the headless `/new-feature` session for Exercise 1, steps 1–2;
- the headless `/leakage-auditor` session on my teammate's repo
  (`docs/AUDIT.md`);
- the batch auditor (`AUDIT_MODEL=claude-opus-5-5`);
- one headless `/ablation cnt_lag_7` check of my skill.

The starter's PostToolUse hook did not fire inside the main session. For
the seeded faults, the agent ran the hook script itself with the same
payload after each edit (method in `docs/SEEDED_FAULTS.md`). Exercise 6
(optional) was not attempted.

**Batch audit of this repo (`reports/audit.jsonl`):** leakage `none`;
temporal split, baselines and reproducibility all passed; two findings.

- **"REPORT.md is missing."** It did not exist yet: the required commit
  order puts the audit before the report. This file is it.
- **"Same-day observed weather makes ridge optimistic for a day-ahead
  forecast."** Agreed, and I ran the ablation the auditor asked for. It is
  now stated, with numbers, in the Headline finding and Exercise 5. I kept
  the starter's features and reframed the claim as a nowcast, rather than
  changing the pipeline after reading test.

**Files and functions the agent generated:**

- `src/features.py`: the `cnt_max_7` line.
- `src/config.py`: `RAW_COLUMN_ROLES`, `RAW_SHA256`.
- `src/ingest.py`: `check_raw_unchanged`.
- `src/train.py`: `RelativeTarget`; the `--target-transform`, `--anchor`
  and `--test` flags; `fit_and_score`'s `models` and `splits` arguments.
- `src/ablation.py`.
- `tests/test_leakage.py`: `test_rolling_max_excludes_today` (written by the
  headless `/new-feature` session; I added the hand-computed 2011-10-30
  assert), `test_no_feature_restates_a_leakage_column`,
  `test_unseen_weather_label_raises`.
- `tests/test_contract.py`: `test_every_raw_column_is_classified`,
  `test_raw_data_is_unchanged`.
- `.claude/`: `hooks/protect_raw.py`, its PreToolUse entry in
  `settings.json`, and `skills/ablation/SKILL.md`.
- Documents: `.gitattributes`; `docs/SEEDED_FAULTS.md`; `docs/AUDIT.md`;
  the dated lines in `docs/DECISIONS.md`; the `CLAUDE.md` edits; the README
  test count; this report's prose and tables.
- `reports/*.json` and `reports/audit.jsonl` were written by code.

Everything in the first commit (`f3a53b2`) is the course starter.

**Decisions that were mine** (each traces to a line in `docs/DECISIONS.md`):

1. **What to predict:** daily total `cnt` from calendar, same-day weather
   and lagged demand, the starter's framing, kept but reported as a
   nowcast (the same-day weather is stated in the Headline finding).
2. **What counts as leakage:** `casual`, `registered`, and any feature
   whose values restate them or `cnt`, judged on values, not names. Also,
   a raw column nobody has classified fails the build (the column-role and
   value-based lines).
3. **How to split:** 2011 / H1 2012 / H2 2012 by date. Test was read once
   for the main table and once for the chosen Exercise 4 transform.
4. **Which baselines:** mean and persistence beside every score, plus the
   anchor alone when judging a transform (the Exercise 4 lines).
5. **Whether to keep my feature:** kept, at +0.060 ridge validation R^2
   (the `cnt_max_7` line).
6. **Which transform, and whether it is defensible:** ratio to
   `cnt_roll_7` by a rule fixed beforehand. Defensible as a forecaster, not
   as evidence that trees compete (the two Exercise 4 lines).
7. **Whether the answer is believable:** ridge's R^2 edge over persistence
   is real but sits in the storm days, depends on knowing the day's
   weather, and does not show up on a typical day. So: not deployed as it
   stands (the Exercise 5 line).
8. **Guard rails:** raw data immutable by any route; a test added wherever
   a seeded fault stayed green; the `CLAUDE.md` changes.
