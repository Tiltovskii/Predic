# CatBoost baseline

This baseline predicts a pre-match series winner and, separately, the series
score and aggregate round share. It deliberately ignores round-event payloads
and individual combat metrics. Final map/round totals are only targets or
updates to historical form after a match has been featurized.

## Data and point-in-time rules

- 73,282 played series from 2020-06-15 through 2026-08-26.
- 49,459 train rows before 2025, 14,242 validation rows from 2025, and 9,581
  temporal test rows from 2026 by match start. Requiring the label itself to be
  known before each static cutoff leaves 49,397 train and 14,231 validation rows.
- 3,696 administrative `defwin` rows are excluded from sports modeling.
- A result updates team/player state at `end_at`, never at `start_at`.
  Simultaneous and overlapping matches therefore cannot see unfinished results.
- Five rows with no end time are skipped. A positive `end_at` is retained even
  for an unusually long series; a non-positive duration never updates causal
  state because its result cannot be placed safely on the timeline.
- Unknown map and round totals do not update their counters. Every smoothed rate
  carries a causal support/coverage counter.
- External ranking joins use the latest snapshot from an earlier calendar day.
  Same-day standings are excluded because the publication time is unknown.
- Train and validation rows are mirrored to remove source-side ordering bias.
- S/A tournaments receive full or near-full training weight; B/C/D receive
  0.55/0.35/0.20 to reduce the influence of noisy low-tier matches.

The external rankings are exported from Valve's official
`counter-strike_regional_standings` git history. The export contains 28,198
rows across 67 regional and 45 global snapshots from 2022-12-21 through
2026-08-03. Valve did not publish this ranking in 2018. Before its first
snapshot, the baseline relies on causal Elo and rolling form derived from the
match stream. Direct HLTV automation remained blocked by Cloudflare, so no
invented or current-rank backfill is used.

This baseline has an explicit **confirmed-lineup offline** prediction contract.
Historical lineup loading intentionally does not filter on `current_is_coach`:
that field describes the player now, not their role in an old match. This raises
two-full-lineup coverage from 45.7% to 66.4%. The current match's five principal
participants are treated as a proxy for the announced pre-match lineup, per the
chosen prediction point. This is deployable only when that five-player lineup
is actually available before inference. A substitution made only after the
series starts can still differ from the offline row; a future collector should
persist the announced lineup and its timestamp explicitly.

## Features

The materialized table contains 1,177 columns, including excluded `known_at`
metadata. `--feature-set base` selects 156 original/fixed features; `core`
selects 762; `all` selects 1,164 model inputs.

- causal team Elo, career record, and 1/3/7/14/30/90/180-day windows;
- last-3/5/10/20 form, signed streak, workload, rest, volatility and trends;
- exponentially weighted form with 7/30/90-day half-lives;
- opponent-strength mean/variance and observed-minus-Elo residuals;
- favorite conversion, underdog upset, sweep and decider counters;
- player experience/form distribution, including weakest-player statistics;
- exact-lineup, organization-pair, cross-organization pair and membership
  experience, roster overlap and churn;
- tier, venue, BO format, game version, bracket and current-tournament form;
- pre-veto map-pool depth, breadth, entropy, strengths and matchup compatibility;
- symmetric 90/365-day head-to-head win/map/round-share history;
- point-in-time Valve regional/global rank and points with age, missingness,
  roster-overlap and match-confidence fields.

No current BO3 team rank, current player performance from the predicted match,
actual map set/veto, odds, or BO3 AI prediction is used.

## Static temporal result

Winner model on 9,581 matches in 2026. Every row uses the same temporal split and
CatBoost configuration; only its feature set changes.

| Feature set | Inputs | Accuracy | ROC AUC | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed base | 156 | 66.74% | 0.721 | 0.6133 | 0.2123 | 0.0319 |
| Core counters | 762 | **66.96%** | 0.725 | 0.6091 | 0.2105 | 0.0276 |
| Every counter | 1,164 | 66.80% | **0.726** | **0.6090** | **0.2104** | **0.0268** |

The broad set is marginally better on probabilistic metrics, while the curated
core has higher raw accuracy with 402 fewer inputs. This is why materializing
many counters is useful but blindly training on every column is not automatically
best.

Series-score model on 7,367 completed BO3 test matches with core counters:

| Model | Exact accuracy | Multiclass log loss |
| --- | ---: | ---: |
| Train-prior majority (`2-0`) | 34.55% | 1.3586 |
| CatBoost (`2-0`, `2-1`, `1-2`, `0-2`) | **43.71%** | **1.2466** |

Summing its score probabilities gives 66.96% winner accuracy, ROC AUC 0.729 and
log loss 0.6019 on that BO3 slice.

The older project used a soft score-ratio regression target rather than exact
score classes. The new equivalent is the symmetric aggregate round share
`team1_rounds / (team1_rounds + team2_rounds)`. On 8,948 test matches with
known round totals, core CatBoost reaches MAE 0.1000 and RMSE 0.1284 versus
0.1192 and 0.1499 for constant 0.5. Thresholding at 0.5 identifies the series
winner 67.08% of the time. This is a dominance target, not a calibrated
win probability: in 4.54% of series the winner took fewer aggregate rounds.

## Monthly walk-forward result

The frozen model above intentionally never updates its fitted trees after the
end of 2024, even though its causal Elo/form features continue to update. A
second backtest simulates the production policy more closely: at the start of
each month in 2026 it selects tree count on the preceding 90 days, refits from
scratch on every match whose `known_at` is strictly before that month, and
predicts the complete next month. No fold trains on its own, an overlapping
unfinished match, or any later result.

| Monthly protocol | Inputs | Accuracy | ROC AUC | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed base | 156 | 66.81% | 0.731 | 0.6001 | 0.2074 | 0.0132 |
| Core counters | 762 | **67.40%** | **0.732** | **0.5986** | **0.2068** | **0.0071** |

Core counters improve accuracy in seven of eight 2026 monthly folds. January is
0.30 percentage point worse; the largest gains are March (+1.00), July (+0.91)
and August (+0.83). July remains the hardest month at 63.54%. On 7,310 rows with
two full five-player lineups, core accuracy is 68.00% with log loss 0.5914. With
a missing/partial side it is 65.48% and 0.6219. LAN is much easier in this sample
(72.24%) than online play (65.66%).

The leading additions are EWMA opponent strength, minimum player experience in
the lineup, current-tier/venue form, map-pool depth, pair experience and
within-tournament form. Valve rank contributes less once these causal counters
are present, partly because it only exists from late 2022.

These figures are not profitability. Odds, market timestamp and closing-line
value are required before any betting conclusion. Since 2026 has now informed
feature development, new future matches—not this reused slice—must become the
next real lockbox.

## Reproduce

```bash
python3 -m venv v2/.venv
v2/.venv/bin/python -m pip install --no-user -e 'v2[baseline]'

git clone --filter=blob:none --no-checkout \
  https://github.com/ValveSoftware/counter-strike_regional_standings.git \
  v2/data/valve-rankings-source

v2/.venv/bin/predic-data collect-valve-rankings \
  --repo v2/data/valve-rankings-source \
  --output-csv v2/data/valve-rankings.csv

v2/.venv/bin/predic-data extract-bo3-baseline-matches \
  --state-db v2/data/bo3-history-v2-state.sqlite3 \
  --output-csv v2/data/baseline/matches.csv

v2/.venv/bin/predic-data build-baseline-features \
  --matches-csv v2/data/baseline/matches.csv \
  --rankings-csv v2/data/valve-rankings.csv \
  --output-csv v2/data/baseline/features.csv

v2/.venv/bin/predic-data train-catboost-baseline \
  --features-csv v2/data/baseline/features.csv \
  --output-dir v2/data/baseline/run-2026-08-27 \
  --feature-set core

v2/.venv/bin/predic-data backtest-catboost-walk-forward \
  --features-csv v2/data/baseline/features.csv \
  --output-dir v2/data/baseline/run-2026-08-27 \
  --test-from 2026-01-01 \
  --validation-days 90 \
  --feature-set core
```

Generated datasets, models, predictions and metrics live below `v2/data/` and
are intentionally ignored by git. Each stage is deterministic and can be rerun
from the durable raw archive.
