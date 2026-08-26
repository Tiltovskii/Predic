# CatBoost baseline

This baseline predicts a pre-match series winner and, separately, the exact
score of completed BO3 series. It deliberately ignores round-event payloads and
individual combat metrics. Final map scores are used only to update historical
form after a match has been featurized.

## Data and point-in-time rules

- 76,983 valid BO3-source series from 2020-06-15 through 2026-08-26.
- 51,157 train rows before 2025, 15,142 validation rows from 2025, and 10,684
  untouched test rows from 2026.
- Every counter is frozen before its match result updates team/player state.
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

## Features

- causal team Elo, career record, 30/90/180-day win and map-win rates;
- opponent-strength, round-share, activity and experience counters;
- player Elo/experience aggregates for the expected five-player lineup;
- lineup and player-pair continuity;
- head-to-head history;
- tournament tier, format, game version and prize context;
- point-in-time Valve regional/global rank and points with age, missingness,
  roster-overlap and match-confidence fields.

No current team rank, post-match player statistic, future roster, odds, or AI
prediction is used.

## First temporal holdout result

Winner model on all 10,684 matches in 2026:

| Model | Accuracy | ROC AUC | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| Constant 0.5 | 54.96% | 0.500 | 0.6931 | 0.2500 |
| Dynamic Elo only | 62.84% | 0.676 | 0.6395 | 0.2243 |
| CatBoost | **65.24%** | **0.706** | **0.6229** | **0.2168** |

On the 787 S/A-tier test matches, CatBoost accuracy is 66.07% with ROC AUC
0.703. This slice is still small and should not be treated as a betting result.
At model confidence >=0.65, the all-tier slice contains 4,084 matches and is
76.74% correct; the S/A-only slice contains 334 matches and is 73.95% correct.
These are classification figures, not profitability: bookmaker odds and
closing-line value are still required before any betting conclusion.

Exact-score model on 7,369 completed BO3 test matches:

| Model | Exact accuracy | Multiclass log loss |
| --- | ---: | ---: |
| Train-prior majority (`2-0`) | 34.55% | 1.3586 |
| CatBoost (`2-0`, `2-1`, `1-2`, `0-2`) | **43.83%** | **1.2513** |

Summing the exact-score probabilities into winner probability gives 66.92%
accuracy, ROC AUC 0.727 and log loss 0.6049 on that BO3 slice. The binary model
on the same slice gives 66.82%, 0.723 and 0.6132 respectively. Thus the richer
score target is modestly useful here, although it remains a separate model and
does not add an auxiliary loss to the winner model.

Top winner-model features in the first run are Elo difference, recent opponent
strength, player experience/Elo, recent round share and team identity. External
Valve rank contributes little after the causal counters, partly because it only
exists from late 2022 and is sparse before the global series starts in 2024.

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
  --output-dir v2/data/baseline/run-2026-08-27
```

Generated datasets, models, predictions and metrics live below `v2/data/` and
are intentionally ignored by git. Each stage is deterministic and can be rerun
from the durable raw archive.
