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

The materialized table contains 1,219 columns, including excluded `known_at`
metadata and the optional after-veto layer. `--feature-set base` selects 156
original/fixed features; `core` selects 762; `all` selects 1,164 model inputs.
Those three manifests deliberately remain pre-veto. `core-veto` selects the
same 762 core inputs plus 41 veto/map-matchup inputs, for 803 total.

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
odds, or BO3 AI prediction is used. The ordinary `base`, `core`, and `all`
manifests also exclude the current match's map set and veto. Only the explicit
`core-veto` prediction point includes them.

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

## Retrospective assumed after-veto result

The separate `core-veto` contract answers a later question: what changes if a
complete BO1/BO3 veto is already visible? It uses only the ordered picks, bans,
and decider plus causal 180-day map history known before the series result. It
never derives the selected maps from played games, final score, or map results,
and it ignores the source's mutable current-map-pool flag.

Strict veto validation yields 39,090 BO1/BO3 series: 24,404 label-eligible rows
before 2025, 8,334 in 2025, and 6,338 in the 2026 test. Veto availability is
strongly correlated with event tier and year, so both models below are trained
and evaluated on this exact same cohort. Comparing `core-veto` with the general
67.40% core result would otherwise mix feature value with selection bias.

| Same-cohort monthly protocol | Inputs | Accuracy | ROC AUC | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Core, no current veto | 762 | **67.81%** | 0.7385 | 0.5925 | 0.2041 | 0.0127 |
| Core + current veto | 803 | 67.61% | **0.7393** | **0.5912** | **0.2037** | **0.0104** |

The MVP veto layer does **not** improve threshold accuracy: the paired change
is -0.21 percentage point. It makes the probabilities slightly better on AUC,
log loss, Brier, and calibration. A tournament-cluster bootstrap over 316
events gives an accuracy-delta 95% interval of -0.66 to +0.23 percentage point
and a log-loss-delta interval of -0.00279 to +0.00030, both including zero.
There is no reliable headline lift yet.

The effect is heterogeneous and exploratory. On 5,813 BO3s, accuracy changes
from 67.73% to 67.49% while log loss improves from 0.5951 to 0.5939. LAN moves
from 70.98% to 71.26% with a 0.0042 log-loss improvement; online play moves
from 66.20% to 65.75% and is slightly worse on log loss. These slices should
not be treated as independently validated claims.

This is an **assumed after-veto**, not a historically proven live point-in-time
backtest. The archive contains one post-hoc match-detail snapshot per task,
captured on 2026-08-26, and `match_maps` has no publication timestamp. The veto
content is structurally consistent with a pre-map veto, but the local archive
cannot prove that each value was visible five minutes before its match. A live
collector must persist the first complete snapshot as `veto_known_at`; only
prospectively timestamped rows can validate the deployable five-minute model.
The richer team-by-map rating and series simulation should then be developed on
a new validation period and judged on an untouched future lockbox.

## Individual-map winner baseline

The map baseline answers the narrower after-veto question directly: which side
wins this named map? One row is one played map whose identity can be matched
unambiguously to a strict BO1/BO3 veto. Every map in a series receives the same
pre-series team/player state. Map results update target-map Elo and rolling
form only at the series `known_at`, so Map 1 cannot leak into Map 2 or the
decider. Played-map scores and winners are retained only as targets/QC fields.

The dataset contains 83,464 map targets from 38,429 series: 51,126
label-eligible rows before 2025, 17,959 in 2025, and 14,363 in the 2026 test.
The model uses 932 inputs: the 803 `core-veto` series features, target map,
pick/decider role, and causal target-map Elo/form counters.

| 2026 monthly map protocol | Accuracy | ROC AUC | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-map Elo only | 58.59% | 0.6160 | 0.6701 | 0.2386 | 0.0302 |
| Series team Elo only | 61.09% | 0.6508 | 0.6575 | 0.2323 | 0.0372 |
| Map CatBoost | **63.80%** | **0.6906** | **0.6302** | **0.2204** | **0.0160** |

Against series Elo, CatBoost adds 2.71 percentage points of accuracy and cuts
log loss by 0.0273. A paired bootstrap clustered by 6,281 matches gives a 95%
interval of +2.03 to +3.38 percentage points for the accuracy delta and -0.0308
to -0.0235 for the log-loss delta. The model's own accuracy interval is 62.97%
to 64.61% on this reused development test.

| Slice | Rows | Accuracy | ROC AUC | Log loss |
| --- | ---: | ---: | ---: | ---: |
| BO1 | 525 | 69.33% | 0.7456 | 0.5721 |
| BO3, all played maps | 13,838 | 63.59% | 0.6877 | 0.6324 |
| BO3 Map 1 | 5,756 | 63.81% | 0.6927 | 0.6304 |
| BO3 Map 2 | 5,756 | **65.43%** | **0.7087** | **0.6186** |
| BO3 decider | 2,326 | 58.47% | 0.6162 | 0.6718 |
| LAN | 4,467 | 66.69% | 0.7234 | 0.6069 |
| Online | 9,896 | 62.49% | 0.6746 | 0.6408 |
| Tier S/A | 1,690 | 62.19% | 0.6559 | 0.6534 |

The two picked maps are meaningfully predictable at 64.62% combined, while
simply backing the map's pick owner wins only 54.21%. Pre-series deciders are
hard because reaching Map 3 selects close series and the prediction does not
yet observe Maps 1–2. A separate live decider refresh could legitimately use
those completed maps at that later prediction point.

At model confidence at least 0.60, accuracy is 71.31% on 52.81% coverage; at
0.65 it is 75.99% on 34.19%; at 0.70 it is 81.63% on 19.14%. These are
predictive confidence slices, not betting returns: odds, their timestamps,
selection rules, and a prospective lockbox are still absent. The same
historical `veto_known_at` limitation from the preceding section applies.

## Light target-aware player-history Transformer

The first sequence model is deliberately small. It is an early-binding,
target-aware encoder inspired by Argus rather than a copy of its production
implementation. One target is a named map after veto. For each of the ten
known participants, the candidate map token is appended to at most 32 past
player-map events and participates inside a shared Transformer encoder. The
candidate contains map, organization, veto role, tournament context, and
causal pre-series counters, but never its winner, score, or round share.

Past events contain the player's per-round kills, deaths, assists, damage,
ADR, KAST, rating, openings, trades, accuracy/economy proxies, past outcome,
organization, opponent, map, tier, venue, version, and age. Events are ordered
by `(known_at, match_id, game_id)` and admitted only when `known_at <=` the
target series start. All maps in one series share exactly the same pre-series
histories. The two side scores share weights and are subtracted, so swapping
the complete sides negates the logit exactly.

The compact dataset stores 1,007,022 quality-gated player-map events once and
59,641 map targets as history indices. It occupies 241 MB. There are 32,566
tuning-train targets before 2025, 14,425 validation targets in 2025, and 12,650
test maps in 2026. Another 23,811 map rows are excluded because at least one
side has no complete five-player roster in the captured player archive; they
are not filled or assigned invented participants. On the 2026 cohort, a player
has 30.45 of 32 history slots on average and 91.78% of player sequences are
full.

The main model has 3 Transformer layers, width 128, 4 heads, 2.53M parameters,
BF16 on one A100, AdamW with peak learning rate `2e-4`, gradient clipping, and
one weighted BCE head. Epoch count is selected on 2025; the selected two epochs
are then refit from scratch at each 2026 month boundary using only labels known
before that month.

| Exact player-complete 2026 cohort | Accuracy | ROC AUC | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Series team Elo | 61.31% | 0.6520 | 0.6564 | 0.2318 | 0.0355 |
| Light Argus, one refit before 2026 | 62.49% | 0.6793 | 0.6348 | 0.2228 | 0.0197 |
| Light Argus, monthly full refit | 62.93% | 0.6756 | 0.6368 | 0.2238 | 0.0218 |
| Light Argus, monthly, no player ID | 62.95% | 0.6759 | 0.6367 | 0.2237 | 0.0217 |
| Map CatBoost, same target cohort | **64.03%** | **0.6914** | **0.6278** | **0.2196** | **0.0113** |

The sequence model is a useful first baseline, but it does **not** beat
CatBoost. Clustered over 5,553 matches, monthly Light Argus trails same-cohort
CatBoost by 1.10 percentage points of accuracy (95% interval -1.73 to -0.45)
and has 0.00893 higher log loss (95% interval +0.00590 to +0.01198). Its own
accuracy interval is 62.02% to 63.80%. This is not a seed-level significance
claim—the neural model still needs repeated-seed evaluation—but the current
single run is clearly not a replacement for the tree model.

Removing the learned player-ID embedding changes only two net correct calls:
accuracy rises by 0.016 percentage point and log loss falls by 0.00012. A
same-seed match-cluster bootstrap gives intervals of -0.33 to +0.35 percentage
point and -0.00098 to +0.00070 respectively. This is indistinguishable from
noise, so the ID-enabled architecture remains the named baseline rather than
selecting an ablation on the reused test. It does suggest that future gains
must come from learning transferable form and role representations, not from
memorizing player identity.

Light Argus reaches 71.86% on 494 BO1 maps, 65.19% on BO3 Map 2, and only
56.77% on 2,038 BO3 deciders. It is stronger on LAN (65.44%) than online
(61.66%). CatBoost and Argus probabilities are highly correlated, but each has
exclusive correct calls (808 Argus-only versus 947 CatBoost-only). A blend
weight searched on this reused test would be leakage; stacking must be trained
from out-of-fold Transformer embeddings on a separate validation period.

The next justified experiment is therefore not simply a larger Transformer.
Pretrain the player encoder on the complete million-event stream, export
strictly out-of-fold player/team embeddings, and let the proven CatBoost head
consume them alongside its counters. Partial-roster masking can then recover
training coverage without inventing players. Odds and a prospective veto
lockbox remain separate requirements before any betting conclusion.

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

# Fair same-cohort comparison at the assumed after-veto prediction point.
v2/.venv/bin/predic-data backtest-catboost-walk-forward \
  --features-csv v2/data/baseline/features.csv \
  --output-dir v2/data/baseline/walk-veto-core \
  --test-from 2026-01-01 \
  --validation-days 90 \
  --feature-set core \
  --veto-known-only

v2/.venv/bin/predic-data backtest-catboost-walk-forward \
  --features-csv v2/data/baseline/features.csv \
  --output-dir v2/data/baseline/walk-veto-core-veto \
  --test-from 2026-01-01 \
  --validation-days 90 \
  --feature-set core-veto

v2/.venv/bin/predic-data build-map-baseline-features \
  --matches-csv v2/data/baseline/matches.csv \
  --series-features-csv v2/data/baseline/features.csv \
  --output-csv v2/data/baseline/maps-core-veto.csv \
  --series-feature-set core-veto

v2/.venv/bin/predic-data backtest-map-catboost-walk-forward \
  --features-csv v2/data/baseline/maps-core-veto.csv \
  --output-dir v2/data/baseline/walk-map-core-veto \
  --test-from 2026-01-01 \
  --validation-days 90

v2/.venv/bin/python -m pip install --no-user -e 'v2[argus]'

v2/.venv/bin/predic-data build-light-argus-dataset \
  --state-db v2/data/bo3-history-v2-state.sqlite3 \
  --matches-csv v2/data/baseline/matches.csv \
  --map-features-csv v2/data/baseline/maps-core-veto.csv \
  --output-dir v2/data/light-argus-v1 \
  --max-history 32

v2/.venv/bin/predic-data train-light-argus \
  --dataset-dir v2/data/light-argus-v1 \
  --output-dir v2/data/light-argus-v1/output-monthly \
  --train-before 2025-01-01 \
  --test-from 2026-01-01 \
  --epochs 12 \
  --patience 3 \
  --batch-size 256 \
  --d-model 128 \
  --layers 3 \
  --heads 4 \
  --learning-rate 0.0002 \
  --monthly-refit \
  --catboost-predictions-csv \
    v2/data/baseline/walk-map-core-veto/map_walk_forward_test_predictions.csv

# Retrain the CatBoost comparator on exactly the Argus-eligible target cohort.
v2/.venv/bin/predic-data backtest-map-catboost-walk-forward \
  --features-csv v2/data/baseline/maps-core-veto.csv \
  --cohort-metadata-jsonl v2/data/light-argus-v1/target_metadata.jsonl \
  --output-dir v2/data/light-argus-v1/catboost-player-complete \
  --test-from 2026-01-01 \
  --validation-days 90
```

Generated datasets, models, predictions and metrics live below `v2/data/` and
are intentionally ignored by git. Each stage is deterministic and can be rerun
from the durable raw archive.
