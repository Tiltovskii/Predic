from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .baseline import _binary_metrics
from .light_argus import (
    LightTargetAwareArgus,
    _ArrayStore,
    _autocast,
    _BatchCollator,
    _batches,
    _device,
    _finite_stats,
    _move,
    _Normalisation,
    _normalise,
    _scheduler,
)
from .light_argus_data import EVENT_NUMERIC_FIELDS, FORMAT_VERSION

_WIN_FIELD = EVENT_NUMERIC_FIELDS.index("won_map")
_REGRESSION_FIELDS = tuple(
    index for index in range(len(EVENT_NUMERIC_FIELDS)) if index != _WIN_FIELD
)
_TIER_WEIGHTS = {"s": 1.0, "a": 0.9, "b": 0.55, "c": 0.35, "d": 0.2}


def _cutoff(year: int) -> int:
    return int(datetime(year, 1, 1, tzinfo=UTC).timestamp())


def _limit(indices: np.ndarray, maximum: int | None) -> np.ndarray:
    if maximum is None or len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
    return indices[positions]


def _pretrain_normalisation(
    data: _ArrayStore,
    train_indices: np.ndarray,
    *,
    fixed_event_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> _Normalisation:
    if fixed_event_stats is None:
        event_mean, event_std = _finite_stats(
            np.asarray(data.event_numeric[train_indices], dtype=np.float32)
        )
    else:
        event_mean, event_std = fixed_event_stats
    seen_players = np.zeros(len(data.vocabularies["player"]), dtype=bool)
    seen_players[np.asarray(data.event_player[train_indices], dtype=np.int64)] = True
    seen_players[0] = True
    seen_teams = np.zeros(len(data.vocabularies["team"]), dtype=bool)
    seen_teams[np.asarray(data.event_team[train_indices], dtype=np.int64)] = True
    seen_teams[np.asarray(data.event_opponent[train_indices], dtype=np.int64)] = True
    seen_teams[0] = True
    side_size = data.target_side_numeric.shape[-1]
    shared_size = data.target_shared_numeric.shape[-1]
    return _Normalisation(
        event_mean=event_mean,
        event_std=event_std,
        side_mean=np.zeros(side_size, dtype=np.float32),
        side_std=np.ones(side_size, dtype=np.float32),
        shared_mean=np.zeros(shared_size, dtype=np.float32),
        shared_std=np.ones(shared_size, dtype=np.float32),
        seen_players=seen_players,
        seen_teams=seen_teams,
    )


class _EventPretrainCollator:
    def __init__(self, data: _ArrayStore, normalisation: _Normalisation):
        self.data = data
        self.normalisation = normalisation
        tier_weights = np.full(len(data.vocabularies["tier"]), 0.5, dtype=np.float32)
        for name, index in data.vocabularies["tier"].items():
            tier_weights[index] = _TIER_WEIGHTS.get(name, 0.5)
        self.tier_weights = tier_weights

    def __call__(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        indices = np.asarray(indices, dtype=np.int64)
        history = np.asarray(self.data.event_history_indices[indices], dtype=np.int64)[
            :, None, :
        ]
        history_mask = history >= 0
        safe_history = np.maximum(history, 0)
        numeric = _normalise(
            np.asarray(self.data.event_numeric[safe_history], dtype=np.float32),
            self.normalisation.event_mean,
            self.normalisation.event_std,
        )
        starts = np.asarray(self.data.event_start_ts[indices], dtype=np.int64)
        history_ts = np.asarray(self.data.event_known_ts[safe_history], dtype=np.int64)
        age_days = np.maximum(
            0.0,
            (starts[:, None, None] - history_ts).astype(np.float32) / 86_400.0,
        )
        age = np.clip(np.log1p(age_days) / math.log1p(3650.0), 0.0, 1.5)
        age[~history_mask] = 0.0
        recency = np.linspace(0.0, 1.0, history.shape[-1], dtype=np.float32)[
            None, None, :
        ]
        recency = np.broadcast_to(recency, history.shape).copy()
        recency[~history_mask] = 0.0
        numeric = np.concatenate((numeric, age[..., None], recency[..., None]), axis=-1)

        current_player = np.asarray(self.data.event_player[indices], dtype=np.int64)[
            :, None
        ]
        current_team = np.asarray(self.data.event_team[indices], dtype=np.int64)[
            :, None
        ]
        current_opponent = np.asarray(
            self.data.event_opponent[indices], dtype=np.int64
        )[:, None]
        current_player = np.where(
            self.normalisation.seen_players[current_player], current_player, 0
        )
        current_team = np.where(
            self.normalisation.seen_teams[current_team], current_team, 0
        )
        current_opponent = np.where(
            self.normalisation.seen_teams[current_opponent], current_opponent, 0
        )
        event_team = np.asarray(self.data.event_team[safe_history], dtype=np.int64)
        event_opponent = np.asarray(
            self.data.event_opponent[safe_history], dtype=np.int64
        )
        event_team = np.where(self.normalisation.seen_teams[event_team], event_team, 0)
        event_opponent = np.where(
            self.normalisation.seen_teams[event_opponent], event_opponent, 0
        )

        target_raw = np.asarray(self.data.event_numeric[indices], dtype=np.float32)
        target_mask = np.isfinite(target_raw)
        target_numeric = np.where(
            target_mask,
            (target_raw - self.normalisation.event_mean) / self.normalisation.event_std,
            0.0,
        ).astype(np.float32)
        tier = np.asarray(self.data.event_tier[indices], dtype=np.int64)
        result = {
            "history_mask": history_mask,
            "event_numeric": numeric,
            "event_team": event_team,
            "event_opponent": event_opponent,
            "event_map": np.asarray(self.data.event_map[safe_history], dtype=np.int64),
            "event_tier": np.asarray(
                self.data.event_tier[safe_history], dtype=np.int64
            ),
            "event_event_type": np.asarray(
                self.data.event_event_type[safe_history], dtype=np.int64
            ),
            "event_version": np.asarray(
                self.data.event_version[safe_history], dtype=np.int64
            ),
            "candidate_player": current_player,
            "candidate_team": current_team,
            "candidate_opponent": current_opponent,
            "candidate_map": np.asarray(self.data.event_map[indices], dtype=np.int64)[
                :, None
            ],
            "candidate_tier": tier[:, None],
            "candidate_event_type": np.asarray(
                self.data.event_event_type[indices], dtype=np.int64
            )[:, None],
            "candidate_version": np.asarray(
                self.data.event_version[indices], dtype=np.int64
            )[:, None],
            "target_numeric": target_numeric,
            "target_mask": target_mask,
            "target_win": np.where(
                target_mask[:, _WIN_FIELD], target_raw[:, _WIN_FIELD], 0.0
            ).astype(np.float32),
            "target_win_mask": target_mask[:, _WIN_FIELD],
            "sample_weight": self.tier_weights[tier],
        }
        return {
            key: torch.from_numpy(np.asarray(value)) for key, value in result.items()
        }


class PlayerEventAuxiliaryModel(nn.Module):
    """Pretrain the shared player encoder on the next observed map performance."""

    def __init__(self, backbone: LightTargetAwareArgus):
        super().__init__()
        self.backbone = backbone
        dimension = backbone.config.d_model
        self.regression_head = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, 2 * dimension),
            nn.GELU(),
            nn.Linear(2 * dimension, len(_REGRESSION_FIELDS)),
        )
        self.win_head = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, 1),
        )

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.backbone.encode_candidate_players(batch).squeeze(1)
        return self.regression_head(state), self.win_head(state).squeeze(-1)


def _auxiliary_loss(
    regression: torch.Tensor,
    win_logit: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fields = torch.as_tensor(_REGRESSION_FIELDS, device=regression.device)
    target = batch["target_numeric"].index_select(1, fields)
    mask = batch["target_mask"].index_select(1, fields)
    row_mse = ((regression - target).square() * mask).sum(dim=1)
    row_mse = row_mse / mask.sum(dim=1).clamp_min(1)
    win_target = batch["target_win"]
    win_mask = batch["target_win_mask"]
    win_loss = nn.functional.binary_cross_entropy_with_logits(
        win_logit, win_target, reduction="none"
    )
    weight = batch["sample_weight"]
    regression_loss = (row_mse * weight).sum() / weight.sum().clamp_min(1e-6)
    win_weight = weight * win_mask
    classification_loss = (win_loss * win_weight).sum() / win_weight.sum().clamp_min(
        1e-6
    )
    return (
        regression_loss + 0.5 * classification_loss,
        regression_loss,
        classification_loss,
    )


def _train_auxiliary(
    model: PlayerEventAuxiliaryModel,
    collator: _EventPretrainCollator,
    indices: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> list[dict[str, float]]:
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    steps = max(1, math.ceil(len(indices) / batch_size))
    scheduler = _scheduler(optimizer, steps * epochs)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        totals = np.zeros(3, dtype=np.float64)
        rows = 0
        for raw_indices in _batches(indices, batch_size, shuffle=True, rng=rng):
            batch = _move(collator(raw_indices), device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                regression, win_logit = model(batch)
                loss, regression_loss, win_loss = _auxiliary_loss(
                    regression, win_logit, batch
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            count = len(raw_indices)
            totals += (
                np.array(
                    [
                        float(loss.detach()),
                        float(regression_loss.detach()),
                        float(win_loss.detach()),
                    ]
                )
                * count
            )
            rows += count
        values = totals / max(1, rows)
        result = {
            "epoch": float(epoch),
            "loss": float(values[0]),
            "regression_loss": float(values[1]),
            "win_logloss": float(values[2]),
        }
        history.append(result)
        print(
            "argus-pretrain "
            + " ".join(f"{key}={value:.5f}" for key, value in result.items()),
            file=sys.stderr,
            flush=True,
        )
    return history


@torch.inference_mode()
def _evaluate_auxiliary(
    model: PlayerEventAuxiliaryModel,
    collator: _EventPretrainCollator,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    regressions: list[np.ndarray] = []
    win_probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    win_targets: list[np.ndarray] = []
    win_masks: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    for raw_indices in _batches(indices, batch_size, shuffle=False, rng=rng):
        batch = _move(collator(raw_indices), device)
        with _autocast(device):
            regression, win_logit = model(batch)
        regressions.append(regression.float().cpu().numpy())
        win_probabilities.append(torch.sigmoid(win_logit).float().cpu().numpy())
        targets.append(batch["target_numeric"].float().cpu().numpy())
        masks.append(batch["target_mask"].cpu().numpy())
        win_targets.append(batch["target_win"].float().cpu().numpy())
        win_masks.append(batch["target_win_mask"].cpu().numpy())
    prediction = np.concatenate(regressions)
    probability = np.concatenate(win_probabilities)
    target = np.concatenate(targets)
    mask = np.concatenate(masks)
    win_target = np.concatenate(win_targets)
    win_valid = np.concatenate(win_masks).astype(bool)
    regression_target = target[:, _REGRESSION_FIELDS]
    regression_mask = mask[:, _REGRESSION_FIELDS]
    squared = np.square(prediction - regression_target)
    baseline_squared = np.square(regression_target)
    field_metrics: dict[str, object] = {}
    for output_index, field_index in enumerate(_REGRESSION_FIELDS):
        valid = regression_mask[:, output_index]
        if not valid.any():
            continue
        field_metrics[EVENT_NUMERIC_FIELDS[field_index]] = {
            "rows": int(valid.sum()),
            "normalised_rmse": float(np.sqrt(squared[valid, output_index].mean())),
            "mean_baseline_normalised_rmse": float(
                np.sqrt(baseline_squared[valid, output_index].mean())
            ),
        }
    valid_cells = regression_mask
    return {
        "rows": len(indices),
        "normalised_rmse": float(np.sqrt(squared[valid_cells].mean())),
        "mean_baseline_normalised_rmse": float(
            np.sqrt(baseline_squared[valid_cells].mean())
        ),
        "fields": field_metrics,
        "map_win": _binary_metrics(win_target[win_valid], probability[win_valid]),
    }


@torch.inference_mode()
def _export_target_fold(
    model: PlayerEventAuxiliaryModel,
    data: _ArrayStore,
    normalisation: _Normalisation,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.to(device).eval()
    collator = _BatchCollator(data, normalisation)
    team1: list[np.ndarray] = []
    team2: list[np.ndarray] = []
    team1_auxiliary: list[np.ndarray] = []
    team2_auxiliary: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    for raw_indices in _batches(indices, batch_size, shuffle=False, rng=rng):
        batch = _move(collator(raw_indices), device)
        with _autocast(device):
            states = model.backbone.encode_candidate_players(batch)
            flattened = states.reshape(-1, states.shape[-1])
            regression = model.regression_head(flattened).reshape(
                len(raw_indices), 10, -1
            )
            win_probability = torch.sigmoid(model.win_head(flattened)).reshape(
                len(raw_indices), 10, 1
            )
            auxiliary = torch.cat((regression, win_probability), dim=-1)
        states_array = states.float().cpu().numpy()
        auxiliary_array = auxiliary.float().cpu().numpy()
        team1.append(states_array[:, :5].mean(axis=1))
        team2.append(states_array[:, 5:].mean(axis=1))
        team1_auxiliary.append(auxiliary_array[:, :5].mean(axis=1))
        team2_auxiliary.append(auxiliary_array[:, 5:].mean(axis=1))
    return (
        indices,
        np.concatenate(team1),
        np.concatenate(team2),
        np.concatenate(team1_auxiliary),
        np.concatenate(team2_auxiliary),
    )


def _atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _metadata_map_ids(path: Path, count: int) -> list[str]:
    result = [""] * count
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            result[int(row["row_index"])] = str(row["map_row_id"])
    if any(not value for value in result):
        raise ValueError("target metadata does not cover every row")
    return result


def pretrain_and_export_argus_embeddings(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    first_fold_year: int = 2021,
    last_fold_year: int = 2026,
    epochs_per_fold: int = 2,
    final_refit_epochs: int = 2,
    batch_size: int = 512,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    d_model: int = 64,
    layers: int = 2,
    heads: int = 4,
    dropout: float = 0.10,
    use_player_identity: bool = False,
    use_team_identity: bool = True,
    device: str = "auto",
    seed: int = 20260828,
    max_train_events: int | None = None,
    max_evaluation_events: int | None = None,
    max_target_rows: int | None = None,
) -> dict[str, object]:
    """Sequentially pretrain and export strictly out-of-time map embeddings."""
    import pandas as pd

    if first_fold_year > last_fold_year:
        raise ValueError("first_fold_year must not exceed last_fold_year")
    if min(epochs_per_fold, final_refit_epochs, batch_size) < 1:
        raise ValueError("epoch counts and batch_size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    data = _ArrayStore(dataset_dir)
    output = Path(output_dir).resolve()
    folds_dir = output / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    selected_device = _device(device)
    config = data.model_config(
        d_model=d_model,
        layers=layers,
        heads=heads,
        dropout=dropout,
        use_player_identity=use_player_identity,
        use_team_identity=use_team_identity,
    )
    run_config = {
        "format_version": FORMAT_VERSION,
        "events": len(data.event_known_ts),
        "targets": len(data.target_start_ts),
        "first_fold_year": first_fold_year,
        "last_fold_year": last_fold_year,
        "epochs_per_fold": epochs_per_fold,
        "final_refit_epochs": final_refit_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "model": asdict(config),
        "seed": seed,
        "max_train_events": max_train_events,
        "max_evaluation_events": max_evaluation_events,
        "max_target_rows": max_target_rows,
        "coordinate_policy": "sequential expanding warm-start",
    }
    config_path = output / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise ValueError("output directory contains a different pretrain run")
    else:
        config_path.write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    first_train = np.flatnonzero(
        np.asarray(data.event_known_ts) < _cutoff(first_fold_year)
    )
    if len(first_train) == 0:
        raise ValueError("the first fold has no preceding pretrain events")
    fixed_stats = _finite_stats(
        np.asarray(data.event_numeric[first_train], dtype=np.float32)
    )
    model = PlayerEventAuxiliaryModel(LightTargetAwareArgus(config))
    event_known = np.asarray(data.event_known_ts)
    event_history_count = (np.asarray(data.event_history_indices) >= 0).sum(axis=1)
    target_starts = np.asarray(data.target_start_ts)
    fold_reports: list[dict[str, object]] = []
    fold_files: list[Path] = []

    for fold_number, year in enumerate(
        range(first_fold_year, last_fold_year + 1), start=1
    ):
        fold_start, fold_end = _cutoff(year), _cutoff(year + 1)
        train_indices = np.flatnonzero(
            (event_known < fold_start) & (event_history_count > 0)
        )
        evaluation_indices = np.flatnonzero(
            (event_known >= fold_start)
            & (event_known < fold_end)
            & (event_history_count > 0)
        )
        target_indices = np.flatnonzero(
            (target_starts >= fold_start) & (target_starts < fold_end)
        )
        train_indices = _limit(train_indices, max_train_events)
        evaluation_indices = _limit(evaluation_indices, max_evaluation_events)
        target_indices = _limit(target_indices, max_target_rows)
        if min(len(train_indices), len(evaluation_indices), len(target_indices)) == 0:
            raise ValueError(
                f"fold {year} has an empty train, evaluation or target set"
            )
        checkpoint_path = folds_dir / f"fold-{year}.pt"
        embeddings_path = folds_dir / f"fold-{year}-embeddings.npz"
        metrics_path = folds_dir / f"fold-{year}-metrics.json"
        normalisation = _pretrain_normalisation(
            data, train_indices, fixed_event_stats=fixed_stats
        )
        if (
            checkpoint_path.exists()
            and embeddings_path.exists()
            and metrics_path.exists()
        ):
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["state_dict"])
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            with np.load(embeddings_path) as existing_embeddings:
                has_auxiliary = {
                    "team1_auxiliary",
                    "team2_auxiliary",
                }.issubset(existing_embeddings.files)
            if not has_auxiliary:
                row_index, team1, team2, team1_auxiliary, team2_auxiliary = (
                    _export_target_fold(
                        model,
                        data,
                        normalisation,
                        target_indices,
                        batch_size=batch_size,
                        device=selected_device,
                    )
                )
                _atomic_npz(
                    embeddings_path,
                    row_index=row_index.astype(np.int32),
                    team1=team1.astype(np.float32),
                    team2=team2.astype(np.float32),
                    team1_auxiliary=team1_auxiliary.astype(np.float32),
                    team2_auxiliary=team2_auxiliary.astype(np.float32),
                )
                print(
                    f"argus-pretrain enrich fold {year} auxiliary outputs",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(f"argus-pretrain resume fold {year}", file=sys.stderr, flush=True)
        else:
            train_collator = _EventPretrainCollator(data, normalisation)
            training_history = _train_auxiliary(
                model,
                train_collator,
                train_indices,
                epochs=epochs_per_fold,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=selected_device,
                seed=seed + fold_number,
            )
            evaluation = _evaluate_auxiliary(
                model,
                train_collator,
                evaluation_indices,
                batch_size=batch_size,
                device=selected_device,
            )
            row_index, team1, team2, team1_auxiliary, team2_auxiliary = (
                _export_target_fold(
                    model,
                    data,
                    normalisation,
                    target_indices,
                    batch_size=batch_size,
                    device=selected_device,
                )
            )
            report = {
                "year": year,
                "train_events": len(train_indices),
                "evaluation_events": len(evaluation_indices),
                "target_maps": len(target_indices),
                "history": training_history,
                "auxiliary_evaluation": evaluation,
            }
            _atomic_npz(
                embeddings_path,
                row_index=row_index.astype(np.int32),
                team1=team1.astype(np.float32),
                team2=team2.astype(np.float32),
                team1_auxiliary=team1_auxiliary.astype(np.float32),
                team2_auxiliary=team2_auxiliary.astype(np.float32),
            )
            _atomic_torch_save(
                {
                    "format_version": FORMAT_VERSION,
                    "year": year,
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                },
                checkpoint_path,
            )
            metrics_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        fold_reports.append(report)
        fold_files.append(embeddings_path)

    row_indices: list[np.ndarray] = []
    team1_embeddings: list[np.ndarray] = []
    team2_embeddings: list[np.ndarray] = []
    team1_auxiliary_outputs: list[np.ndarray] = []
    team2_auxiliary_outputs: list[np.ndarray] = []
    for path in fold_files:
        with np.load(path) as fold:
            row_indices.append(fold["row_index"])
            team1_embeddings.append(fold["team1"])
            team2_embeddings.append(fold["team2"])
            team1_auxiliary_outputs.append(fold["team1_auxiliary"])
            team2_auxiliary_outputs.append(fold["team2_auxiliary"])
    rows = np.concatenate(row_indices)
    team1 = np.concatenate(team1_embeddings)
    team2 = np.concatenate(team2_embeddings)
    team1_auxiliary = np.concatenate(team1_auxiliary_outputs)
    team2_auxiliary = np.concatenate(team2_auxiliary_outputs)
    order = np.argsort(rows)
    rows, team1, team2 = rows[order], team1[order], team2[order]
    team1_auxiliary = team1_auxiliary[order]
    team2_auxiliary = team2_auxiliary[order]
    map_ids = _metadata_map_ids(
        data.directory / "target_metadata.jsonl", len(data.target_start_ts)
    )
    frame: dict[str, object] = {
        "map_row_id": [map_ids[int(index)] for index in rows],
        "argus_oof_available": np.ones(len(rows), dtype=np.uint8),
    }
    for dimension in range(d_model):
        frame[f"team1_argus_oof_{dimension:03d}"] = team1[:, dimension]
        frame[f"team2_argus_oof_{dimension:03d}"] = team2[:, dimension]
        frame[f"diff_argus_oof_{dimension:03d}"] = (
            team1[:, dimension] - team2[:, dimension]
        )
    auxiliary_fields = [
        *(EVENT_NUMERIC_FIELDS[index] for index in _REGRESSION_FIELDS),
        "map_win_probability",
    ]
    for dimension, field in enumerate(auxiliary_fields):
        frame[f"team1_argus_aux_{field}"] = team1_auxiliary[:, dimension]
        frame[f"team2_argus_aux_{field}"] = team2_auxiliary[:, dimension]
        frame[f"diff_argus_aux_{field}"] = (
            team1_auxiliary[:, dimension] - team2_auxiliary[:, dimension]
        )
    embeddings_csv = output / "argus_oof_embeddings.csv"
    pd.DataFrame(frame).to_csv(embeddings_csv, index=False)

    final_path = output / "argus_player_encoder_final.pt"
    all_indices = np.flatnonzero(event_history_count > 0)
    all_indices = _limit(all_indices, max_train_events)
    all_normalisation = _pretrain_normalisation(
        data, all_indices, fixed_event_stats=fixed_stats
    )
    if not final_path.exists():
        final_history = _train_auxiliary(
            model,
            _EventPretrainCollator(data, all_normalisation),
            all_indices,
            epochs=final_refit_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=selected_device,
            seed=seed + 10_000,
        )
        _atomic_torch_save(
            {
                "format_version": FORMAT_VERSION,
                "model_config": asdict(config),
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "normalisation": all_normalisation.serialise(),
                "trained_events": len(all_indices),
                "history": final_history,
            },
            final_path,
        )
    else:
        final_checkpoint = torch.load(final_path, map_location="cpu", weights_only=True)
        final_history = final_checkpoint["history"]

    result: dict[str, object] = {
        "protocol": {
            "future_event_targets_used_for_oof_rows": False,
            "folds": [first_fold_year, last_fold_year],
            "coordinate_policy": "sequential expanding warm-start",
            "normalisation_fit_before": first_fold_year,
            "player_identity": use_player_identity,
            "team_identity": use_team_identity,
            "current_event_outcome_in_candidate": False,
        },
        "model": {
            **asdict(config),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "total_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        "rows": {
            "dataset_targets": len(data.target_start_ts),
            "oof_embeddings": len(rows),
            "without_embedding_before_first_fold": int(
                (target_starts < _cutoff(first_fold_year)).sum()
            ),
            "final_refit_events": len(all_indices),
        },
        "embedding_dimensions_per_side": d_model,
        "catboost_feature_columns": 3 * (d_model + len(auxiliary_fields)) + 1,
        "folds": fold_reports,
        "final_refit_history": final_history,
    }
    (output / "argus_pretrain_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
