from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .baseline import _binary_metrics, _confidence_slices
from .light_argus_data import FORMAT_VERSION
from .map_baseline import _slice_metrics


@dataclass(frozen=True)
class LightArgusConfig:
    player_vocab_size: int
    team_vocab_size: int
    map_vocab_size: int
    tier_vocab_size: int
    event_type_vocab_size: int
    version_vocab_size: int
    bo_vocab_size: int
    role_vocab_size: int
    event_numeric_size: int
    side_numeric_size: int
    shared_numeric_size: int
    max_history: int
    d_model: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.10
    use_player_identity: bool = True
    use_team_identity: bool = True


@dataclass
class _Normalisation:
    event_mean: np.ndarray
    event_std: np.ndarray
    side_mean: np.ndarray
    side_std: np.ndarray
    shared_mean: np.ndarray
    shared_std: np.ndarray
    seen_players: np.ndarray
    seen_teams: np.ndarray

    def serialise(self) -> dict[str, object]:
        return {
            "event_mean": self.event_mean.tolist(),
            "event_std": self.event_std.tolist(),
            "side_mean": self.side_mean.tolist(),
            "side_std": self.side_std.tolist(),
            "shared_mean": self.shared_mean.tolist(),
            "shared_std": self.shared_std.tolist(),
            "seen_players": np.flatnonzero(self.seen_players).tolist(),
            "seen_teams": np.flatnonzero(self.seen_teams).tolist(),
        }


class _ArrayStore:
    _EVENT_ARRAYS = (
        "event_numeric",
        "event_player",
        "event_team",
        "event_opponent",
        "event_map",
        "event_tier",
        "event_event_type",
        "event_version",
        "event_known_ts",
        "event_match_id",
        "event_game_id",
        "player_offsets",
    )
    _TARGET_ARRAYS = (
        "target_history_indices",
        "target_players",
        "target_teams",
        "target_side_numeric",
        "target_shared_numeric",
        "target_start_ts",
        "target_known_ts",
        "target_label",
        "target_weight",
        "target_map",
        "target_tier",
        "target_event_type",
        "target_version",
        "target_bo_type",
        "target_role",
        "target_slot",
        "target_match_id",
    )

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.metadata = json.loads(
            (self.directory / "dataset.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                "unsupported Light Argus dataset format: "
                f"{self.metadata.get('format_version')!r}"
            )
        self.vocabularies = json.loads(
            (self.directory / "vocabularies.json").read_text(encoding="utf-8")
        )
        for name in (*self._EVENT_ARRAYS, *self._TARGET_ARRAYS):
            setattr(
                self,
                name,
                np.load(self.directory / f"{name}.npy", mmap_mode="r"),
            )

    def model_config(
        self,
        *,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
        use_player_identity: bool,
        use_team_identity: bool,
    ) -> LightArgusConfig:
        vocab = self.vocabularies
        return LightArgusConfig(
            player_vocab_size=len(vocab["player"]),
            team_vocab_size=len(vocab["team"]),
            map_vocab_size=len(vocab["map"]),
            tier_vocab_size=len(vocab["tier"]),
            event_type_vocab_size=len(vocab["event_type"]),
            version_vocab_size=len(vocab["version"]),
            bo_vocab_size=len(vocab["bo_type"]),
            role_vocab_size=len(vocab["role"]),
            event_numeric_size=self.event_numeric.shape[1] * 2 + 2,
            side_numeric_size=self.target_side_numeric.shape[2] * 2,
            shared_numeric_size=self.target_shared_numeric.shape[1] * 2,
            max_history=int(self.metadata["max_history"]),
            d_model=d_model,
            layers=layers,
            heads=heads,
            dropout=dropout,
            use_player_identity=use_player_identity,
            use_team_identity=use_team_identity,
        )


def _finite_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.where(finite, values, 0.0).sum(axis=0, dtype=np.float64)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    centered = np.where(finite, values - mean, 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=0, dtype=np.float64),
        count,
        out=np.ones_like(total),
        where=count > 0,
    )
    std = np.sqrt(variance)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _fit_normalisation(
    data: _ArrayStore, target_indices: np.ndarray, cutoff_ts: int
) -> _Normalisation:
    event_indices = np.flatnonzero(np.asarray(data.event_known_ts) < cutoff_ts)
    event_mean, event_std = _finite_stats(
        np.asarray(data.event_numeric[event_indices], dtype=np.float32)
    )
    side = np.asarray(data.target_side_numeric[target_indices], dtype=np.float32)
    side_mean, side_std = _finite_stats(side.reshape(-1, side.shape[-1]))
    shared_mean, shared_std = _finite_stats(
        np.asarray(data.target_shared_numeric[target_indices], dtype=np.float32)
    )
    seen_players = np.zeros(len(data.vocabularies["player"]), dtype=bool)
    seen_players[np.asarray(data.target_players[target_indices]).reshape(-1)] = True
    seen_players[0] = True
    seen_teams = np.zeros(len(data.vocabularies["team"]), dtype=bool)
    seen_teams[np.asarray(data.target_teams[target_indices]).reshape(-1)] = True
    seen_teams[np.asarray(data.event_team[event_indices])] = True
    seen_teams[np.asarray(data.event_opponent[event_indices])] = True
    seen_teams[0] = True
    return _Normalisation(
        event_mean,
        event_std,
        side_mean,
        side_std,
        shared_mean,
        shared_std,
        seen_players,
        seen_teams,
    )


def _normalise(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    scaled = np.where(finite, (values - mean) / std, 0.0)
    scaled = np.clip(scaled, -8.0, 8.0)
    return np.concatenate((scaled, (~finite).astype(np.float32)), axis=-1).astype(
        np.float32
    )


class _BatchCollator:
    def __init__(self, data: _ArrayStore, normalisation: _Normalisation):
        self.data = data
        self.normalisation = normalisation

    def __call__(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        indices = np.asarray(indices, dtype=np.int64)
        history = np.asarray(self.data.target_history_indices[indices], dtype=np.int64)
        history_mask = history >= 0
        safe_history = np.maximum(history, 0)
        event_numeric = np.asarray(
            self.data.event_numeric[safe_history], dtype=np.float32
        )
        event_numeric = _normalise(
            event_numeric,
            self.normalisation.event_mean,
            self.normalisation.event_std,
        )
        start_ts = np.asarray(self.data.target_start_ts[indices], dtype=np.int64)
        event_ts = np.asarray(self.data.event_known_ts[safe_history], dtype=np.int64)
        age_days = np.maximum(
            0.0,
            (start_ts[:, None, None] - event_ts).astype(np.float32) / 86_400.0,
        )
        age_feature = np.clip(np.log1p(age_days) / math.log1p(3650.0), 0.0, 1.5)
        age_feature[~history_mask] = 0.0
        recency = np.linspace(
            0.0,
            1.0,
            history.shape[-1],
            dtype=np.float32,
        )
        recency = np.broadcast_to(recency, history.shape).copy()
        recency[~history_mask] = 0.0
        event_numeric = np.concatenate(
            (event_numeric, age_feature[..., None], recency[..., None]), axis=-1
        )
        event_team = np.asarray(self.data.event_team[safe_history], dtype=np.int64)
        event_opponent = np.asarray(
            self.data.event_opponent[safe_history], dtype=np.int64
        )
        event_team = np.where(self.normalisation.seen_teams[event_team], event_team, 0)
        event_opponent = np.where(
            self.normalisation.seen_teams[event_opponent], event_opponent, 0
        )
        players = np.asarray(self.data.target_players[indices], dtype=np.int64)
        players = np.where(self.normalisation.seen_players[players], players, 0)
        teams = np.asarray(self.data.target_teams[indices], dtype=np.int64)
        teams = np.where(self.normalisation.seen_teams[teams], teams, 0)
        side_numeric = _normalise(
            np.asarray(self.data.target_side_numeric[indices], dtype=np.float32),
            self.normalisation.side_mean,
            self.normalisation.side_std,
        )
        shared_numeric = _normalise(
            np.asarray(self.data.target_shared_numeric[indices], dtype=np.float32),
            self.normalisation.shared_mean,
            self.normalisation.shared_std,
        )
        result = {
            "history_mask": history_mask,
            "event_numeric": event_numeric,
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
            "target_players": players,
            "target_teams": teams,
            "target_side_numeric": side_numeric,
            "target_shared_numeric": shared_numeric,
            "target_map": np.asarray(self.data.target_map[indices], dtype=np.int64),
            "target_tier": np.asarray(self.data.target_tier[indices], dtype=np.int64),
            "target_event_type": np.asarray(
                self.data.target_event_type[indices], dtype=np.int64
            ),
            "target_version": np.asarray(
                self.data.target_version[indices], dtype=np.int64
            ),
            "target_bo_type": np.asarray(
                self.data.target_bo_type[indices], dtype=np.int64
            ),
            "target_role": np.asarray(self.data.target_role[indices], dtype=np.int64),
            "target_slot": np.asarray(self.data.target_slot[indices], dtype=np.int64),
            "target_label": np.asarray(
                self.data.target_label[indices], dtype=np.float32
            ),
            "target_weight": np.asarray(
                self.data.target_weight[indices], dtype=np.float32
            ),
            "row_index": indices,
        }
        return {
            key: torch.from_numpy(np.asarray(value)) for key, value in result.items()
        }


class LightTargetAwareArgus(nn.Module):
    """Small early-binding encoder over ten causal player histories."""

    def __init__(self, config: LightArgusConfig):
        super().__init__()
        if config.d_model % config.heads:
            raise ValueError("d_model must be divisible by heads")
        self.config = config
        dimension = config.d_model
        self.player_embedding = nn.Embedding(
            config.player_vocab_size, dimension, padding_idx=0
        )
        self.team_embedding = nn.Embedding(
            config.team_vocab_size, dimension, padding_idx=0
        )
        if not config.use_player_identity:
            self.player_embedding.weight.requires_grad_(False)
        if not config.use_team_identity:
            self.team_embedding.weight.requires_grad_(False)
        self.map_embedding = nn.Embedding(
            config.map_vocab_size, dimension, padding_idx=0
        )
        self.tier_embedding = nn.Embedding(
            config.tier_vocab_size, dimension, padding_idx=0
        )
        self.event_type_embedding = nn.Embedding(
            config.event_type_vocab_size, dimension, padding_idx=0
        )
        self.version_embedding = nn.Embedding(
            config.version_vocab_size, dimension, padding_idx=0
        )
        self.bo_embedding = nn.Embedding(config.bo_vocab_size, dimension, padding_idx=0)
        self.role_embedding = nn.Embedding(
            config.role_vocab_size, dimension, padding_idx=0
        )
        self.slot_embedding = nn.Embedding(8, dimension, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_history + 1, dimension)
        self.token_type_embedding = nn.Embedding(2, dimension)
        self.event_numeric_projection = nn.Linear(config.event_numeric_size, dimension)
        self.side_numeric_projection = nn.Linear(config.side_numeric_size, dimension)
        self.shared_numeric_projection = nn.Linear(
            config.shared_numeric_size, dimension
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=config.heads,
            dim_feedforward=4 * dimension,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            layer, config.layers, norm=nn.LayerNorm(dimension)
        )
        self.input_norm = nn.LayerNorm(dimension)
        self.roster_pool = nn.Sequential(
            nn.Linear(2 * dimension, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
        )
        self.side_context = nn.Sequential(
            nn.Linear(config.side_numeric_size, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
        )
        self.shared_context = nn.Sequential(
            nn.Linear(config.shared_numeric_size, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
        )
        self.side_scorer = nn.Sequential(
            nn.Linear(4 * dimension, 2 * dimension),
            nn.GELU(),
            nn.LayerNorm(2 * dimension),
            nn.Linear(2 * dimension, 1),
        )

    def _shared_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.map_embedding(batch["target_map"])
            + self.tier_embedding(batch["target_tier"])
            + self.event_type_embedding(batch["target_event_type"])
            + self.version_embedding(batch["target_version"])
            + self.bo_embedding(batch["target_bo_type"])
            + self.role_embedding(batch["target_role"])
            + self.slot_embedding(batch["target_slot"].clamp(0, 7))
            + self.shared_numeric_projection(batch["target_shared_numeric"])
        )

    def _player_identity(self, player_ids: torch.Tensor) -> torch.Tensor:
        if self.config.use_player_identity:
            return self.player_embedding(player_ids)
        return torch.zeros(
            (*player_ids.shape, self.config.d_model),
            dtype=self.player_embedding.weight.dtype,
            device=player_ids.device,
        )

    def _team_identity(self, team_ids: torch.Tensor) -> torch.Tensor:
        if self.config.use_team_identity:
            return self.team_embedding(team_ids)
        return torch.zeros(
            (*team_ids.shape, self.config.d_model),
            dtype=self.team_embedding.weight.dtype,
            device=team_ids.device,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["history_mask"]
        batch_size, players, history_length = mask.shape
        if players != 10 or history_length != self.config.max_history:
            raise ValueError("Light Argus expects ten fixed-length player histories")
        player_embedding = self._player_identity(batch["target_players"])
        history = self.event_numeric_projection(batch["event_numeric"])
        history = history + player_embedding[:, :, None, :]
        history = history + self._team_identity(batch["event_team"])
        history = history + 0.5 * self._team_identity(batch["event_opponent"])
        history = history + self.map_embedding(batch["event_map"])
        history = history + self.tier_embedding(batch["event_tier"])
        history = history + self.event_type_embedding(batch["event_event_type"])
        history = history + self.version_embedding(batch["event_version"])
        positions = torch.arange(history_length, device=history.device)
        history = history + self.position_embedding(positions)[None, None, :, :]
        history = history + self.token_type_embedding.weight[0]

        current_teams = torch.cat(
            (
                batch["target_teams"][:, :1].expand(-1, 5),
                batch["target_teams"][:, 1:].expand(-1, 5),
            ),
            dim=1,
        )
        side_numeric = torch.cat(
            (
                batch["target_side_numeric"][:, :1].expand(-1, 5, -1),
                batch["target_side_numeric"][:, 1:].expand(-1, 5, -1),
            ),
            dim=1,
        )
        shared = self._shared_embedding(batch)
        query = player_embedding + self._team_identity(current_teams)
        query = query + self.side_numeric_projection(side_numeric)
        query = query + shared[:, None, :]
        query = query + self.position_embedding.weight[history_length]
        query = query + self.token_type_embedding.weight[1]

        tokens = torch.cat((history, query[:, :, None, :]), dim=2)
        tokens = self.input_norm(tokens).reshape(
            batch_size * players, history_length + 1, self.config.d_model
        )
        padding = torch.cat(
            (
                ~mask,
                torch.zeros(
                    (batch_size, players, 1), dtype=torch.bool, device=mask.device
                ),
            ),
            dim=2,
        ).reshape(batch_size * players, history_length + 1)
        encoded = self.history_encoder(tokens, src_key_padding_mask=padding)
        player_states = encoded[:, -1].reshape(batch_size, players, self.config.d_model)

        def pool(roster: torch.Tensor) -> torch.Tensor:
            return self.roster_pool(
                torch.cat((roster.mean(dim=1), roster.amax(dim=1)), dim=-1)
            )

        team1 = pool(player_states[:, :5]) + self._team_identity(
            batch["target_teams"][:, 0]
        )
        team2 = pool(player_states[:, 5:]) + self._team_identity(
            batch["target_teams"][:, 1]
        )
        side1 = self.side_context(batch["target_side_numeric"][:, 0])
        side2 = self.side_context(batch["target_side_numeric"][:, 1])
        shared_context = self.shared_context(batch["target_shared_numeric"])
        score1 = self.side_scorer(
            torch.cat((team1, team2, side1, shared_context), dim=-1)
        ).squeeze(-1)
        score2 = self.side_scorer(
            torch.cat((team2, team1, side2, shared_context), dim=-1)
        ).squeeze(-1)
        return score1 - score2


def swap_batch_sides(
    batch: dict[str, torch.Tensor], *, flip_label: bool = True
) -> dict[str, torch.Tensor]:
    result = {key: value for key, value in batch.items()}
    for key in (
        "history_mask",
        "event_numeric",
        "event_team",
        "event_opponent",
        "event_map",
        "event_tier",
        "event_event_type",
        "event_version",
        "target_players",
    ):
        result[key] = torch.cat((batch[key][:, 5:], batch[key][:, :5]), dim=1)
    result["target_teams"] = batch["target_teams"].flip(1)
    result["target_side_numeric"] = batch["target_side_numeric"].flip(1)
    if flip_label and "target_label" in batch:
        result["target_label"] = 1.0 - batch["target_label"]
    return result


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return selected


def _move(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
        if key != "row_index"
    }


def _batches(
    indices: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator,
):
    order = np.array(indices, copy=True)
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )


def _train_epoch(
    model: LightTargetAwareArgus,
    collator: _BatchCollator,
    indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for raw_indices in _batches(indices, batch_size, shuffle=True, rng=rng):
        batch = _move(collator(raw_indices), device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            logits = model(batch)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, batch["target_label"], reduction="none"
            )
            weights = batch["target_weight"]
            loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.detach()) * len(raw_indices)
        total_rows += len(raw_indices)
    return total_loss / max(1, total_rows)


@torch.inference_mode()
def _predict(
    model: LightTargetAwareArgus,
    collator: _BatchCollator,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    model.eval()
    probabilities: list[np.ndarray] = []
    symmetry_error = 0.0
    rng = np.random.default_rng(0)
    for raw_indices in _batches(indices, batch_size, shuffle=False, rng=rng):
        batch = _move(collator(raw_indices), device)
        with _autocast(device):
            logits = model(batch)
            mirrored = model(swap_batch_sides(batch, flip_label=False))
        symmetry_error = max(
            symmetry_error, float((logits + mirrored).abs().max().cpu())
        )
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probabilities), symmetry_error


def _scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, int(total_steps * 0.05))

    def factor(step: int) -> float:
        if step < warmup:
            return max(1e-3, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _train_model(
    model: LightTargetAwareArgus,
    collator: _BatchCollator,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> tuple[LightTargetAwareArgus, int, list[dict[str, float]]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    steps_per_epoch = max(1, math.ceil(len(train_indices) / batch_size))
    scheduler = _scheduler(optimizer, steps_per_epoch * epochs)
    rng = np.random.default_rng(seed)
    best_epoch = 0
    best_logloss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(
            model,
            collator,
            train_indices,
            optimizer,
            scheduler,
            batch_size=batch_size,
            device=device,
            rng=rng,
        )
        row: dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss}
        if validation_indices is not None:
            probability, _ = _predict(
                model,
                collator,
                validation_indices,
                batch_size=batch_size,
                device=device,
            )
            labels = np.asarray(
                collator.data.target_label[validation_indices], dtype=np.float32
            )
            metrics = _binary_metrics(labels, probability)
            row.update(
                {
                    "validation_logloss": metrics["logloss"],
                    "validation_accuracy": metrics["accuracy"],
                    "validation_auc": metrics["auc"],
                }
            )
            if metrics["logloss"] < best_logloss - 1e-5:
                best_logloss = metrics["logloss"]
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
        else:
            best_epoch = epoch
        history.append(row)
        print(
            "light-argus "
            + " ".join(f"{key}={value:.5f}" for key, value in row.items()),
            file=sys.stderr,
            flush=True,
        )
        if validation_indices is not None and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def _cutoff(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.astimezone(UTC).timestamp())


def _subsample(
    indices: np.ndarray, maximum: int | None, rng: np.random.Generator
) -> np.ndarray:
    if maximum is None or len(indices) <= maximum:
        return indices
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def _metadata_rows(path: Path, indices: np.ndarray) -> list[dict[str, object]]:
    wanted = {int(value) for value in indices}
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if int(row["row_index"]) in wanted:
                rows.append(row)
    by_index = {int(row["row_index"]): row for row in rows}
    return [by_index[int(index)] for index in indices]


def train_light_argus(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    train_before: str = "2025-01-01",
    test_from: str = "2026-01-01",
    epochs: int = 12,
    patience: int = 3,
    batch_size: int = 256,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    d_model: int = 128,
    layers: int = 3,
    heads: int = 4,
    dropout: float = 0.10,
    use_player_identity: bool = True,
    use_team_identity: bool = True,
    device: str = "auto",
    seed: int = 20260827,
    refit: bool = True,
    monthly_refit: bool = False,
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
    max_test_rows: int | None = None,
    catboost_predictions_csv: str | Path | None = None,
) -> dict[str, object]:
    """Tune on 2025, refit before 2026, and score the 2026 map holdout."""
    import pandas as pd

    if min(epochs, patience, batch_size) < 1:
        raise ValueError("epochs, patience and batch_size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    data = _ArrayStore(dataset_dir)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_device = _device(device)
    tune_cutoff, test_cutoff = _cutoff(train_before), _cutoff(test_from)
    starts = np.asarray(data.target_start_ts)
    known = np.asarray(data.target_known_ts)
    tune_train = np.flatnonzero((starts < tune_cutoff) & (known < tune_cutoff))
    validation = np.flatnonzero(
        (starts >= tune_cutoff) & (starts < test_cutoff) & (known < test_cutoff)
    )
    full_train = np.flatnonzero((starts < test_cutoff) & (known < test_cutoff))
    test = np.flatnonzero(starts >= test_cutoff)
    rng = np.random.default_rng(seed)
    tune_train = _subsample(tune_train, max_train_rows, rng)
    validation = _subsample(validation, max_validation_rows, rng)
    test = _subsample(test, max_test_rows, rng)
    if min(len(tune_train), len(validation), len(test)) == 0:
        raise ValueError("time split produced an empty train, validation or test set")

    config = data.model_config(
        d_model=d_model,
        layers=layers,
        heads=heads,
        dropout=dropout,
        use_player_identity=use_player_identity,
        use_team_identity=use_team_identity,
    )
    tune_normalisation = _fit_normalisation(data, tune_train, tune_cutoff)
    tune_collator = _BatchCollator(data, tune_normalisation)
    tune_model = LightTargetAwareArgus(config)
    tune_model, best_epoch, tuning_history = _train_model(
        tune_model,
        tune_collator,
        tune_train,
        validation,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=selected_device,
        seed=seed,
    )
    if best_epoch < 1:
        raise ValueError("training did not select an epoch")

    refit_history: list[dict[str, object]] = []
    fold_metrics: list[dict[str, object]] = []
    checkpoint_cutoff = test_from
    if monthly_refit and not refit:
        raise ValueError("monthly_refit requires refit")
    if monthly_refit:
        fold_indices: list[np.ndarray] = []
        fold_probabilities: list[np.ndarray] = []
        symmetry_error = 0.0
        first_month = pd.Timestamp(test_from, tz="UTC")
        last_target = pd.to_datetime(int(starts.max()), unit="s", utc=True)
        fold_starts = pd.date_range(first_month, last_target, freq="MS")
        final_model = tune_model
        final_normalisation = tune_normalisation
        final_collator = tune_collator
        for fold_number, fold_start in enumerate(fold_starts, start=1):
            fold_end = fold_start + pd.offsets.MonthBegin(1)
            fold_cutoff = int(fold_start.timestamp())
            fold_end_ts = int(fold_end.timestamp())
            current_train = np.flatnonzero(
                (starts < fold_cutoff) & (known < fold_cutoff)
            )
            current_train = _subsample(current_train, max_train_rows, rng)
            current_test = np.flatnonzero(
                (starts >= fold_cutoff) & (starts < fold_end_ts)
            )
            current_test = _subsample(current_test, max_test_rows, rng)
            if min(len(current_train), len(current_test)) == 0:
                continue
            current_normalisation = _fit_normalisation(data, current_train, fold_cutoff)
            current_collator = _BatchCollator(data, current_normalisation)
            torch.manual_seed(seed + 100 + fold_number)
            current_model = LightTargetAwareArgus(config)
            current_model, _, current_history = _train_model(
                current_model,
                current_collator,
                current_train,
                None,
                epochs=best_epoch,
                patience=patience,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=selected_device,
                seed=seed + 100 + fold_number,
            )
            current_probability, current_symmetry = _predict(
                current_model,
                current_collator,
                current_test,
                batch_size=batch_size,
                device=selected_device,
            )
            current_labels = np.asarray(
                data.target_label[current_test], dtype=np.float32
            )
            current_metrics = _binary_metrics(current_labels, current_probability)
            fold_metrics.append(
                {
                    "month": fold_start.strftime("%Y-%m"),
                    "train_rows": len(current_train),
                    "test_rows": len(current_test),
                    **current_metrics,
                }
            )
            refit_history.append(
                {
                    "month": fold_start.strftime("%Y-%m"),
                    "epochs": current_history,
                }
            )
            fold_indices.append(current_test)
            fold_probabilities.append(current_probability)
            symmetry_error = max(symmetry_error, current_symmetry)
            final_model = current_model
            final_normalisation = current_normalisation
            final_collator = current_collator
            full_train = current_train
            checkpoint_cutoff = fold_start.date().isoformat()
            print(
                f"light-argus monthly {fold_start:%Y-%m}: "
                f"accuracy={current_metrics['accuracy']:.5f} "
                f"logloss={current_metrics['logloss']:.5f}",
                file=sys.stderr,
                flush=True,
            )
        if not fold_indices:
            raise ValueError("monthly refit produced no folds")
        test = np.concatenate(fold_indices)
        probability = np.concatenate(fold_probabilities)
    elif refit:
        full_train = _subsample(full_train, max_train_rows, rng)
        final_normalisation = _fit_normalisation(data, full_train, test_cutoff)
        final_collator = _BatchCollator(data, final_normalisation)
        torch.manual_seed(seed + 1)
        final_model = LightTargetAwareArgus(config)
        final_model, _, refit_history = _train_model(
            final_model,
            final_collator,
            full_train,
            None,
            epochs=best_epoch,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=selected_device,
            seed=seed + 1,
        )
        probability, symmetry_error = _predict(
            final_model,
            final_collator,
            test,
            batch_size=batch_size,
            device=selected_device,
        )
    else:
        final_model = tune_model
        final_normalisation = tune_normalisation
        final_collator = tune_collator
        full_train = tune_train
        probability, symmetry_error = _predict(
            final_model,
            final_collator,
            test,
            batch_size=batch_size,
            device=selected_device,
        )
    labels = np.asarray(data.target_label[test], dtype=np.float32)
    metadata = _metadata_rows(data.directory / "target_metadata.jsonl", test)
    predictions = pd.DataFrame(metadata)
    predictions["team1_win_probability"] = probability
    if monthly_refit:
        predictions["training_cutoff"] = predictions["start_at"].str.slice(0, 7)
    else:
        predictions["training_cutoff"] = test_from
    predictions.to_csv(output / "light_argus_test_predictions.csv", index=False)

    team_elo_probability = 1.0 / (
        1.0 + 10.0 ** (-predictions.diff_elo.to_numpy(float) / 400.0)
    )
    map_elo_probability = 1.0 / (
        1.0 + 10.0 ** (-predictions.diff_target_map_elo.to_numpy(float) / 400.0)
    )
    baselines: dict[str, object] = {
        "constant_0_5": _binary_metrics(labels, np.full(len(labels), 0.5)),
        "series_team_elo": _binary_metrics(labels, team_elo_probability),
        "target_map_elo": _binary_metrics(labels, map_elo_probability),
    }
    if catboost_predictions_csv is not None:
        catboost = pd.read_csv(
            catboost_predictions_csv,
            usecols=["map_row_id", "team1_win_probability"],
        ).rename(columns={"team1_win_probability": "catboost_probability"})
        matched = predictions[["map_row_id", "team1_win"]].merge(
            catboost, on="map_row_id", how="inner", validate="one_to_one"
        )
        baselines["map_catboost_matched"] = _binary_metrics(
            matched.team1_win, matched.catboost_probability
        )

    history_counts = (np.asarray(data.target_history_indices[test]) >= 0).sum(axis=2)
    metrics: dict[str, object] = {
        "protocol": {
            "prediction_target": "individual_map_winner_after_veto",
            "train_before": train_before,
            "validation": f"{train_before} <= start_at < {test_from}",
            "test_from": test_from,
            "refit_on_all_labels_known_before_test": refit,
            "monthly_retraining": monthly_refit,
            "future_labels_used": False,
            "same_series_maps_share_pre_series_history": True,
            "target_aware_early_binding": True,
            "max_history_per_player": config.max_history,
        },
        "model": {
            **asdict(config),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in final_model.parameters()
                if parameter.requires_grad
            ),
            "total_parameters": sum(
                parameter.numel() for parameter in final_model.parameters()
            ),
            "best_tuning_epoch": best_epoch,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "device": str(selected_device),
        },
        "rows": {
            "tuning_train": len(tune_train),
            "validation": len(validation),
            "refit_train": len(full_train),
            "test": len(test),
        },
        "overall": _binary_metrics(labels, probability),
        "confidence_slices": _confidence_slices(labels, probability),
        "slices": _slice_metrics(predictions),
        "baselines": baselines,
        "history_coverage": {
            "mean_maps_per_player": float(history_counts.mean()),
            "median_maps_per_player": float(np.median(history_counts)),
            "player_sequences_empty_rate": float(np.mean(history_counts == 0)),
            "player_sequences_full_rate": float(
                np.mean(history_counts == config.max_history)
            ),
        },
        "symmetry": {
            "max_abs_logit_plus_swapped_logit": symmetry_error,
        },
        "tuning_history": tuning_history,
        "refit_history": refit_history,
        "folds": fold_metrics,
    }
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "model_config": asdict(config),
        "state_dict": {
            key: value.detach().cpu() for key, value in final_model.state_dict().items()
        },
        "normalisation": final_normalisation.serialise(),
        "train_before": train_before,
        "test_from": test_from,
        "checkpoint_cutoff": checkpoint_cutoff,
        "best_tuning_epoch": best_epoch,
    }
    torch.save(checkpoint, output / "light_argus_map_winner.pt")
    (output / "light_argus_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics
