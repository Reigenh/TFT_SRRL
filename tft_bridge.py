from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
USING_DIR = ROOT / "1using"
if str(USING_DIR) not in sys.path:
    sys.path.insert(0, str(USING_DIR))

from train_tft1 import (  # type: ignore  # noqa: E402
    SingleTargetTFT,
    TARGETS,
    TFT1Config,
    make_features,
    read_series,
)

TARGET_ORDER = ("wind", "sun", "load")
DEFAULT_TIME_INDICES = (0, 24, 48, 72)
LOAD_MAX = np.asarray(
    [-25, -5, -10, -25, -30, -7, -15, -8, -10, -5, -12, -4, -10, -3, -20, -4, -10, -4, -3, -14],
    dtype=np.float32,
)
WIND_MAX = 25.0
SUN_MAX = 20.0


@dataclass
class TFTTargetState:
    target: str
    model: SingleTargetTFT
    config: dict[str, Any]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor
    source_checkpoint: Path

    @property
    def context_length(self) -> int:
        return int(self.config["context_length"])

    @property
    def window_days(self) -> int:
        return self.context_length // 96

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feature_mean) / self.feature_std

    def denormalize_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.target_std + self.target_mean


def parse_time_indices(values: str | Iterable[int] | None) -> tuple[int, ...]:
    if values is None:
        return DEFAULT_TIME_INDICES
    if isinstance(values, str):
        result = tuple(int(part.strip()) for part in values.split(",") if part.strip())
    else:
        result = tuple(int(value) for value in values)
    if not result:
        raise ValueError("At least one time index is required.")
    for idx in result:
        if idx < 0 or idx >= 96:
            raise ValueError(f"TFT time index {idx} is outside [0, 95].")
    return result


def load_tft_states(model_dir: Path, device: torch.device) -> dict[str, TFTTargetState]:
    states: dict[str, TFTTargetState] = {}
    for target in TARGET_ORDER:
        checkpoint_path = model_dir / f"{target}_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"TFT checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = dict(payload["config"])
        model = SingleTargetTFT(TFT1Config(**config)).to(device)
        model.load_state_dict(payload["model"])
        model.train()
        states[target] = TFTTargetState(
            target=target,
            model=model,
            config=config,
            feature_mean=torch.as_tensor(payload["feature_mean"], dtype=torch.float32, device=device),
            feature_std=torch.as_tensor(payload["feature_std"], dtype=torch.float32, device=device),
            target_mean=torch.as_tensor(float(payload["target_mean"]), dtype=torch.float32, device=device),
            target_std=torch.as_tensor(float(payload["target_std"]), dtype=torch.float32, device=device),
            source_checkpoint=checkpoint_path,
        )
    return states


def save_tft_states(
    states: dict[str, TFTTargetState],
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for target, state in states.items():
        payload = {
            "target": target,
            "model": state.model.state_dict(),
            "config": state.config,
            "feature_mean": state.feature_mean.detach().cpu().numpy().tolist(),
            "feature_std": state.feature_std.detach().cpu().numpy().tolist(),
            "target_mean": float(state.target_mean.detach().cpu()),
            "target_std": float(state.target_std.detach().cpu()),
            "best_nrmse": float("nan"),
            "best_epoch": -1,
            "source_checkpoint": str(state.source_checkpoint),
            "dfl_metadata": metadata or {},
        }
        torch.save(payload, output_dir / f"{target}_best.pt")


def make_context_features(days: np.ndarray, window_days: int, day_index: int | None = None) -> np.ndarray:
    if day_index is None:
        day_index = len(days)
    if day_index < window_days or day_index > len(days):
        raise ValueError(
            f"day_index={day_index} must be in [{window_days}, {len(days)}] for window_days={window_days}."
        )
    tod = np.arange(96, dtype=np.float32) / 96.0
    context = days[day_index - window_days : day_index].reshape(-1)
    features = np.stack(
        [
            context,
            np.tile(np.sin(2.0 * math.pi * tod), window_days),
            np.tile(np.cos(2.0 * math.pi * tod), window_days),
            np.repeat(np.linspace(0.0, 1.0, window_days, dtype=np.float32), 96),
        ],
        axis=-1,
    )
    return features.astype(np.float32)


def load_decision_contexts(
    states: dict[str, TFTTargetState],
    data_dir: Path,
    day_index: int | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    contexts: dict[str, torch.Tensor] = {}
    for target, state in states.items():
        csv_name, column = TARGETS[target]
        days = read_series(data_dir / csv_name, column)
        x = make_context_features(days, state.window_days, day_index=day_index)
        x_tensor = torch.as_tensor(x[None, :, :], dtype=torch.float32, device=device)
        contexts[target] = state.normalize_x(x_tensor)
    return contexts


def load_supervised_tensors(
    states: dict[str, TFTTargetState],
    data_dir: Path,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for target, state in states.items():
        csv_name, column = TARGETS[target]
        days = read_series(data_dir / csv_name, column)
        x, y = make_features(days, state.window_days)
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
        y_norm = (y_tensor - state.target_mean) / state.target_std
        tensors[target] = (state.normalize_x(x_tensor), y_norm)
    return tensors


def predict_raw96(
    states: dict[str, TFTTargetState],
    contexts: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    predictions: dict[str, torch.Tensor] = {}
    for target, state in states.items():
        pred_norm = state.model(contexts[target])
        predictions[target] = state.denormalize_y(pred_norm)
    return predictions


def selected_raw_to_profile_flat(
    raw96: dict[str, torch.Tensor],
    time_indices: tuple[int, ...] = DEFAULT_TIME_INDICES,
    clip_raw: bool = True,
) -> torch.Tensor:
    wind_raw = raw96["wind"][:, list(time_indices)]
    sun_raw = raw96["sun"][:, list(time_indices)]
    load_raw = raw96["load"][:, list(time_indices)]
    if clip_raw:
        wind_raw = torch.clamp(wind_raw, min=0.0)
        sun_raw = torch.clamp(sun_raw, min=0.0)
        load_raw = torch.clamp(load_raw, min=1e-6)

    wind_scale = WIND_MAX / torch.clamp(torch.max(torch.abs(wind_raw), dim=1, keepdim=True).values, min=1e-6)
    sun_scale = SUN_MAX / torch.clamp(torch.max(torch.abs(sun_raw), dim=1, keepdim=True).values, min=1e-6)
    load_scale = torch.as_tensor(np.abs(LOAD_MAX), dtype=load_raw.dtype, device=load_raw.device).view(1, -1, 1)
    load_denominator = torch.clamp(torch.max(torch.abs(load_raw), dim=1, keepdim=True).values, min=1e-6)

    wind = wind_raw * wind_scale
    sun = sun_raw * sun_scale
    load = -load_raw[:, None, :] * load_scale / load_denominator[:, None, :]
    return torch.cat([wind, sun, load.reshape(load.shape[0], -1)], dim=1)


def flat_profile_bounds(
    time_count: int = 4,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    caps = torch.as_tensor(np.abs(LOAD_MAX), dtype=dtype, device=device)
    wind_scale = torch.full((time_count,), WIND_MAX, dtype=dtype, device=device)
    sun_scale = torch.full((time_count,), SUN_MAX, dtype=dtype, device=device)
    load_scale = caps.repeat_interleave(time_count)
    scale = torch.cat([wind_scale, sun_scale, load_scale])
    lower = torch.cat([torch.zeros_like(wind_scale), torch.zeros_like(sun_scale), -load_scale])
    upper = torch.cat([wind_scale, sun_scale, torch.zeros_like(load_scale)])
    return lower, upper, scale


def flat_to_arrays(flat: np.ndarray, time_count: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    expected = 2 * time_count + len(LOAD_MAX) * time_count
    if flat.shape[0] != expected:
        raise ValueError(f"Expected flat profile length {expected}, got {flat.shape[0]}.")
    wind = flat[:time_count].copy()
    sun = flat[time_count : 2 * time_count].copy()
    load = flat[2 * time_count :].reshape(len(LOAD_MAX), time_count).copy()
    return wind, sun, load


def arrays_to_flat(wind: np.ndarray, sun: np.ndarray, load: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(wind, dtype=np.float32).reshape(-1),
            np.asarray(sun, dtype=np.float32).reshape(-1),
            np.asarray(load, dtype=np.float32).reshape(-1),
        ]
    )


def save_profile_npz(path: Path, flat: np.ndarray, time_count: int = 4) -> None:
    wind, sun, load = flat_to_arrays(flat, time_count=time_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, wind=wind, sun=sun, load=load)


def export_forecast_tables(
    output_dir: Path,
    raw96: dict[str, torch.Tensor],
    flat_profile: torch.Tensor,
    time_indices: tuple[int, ...] = DEFAULT_TIME_INDICES,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_np = {key: value.detach().cpu().numpy()[0] for key, value in raw96.items()}
    with open(output_dir / "tft_raw_96_forecast.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "wind_raw", "sun_raw", "load_raw"])
        for idx in range(96):
            writer.writerow([idx + 1, raw_np["wind"][idx], raw_np["sun"][idx], raw_np["load"][idx]])

    wind, sun, load = flat_to_arrays(flat_profile.detach().cpu().numpy()[0], time_count=len(time_indices))
    with open(output_dir / "dispatch_profile_4point.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_index", "step", "wind_norm", "sun_norm", *[f"load_bus_{idx + 1}_norm" for idx in range(len(LOAD_MAX))]])
        for pos, time_index in enumerate(time_indices):
            writer.writerow([time_index, time_index + 1, wind[pos], sun[pos], *load[:, pos].tolist()])


def all_tft_parameters(states: dict[str, TFTTargetState]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for state in states.values():
        params.extend(list(state.model.parameters()))
    return params

