from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .tft_bridge import flat_profile_bounds, flat_to_arrays
except ImportError:  # pragma: no cover
    from tft_bridge import flat_profile_bounds, flat_to_arrays


class DecisionSurrogate(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = (128, 128), dropout: float = 0.05) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            last_dim = hidden
        layers.append(nn.Linear(last_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SurrogateBundle:
    model: DecisionSurrogate
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    metadata: dict[str, Any]

    def predict_normalized(self, flat_profile: torch.Tensor) -> torch.Tensor:
        x = (flat_profile - self.x_mean) / self.x_std
        return self.model(x)

    def predict(self, flat_profile: torch.Tensor) -> torch.Tensor:
        return self.predict_normalized(flat_profile) * self.y_std + self.y_mean

    def normalized_cost_target(self, cost_target: float) -> torch.Tensor:
        return (torch.as_tensor(cost_target, dtype=self.y_mean.dtype, device=self.y_mean.device) - self.y_mean[1]) / self.y_std[1]


def make_local_profile_samples(
    base_flat: np.ndarray,
    sample_count: int,
    radius: float,
    seed: int,
    include_central: bool = False,
    central_step: float = 0.01,
) -> np.ndarray:
    base = np.asarray(base_flat, dtype=np.float32).reshape(1, -1)
    if base.shape[1] % 22 != 0:
        raise ValueError(f"Flat profile length {base.shape[1]} is not compatible with wind/sun/load layout.")
    time_count = base.shape[1] // 22
    device = torch.device("cpu")
    lower_t, upper_t, scale_t = flat_profile_bounds(time_count=time_count, device=device)
    lower = lower_t.numpy()
    upper = upper_t.numpy()
    scale = scale_t.numpy()
    rng = np.random.default_rng(seed)
    rows = [base.reshape(-1)]

    if include_central:
        for idx in range(base.shape[1]):
            step = central_step * scale[idx]
            plus = base.reshape(-1).copy()
            minus = base.reshape(-1).copy()
            plus[idx] += step
            minus[idx] -= step
            rows.append(np.clip(plus, lower, upper))
            rows.append(np.clip(minus, lower, upper))

    if sample_count > 0:
        noise = rng.uniform(-radius, radius, size=(sample_count, base.shape[1])).astype(np.float32) * scale.reshape(1, -1)
        random_rows = np.clip(base + noise, lower.reshape(1, -1), upper.reshape(1, -1))
        rows.extend(random_rows)

    return np.asarray(rows, dtype=np.float32)


def profile_rows_to_arrays(rows: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if rows.shape[1] % 22 != 0:
        raise ValueError(f"Flat profile length {rows.shape[1]} is not compatible with wind/sun/load layout.")
    time_count = rows.shape[1] // 22
    return [flat_to_arrays(row, time_count=time_count) for row in rows]


def train_surrogate(
    x: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_sizes: tuple[int, ...] = (128, 128),
    dropout: float = 0.05,
    seed: int = 0,
    device: str | torch.device = "cpu",
    metadata: dict[str, Any] | None = None,
) -> SurrogateBundle:
    if len(x) < 2:
        raise ValueError("At least two samples are required to fit a surrogate.")
    torch.manual_seed(seed)
    device = torch.device(device)
    x_np = np.asarray(x, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.float32)
    x_mean = x_np.mean(axis=0)
    x_std = np.maximum(x_np.std(axis=0), 1e-6)
    y_mean = y_np.mean(axis=0)
    y_std = np.maximum(y_np.std(axis=0), 1e-6)

    x_norm = (x_np - x_mean) / x_std
    y_norm = (y_np - y_mean) / y_std
    dataset = TensorDataset(torch.as_tensor(x_norm), torch.as_tensor(y_norm))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    model = DecisionSurrogate(x_np.shape[1], hidden_sizes=hidden_sizes, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        losses = []
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            history.append({"epoch": float(epoch), "loss": float(np.mean(losses))})

    bundle = SurrogateBundle(
        model=model,
        x_mean=torch.as_tensor(x_mean, dtype=torch.float32, device=device),
        x_std=torch.as_tensor(x_std, dtype=torch.float32, device=device),
        y_mean=torch.as_tensor(y_mean, dtype=torch.float32, device=device),
        y_std=torch.as_tensor(y_std, dtype=torch.float32, device=device),
        metadata={**(metadata or {}), "history": history},
    )
    save_surrogate(bundle, output_path)
    return bundle


def save_surrogate(bundle: SurrogateBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": bundle.model.state_dict(),
            "input_dim": int(bundle.x_mean.numel()),
            "x_mean": bundle.x_mean.detach().cpu().numpy(),
            "x_std": bundle.x_std.detach().cpu().numpy(),
            "y_mean": bundle.y_mean.detach().cpu().numpy(),
            "y_std": bundle.y_std.detach().cpu().numpy(),
            "metadata": bundle.metadata,
        },
        path,
    )
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as file:
        json.dump(bundle.metadata, file, ensure_ascii=True, indent=2)


def load_surrogate(path: Path, device: str | torch.device = "cpu") -> SurrogateBundle:
    device = torch.device(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model = DecisionSurrogate(int(payload["input_dim"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return SurrogateBundle(
        model=model,
        x_mean=torch.as_tensor(payload["x_mean"], dtype=torch.float32, device=device),
        x_std=torch.as_tensor(payload["x_std"], dtype=torch.float32, device=device),
        y_mean=torch.as_tensor(payload["y_mean"], dtype=torch.float32, device=device),
        y_std=torch.as_tensor(payload["y_std"], dtype=torch.float32, device=device),
        metadata=dict(payload.get("metadata", {})),
    )


def save_samples_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(np.asarray(x, dtype=np.float32))
    frame.insert(0, "sample_id", np.arange(len(frame)))
    frame["reward"] = np.asarray(y, dtype=np.float32)[:, 0]
    frame["cost"] = np.asarray(y, dtype=np.float32)[:, 1]
    frame.to_csv(path, index=False)
