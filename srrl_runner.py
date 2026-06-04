from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .tft_bridge import flat_to_arrays
except ImportError:  # pragma: no cover
    from tft_bridge import flat_to_arrays

ROOT = Path(__file__).resolve().parents[1]
USING_DIR = ROOT / "1using"
if str(USING_DIR) not in sys.path:
    sys.path.insert(0, str(USING_DIR))

from surrogate_gradient import evaluate_profile, load_actor, load_module  # type: ignore  # noqa: E402


@dataclass(frozen=True)
class Profile:
    wind: np.ndarray
    sun: np.ndarray
    load: np.ndarray


def latest_checkpoint(torch_dir: Path) -> Path | None:
    checkpoints = list(torch_dir.glob("epoch-*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.stem.split("-")[-1]))


def read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"SRRL run did not produce {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_checkpoint_from_manifest(manifest: dict[str, Any]) -> Path:
    for key in ("final_model_path", "best_model_path"):
        value = manifest.get(key)
        if value and Path(value).exists():
            return Path(value)
    log_dir = Path(manifest["log_dir"])
    checkpoint = latest_checkpoint(log_dir / "torch_save")
    if checkpoint is None:
        raise FileNotFoundError(f"No SRRL checkpoint found under {log_dir / 'torch_save'}")
    return checkpoint


def run_srrl_training(
    main_script: Path,
    output_dir: Path,
    profile_npz: Path,
    total_steps: int,
    steps_per_epoch: int,
    batch_size: int,
    cost_limit: float,
    gamma: float,
    cost_gamma: float,
    save_model_freq: int,
    seed: int,
    init_model_path: Path | None = None,
    skip_plot: bool = True,
) -> tuple[Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(main_script),
        "--output-dir",
        str(output_dir),
        "--total-steps",
        str(total_steps),
        "--steps-per-epoch",
        str(steps_per_epoch),
        "--batch-size",
        str(batch_size),
        "--cost-limit",
        str(cost_limit),
        "--gamma",
        str(gamma),
        "--cost-gamma",
        str(cost_gamma),
        "--save-model-freq",
        str(save_model_freq),
        "--seed",
        str(seed),
        "--profile-npz",
        str(profile_npz),
    ]
    if skip_plot:
        command.append("--skip-plot")
    if init_model_path is not None:
        command.extend(["--init-model-path", str(init_model_path)])

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else f"{ROOT}{os.pathsep}{existing_pythonpath}"
    stdout_path = output_dir / "train.out.log"
    stderr_path = output_dir / "train.err.log"
    with open(stdout_path, "w", encoding="utf-8") as stdout_file, open(stderr_path, "w", encoding="utf-8") as stderr_file:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"SRRL training failed in {output_dir}. See {stdout_path} and {stderr_path}.")
    manifest = read_manifest(output_dir)
    return resolve_checkpoint_from_manifest(manifest), manifest


def flat_rows_to_profiles(rows: np.ndarray) -> list[Profile]:
    if rows.shape[1] % 22 != 0:
        raise ValueError(f"Flat profile length {rows.shape[1]} is not compatible with wind/sun/load layout.")
    time_count = rows.shape[1] // 22
    profiles = []
    for row in rows:
        wind, sun, load = flat_to_arrays(row, time_count=time_count)
        profiles.append(Profile(wind=wind, sun=sun, load=load))
    return profiles


def evaluate_flat_profiles(
    main_script: Path,
    checkpoint_path: Path,
    flat_profiles: np.ndarray,
    episodes: int,
    seed: int,
    device: str = "cpu",
    unclip_cost: bool = False,
    show_env_spaces: bool = False,
) -> np.ndarray:
    for env_name in ("SRRL_PROFILE_NPZ", "SRRL_WIND_CSV", "SRRL_SUN_CSV", "SRRL_LOAD_CSV"):
        os.environ.pop(env_name, None)

    main_module = load_module(main_script, "srrl_three_main_corr_eval")
    profiles = flat_rows_to_profiles(flat_profiles)
    actor, normalizer = load_actor(checkpoint_path, torch.device(device))
    current_profile = {
        "wind": profiles[0].wind,
        "sun": profiles[0].sun,
        "load": profiles[0].load,
    }
    main_module.wind_sun_load_bus = lambda *_args, **_kwargs: (
        current_profile["wind"],
        current_profile["sun"],
        current_profile["load"],
    )

    if show_env_spaces:
        env = main_module.CustomExampleEnv("OptimalPowerFlowEnv-v0")
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            env = main_module.CustomExampleEnv("OptimalPowerFlowEnv-v0")
    if unclip_cost:
        low, _ = getattr(env, "costs_clipping", (0, np.inf))
        env.costs_clipping = (low, np.inf)

    max_steps = int(getattr(main_module, "POINTS_PER_DAY", profiles[0].wind.shape[0]))
    rows: list[list[float]] = []
    try:
        for idx, profile in enumerate(profiles):
            reward, cost = evaluate_profile(
                env=env,
                current_profile=current_profile,
                profile=profile,
                actor=actor,
                normalizer=normalizer,
                episodes=episodes,
                seed=seed + idx * max(episodes, 1),
                device=torch.device(device),
                max_steps=max_steps,
            )
            rows.append([float(reward), float(cost)])
    finally:
        if hasattr(env, "close"):
            env.close()
    return np.asarray(rows, dtype=np.float32)
