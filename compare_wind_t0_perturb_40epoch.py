from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from surrogate_gradient import (
    Profile,
    evaluate_profile,
    json_safe,
    load_actor,
    load_baseline_profile,
    load_module,
    resolve_path,
)


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DEFAULT_MAIN_SCRIPT = THIS_DIR / "3main.py"
DEFAULT_OUTPUT_DIR = THIS_DIR / "result" / "wind_t0_perturb_40epoch"


def save_profile_npz(path: Path, profile: Profile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, wind=profile.wind, sun=profile.sun, load=profile.load)


def latest_checkpoint(torch_dir: Path) -> Path | None:
    checkpoints = list(torch_dir.glob("epoch-*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.stem.split("-")[-1]))


def load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing training manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as file:
        return json.load(file)


def checkpoint_from_manifest(manifest: dict[str, Any]) -> Path:
    for key in ("final_model_path", "best_model_path"):
        value = manifest.get(key)
        if value:
            path = Path(value)
            if path.exists():
                return path
    log_dir = Path(manifest["log_dir"])
    checkpoint = latest_checkpoint(log_dir / "torch_save")
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found under {log_dir / 'torch_save'}")
    return checkpoint


def run_training(
    *,
    label: str,
    main_script: Path,
    run_dir: Path,
    profile_npz: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(main_script),
        "--output-dir",
        str(run_dir),
        "--total-steps",
        str(args.total_steps),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--batch-size",
        str(args.batch_size),
        "--cost-limit",
        str(args.cost_limit),
        "--gamma",
        str(args.gamma),
        "--cost-gamma",
        str(args.cost_gamma),
        "--save-model-freq",
        str(args.save_model_freq),
        "--seed",
        str(args.seed),
        "--profile-npz",
        str(profile_npz),
        "--skip-plot",
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
    )
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    stdout_path = run_dir / "train.out.log"
    stderr_path = run_dir / "train.err.log"
    started = time.time()
    print(f"[train] {label}: starting -> {run_dir}")
    with open(stdout_path, "w", encoding="utf-8") as stdout_file, open(
        stderr_path,
        "w",
        encoding="utf-8",
    ) as stderr_file:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    elapsed_sec = time.time() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Training failed for {label}. See {stdout_path} and {stderr_path}."
        )

    manifest = load_manifest(run_dir)
    checkpoint = checkpoint_from_manifest(manifest)
    manifest["checkpoint_for_eval"] = str(checkpoint)
    manifest["elapsed_sec"] = elapsed_sec
    print(f"[train] {label}: done in {elapsed_sec / 60.0:.2f} min")
    return manifest


def build_epoch_curve(progress_csv: Path) -> pd.DataFrame:
    progress = pd.read_csv(progress_csv)
    epoch_col = "Train/Epoch"
    reward_col = "Metrics/EpRet"
    cost_col = "Metrics/EpCost"
    if progress.empty:
        return pd.DataFrame(
            columns=[
                "epoch",
                "rows",
                "reward_mean",
                "reward_last",
                "cost_mean",
                "cost_last",
                "total_env_steps_last",
            ]
        )
    progress[epoch_col] = progress[epoch_col].astype(float).astype(int)
    grouped = progress.groupby(epoch_col, sort=True)
    rows = []
    for epoch, frame in grouped:
        rows.append(
            {
                "epoch": int(epoch) + 1,
                "rows": int(len(frame)),
                "reward_mean": float(frame[reward_col].astype(float).mean()),
                "reward_last": float(frame[reward_col].astype(float).iloc[-1]),
                "cost_mean": float(frame[cost_col].astype(float).mean()),
                "cost_last": float(frame[cost_col].astype(float).iloc[-1]),
                "total_env_steps_last": float(frame["TotalEnvSteps"].astype(float).iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    main_module: Any,
    original_profile: Profile,
    episodes: int,
    seed: int,
    device_name: str,
    show_env_spaces: bool,
) -> tuple[float, float]:
    device = torch.device(device_name)
    actor, normalizer = load_actor(checkpoint_path, device)
    current_profile = {
        "wind": original_profile.wind,
        "sun": original_profile.sun,
        "load": original_profile.load,
    }
    main_module.wind_sun_load_bus = lambda *_args, **_kwargs: (
        current_profile["wind"],
        current_profile["sun"],
        current_profile["load"],
    )
    previous_profile_npz = os.environ.pop("SRRL_PROFILE_NPZ", None)
    previous_analysis_dir = os.environ.pop("SRRL_ANALYSIS_DIR", None)
    try:
        if show_env_spaces:
            env = main_module.CustomExampleEnv("OptimalPowerFlowEnv-v0")
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                env = main_module.CustomExampleEnv("OptimalPowerFlowEnv-v0")
        try:
            return evaluate_profile(
                env=env,
                current_profile=current_profile,
                profile=original_profile,
                actor=actor,
                normalizer=normalizer,
                episodes=episodes,
                seed=seed,
                device=device,
                max_steps=int(getattr(main_module, "POINTS_PER_DAY", original_profile.wind.shape[0])),
            )
        finally:
            if hasattr(env, "close"):
                env.close()
    finally:
        if previous_profile_npz is not None:
            os.environ["SRRL_PROFILE_NPZ"] = previous_profile_npz
        if previous_analysis_dir is not None:
            os.environ["SRRL_ANALYSIS_DIR"] = previous_analysis_dir


def plot_training_curves(
    baseline_curve: pd.DataFrame,
    perturbed_curve: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        baseline_curve["epoch"],
        baseline_curve["reward_mean"],
        label="original train",
        linewidth=1.8,
    )
    axes[0].plot(
        perturbed_curve["epoch"],
        perturbed_curve["reward_mean"],
        label="wind_t0 perturbed train",
        linewidth=1.8,
    )
    axes[0].set_ylabel("Mean train reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        baseline_curve["epoch"],
        baseline_curve["cost_mean"],
        label="original train",
        linewidth=1.8,
    )
    axes[1].plot(
        perturbed_curve["epoch"],
        perturbed_curve["cost_mean"],
        label="wind_t0 perturbed train",
        linewidth=1.8,
    )
    axes[1].set_xlabel("Training epoch")
    axes[1].set_ylabel("Mean train cost")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train 3main SRRL on original wind4 data and a tiny wind t0 perturbation, "
            "then evaluate both final models on the original unperturbed profile."
        )
    )
    parser.add_argument("--main-script", type=Path, default=DEFAULT_MAIN_SCRIPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wind-file", type=Path, default=None)
    parser.add_argument("--sun-file", type=Path, default=None)
    parser.add_argument("--load-file", type=Path, default=None)
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--wind-t0-delta-mw", type=float, default=-1e-6)
    parser.add_argument("--total-steps", type=int, default=4 * 20 * 40)
    parser.add_argument("--steps-per-epoch", type=int, default=4 * 20)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--cost-limit", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--cost-gamma", type=float, default=0.5)
    parser.add_argument("--save-model-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show-env-spaces", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    launch_cwd = Path.cwd()
    main_script = resolve_path(args.main_script, launch_cwd)
    output_dir = resolve_path(args.output_dir, launch_cwd)
    assert main_script is not None
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    main_module = load_module(main_script, "srrl_three_main_wind_t0_compare")
    import get_sun_wind_load_data_30 as data_module

    wind_file = resolve_path(args.wind_file, launch_cwd) or Path(main_module.WIND_CSV)
    sun_file = resolve_path(args.sun_file, launch_cwd) or Path(main_module.SUN_CSV)
    load_file = resolve_path(args.load_file, launch_cwd) or Path(main_module.LOAD_CSV)
    original_profile = load_baseline_profile(
        data_module,
        wind_file,
        sun_file,
        load_file,
        args.sheet_name,
    )
    perturbed_profile = Profile(
        wind=original_profile.wind.copy(),
        sun=original_profile.sun.copy(),
        load=original_profile.load.copy(),
    )
    perturbed_profile.wind[0] += args.wind_t0_delta_mw
    wind_cap = float(np.abs(np.asarray(data_module.wind_max, dtype=float))[0])
    perturbed_profile.wind[:] = np.clip(perturbed_profile.wind, 0.0, wind_cap)

    profiles_dir = output_dir / "profiles"
    original_profile_npz = profiles_dir / "original_profile.npz"
    perturbed_profile_npz = profiles_dir / "wind_t0_perturbed_profile.npz"
    save_profile_npz(original_profile_npz, original_profile)
    save_profile_npz(perturbed_profile_npz, perturbed_profile)

    original_wind_t0 = float(original_profile.wind[0])
    perturbed_wind_t0 = float(perturbed_profile.wind[0])
    actual_delta = perturbed_wind_t0 - original_wind_t0
    print(
        "wind t0 normalized MW: "
        f"original={original_wind_t0:.12g}, perturbed={perturbed_wind_t0:.12g}, "
        f"delta={actual_delta:.12g}"
    )

    baseline_manifest = run_training(
        label="original",
        main_script=main_script,
        run_dir=output_dir / "train_original",
        profile_npz=original_profile_npz,
        args=args,
    )
    perturbed_manifest = run_training(
        label="wind_t0_perturbed",
        main_script=main_script,
        run_dir=output_dir / "train_wind_t0_perturbed",
        profile_npz=perturbed_profile_npz,
        args=args,
    )

    baseline_checkpoint = Path(baseline_manifest["checkpoint_for_eval"])
    perturbed_checkpoint = Path(perturbed_manifest["checkpoint_for_eval"])
    baseline_eval_reward, baseline_eval_cost = evaluate_checkpoint(
        checkpoint_path=baseline_checkpoint,
        main_module=main_module,
        original_profile=original_profile,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        device_name=args.device,
        show_env_spaces=args.show_env_spaces,
    )
    perturbed_eval_reward, perturbed_eval_cost = evaluate_checkpoint(
        checkpoint_path=perturbed_checkpoint,
        main_module=main_module,
        original_profile=original_profile,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        device_name=args.device,
        show_env_spaces=args.show_env_spaces,
    )

    baseline_curve = build_epoch_curve(Path(baseline_manifest["progress_csv"]))
    perturbed_curve = build_epoch_curve(Path(perturbed_manifest["progress_csv"]))
    baseline_curve.to_csv(output_dir / "train_curve_original_by_epoch.csv", index=False)
    perturbed_curve.to_csv(output_dir / "train_curve_wind_t0_perturbed_by_epoch.csv", index=False)
    curve_compare = baseline_curve.merge(
        perturbed_curve,
        on="epoch",
        suffixes=("_original", "_perturbed"),
    )
    curve_compare["delta_reward_mean"] = (
        curve_compare["reward_mean_perturbed"] - curve_compare["reward_mean_original"]
    )
    curve_compare["delta_cost_mean"] = (
        curve_compare["cost_mean_perturbed"] - curve_compare["cost_mean_original"]
    )
    curve_compare.to_csv(output_dir / "train_curve_delta_by_epoch.csv", index=False)
    plot_training_curves(
        baseline_curve,
        perturbed_curve,
        output_dir / "training_curve_original_vs_wind_t0_perturbed.png",
    )

    summary_rows = [
        {
            "model": "trained_on_original",
            "eval_profile": "original",
            "eval_episodes": args.eval_episodes,
            "eval_seed": args.eval_seed,
            "eval_reward": baseline_eval_reward,
            "eval_cost": baseline_eval_cost,
            "train_final_ep_ret": baseline_manifest["metrics"]["ep_ret"],
            "train_final_ep_cost": baseline_manifest["metrics"]["ep_cost"],
            "checkpoint": str(baseline_checkpoint),
        },
        {
            "model": "trained_on_wind_t0_perturbed",
            "eval_profile": "original",
            "eval_episodes": args.eval_episodes,
            "eval_seed": args.eval_seed,
            "eval_reward": perturbed_eval_reward,
            "eval_cost": perturbed_eval_cost,
            "train_final_ep_ret": perturbed_manifest["metrics"]["ep_ret"],
            "train_final_ep_cost": perturbed_manifest["metrics"]["ep_cost"],
            "checkpoint": str(perturbed_checkpoint),
        },
    ]
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame["delta_vs_original_eval_reward"] = (
        summary_frame["eval_reward"] - baseline_eval_reward
    )
    summary_frame["delta_vs_original_eval_cost"] = (
        summary_frame["eval_cost"] - baseline_eval_cost
    )
    summary_frame.to_csv(output_dir / "evaluation_summary.csv", index=False)

    summary = {
        "meaning": (
            "Both SRRL models are trained for the same 3main 40-epoch setting. "
            "The perturbed model is trained with only wind[0] changed, then both final "
            "models are evaluated on the original unperturbed profile."
        ),
        "main_script": str(main_script),
        "output_dir": str(output_dir),
        "wind_file": str(wind_file),
        "sun_file": str(sun_file),
        "load_file": str(load_file),
        "wind_t0_delta_mw_requested": args.wind_t0_delta_mw,
        "wind_t0_delta_mw_actual": actual_delta,
        "original_profile": {
            "wind": original_profile.wind.tolist(),
            "sun": original_profile.sun.tolist(),
            "load": original_profile.load.tolist(),
        },
        "perturbed_profile": {
            "wind": perturbed_profile.wind.tolist(),
            "sun": perturbed_profile.sun.tolist(),
            "load": perturbed_profile.load.tolist(),
        },
        "training_args": {
            "total_steps": args.total_steps,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "cost_limit": args.cost_limit,
            "gamma": args.gamma,
            "cost_gamma": args.cost_gamma,
            "save_model_freq": args.save_model_freq,
            "seed": args.seed,
        },
        "evaluation_args": {
            "eval_episodes": args.eval_episodes,
            "eval_seed": args.eval_seed,
            "device": args.device,
        },
        "trained_on_original": baseline_manifest,
        "trained_on_wind_t0_perturbed": perturbed_manifest,
        "evaluation_on_original": {
            "trained_on_original": {
                "reward": baseline_eval_reward,
                "cost": baseline_eval_cost,
            },
            "trained_on_wind_t0_perturbed": {
                "reward": perturbed_eval_reward,
                "cost": perturbed_eval_cost,
                "delta_reward_vs_original_model": perturbed_eval_reward - baseline_eval_reward,
                "delta_cost_vs_original_model": perturbed_eval_cost - baseline_eval_cost,
            },
        },
        "training_curve_outputs": {
            "original_by_epoch_csv": str(output_dir / "train_curve_original_by_epoch.csv"),
            "perturbed_by_epoch_csv": str(output_dir / "train_curve_wind_t0_perturbed_by_epoch.csv"),
            "delta_by_epoch_csv": str(output_dir / "train_curve_delta_by_epoch.csv"),
            "plot": str(output_dir / "training_curve_original_vs_wind_t0_perturbed.png"),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(json_safe(summary), file, ensure_ascii=True, indent=2, allow_nan=False)

    print("\nComparison complete.")
    print(f"  original eval reward:  {baseline_eval_reward:.9g}")
    print(f"  original eval cost:    {baseline_eval_cost:.9g}")
    print(f"  perturbed eval reward: {perturbed_eval_reward:.9g}")
    print(f"  perturbed eval cost:   {perturbed_eval_cost:.9g}")
    print(f"  delta reward:          {perturbed_eval_reward - baseline_eval_reward:.9g}")
    print(f"  delta cost:            {perturbed_eval_cost - baseline_eval_cost:.9g}")
    print(f"  outputs:               {output_dir}")


if __name__ == "__main__":
    main()
