from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .srrl_runner import evaluate_flat_profiles, run_srrl_training
    from .surrogate_model import (
        SurrogateBundle,
        load_surrogate,
        make_local_profile_samples,
        save_samples_csv,
        train_surrogate,
    )
    from .tft_bridge import (
        TARGET_ORDER,
        all_tft_parameters,
        export_forecast_tables,
        load_decision_contexts,
        load_supervised_tensors,
        load_tft_states,
        parse_time_indices,
        predict_raw96,
        save_profile_npz,
        save_tft_states,
        selected_raw_to_profile_flat,
    )
except ImportError:  # pragma: no cover
    from srrl_runner import evaluate_flat_profiles, run_srrl_training
    from surrogate_model import (
        SurrogateBundle,
        load_surrogate,
        make_local_profile_samples,
        save_samples_csv,
        train_surrogate,
    )
    from tft_bridge import (
        TARGET_ORDER,
        all_tft_parameters,
        export_forecast_tables,
        load_decision_contexts,
        load_supervised_tensors,
        load_tft_states,
        parse_time_indices,
        predict_raw96,
        save_profile_npz,
        save_tft_states,
        selected_raw_to_profile_flat,
    )


ROOT = Path(__file__).resolve().parents[1]
USING_DIR = ROOT / "1using"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decision-focused alternating training for existing TFT forecasters and 1using/3main SRRL."
    )
    parser.add_argument("--run-root", type=Path, default=ROOT / "1corr" / "runs" / "dfl")
    parser.add_argument("--base-tft-dir", type=Path, default=USING_DIR / "tft1")
    parser.add_argument("--data-dir", type=Path, default=USING_DIR / "realdata")
    parser.add_argument("--main-script", type=Path, default=USING_DIR / "3main.py")
    parser.add_argument("--init-srrl-checkpoint", type=Path, default=None)
    parser.add_argument("--init-surrogate", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--decision-day-index", type=int, default=None)
    parser.add_argument("--time-indices", default="0,24,48,72", help="0-based TFT forecast indices passed to 3main.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--tft-epochs", type=int, default=5)
    parser.add_argument("--tft-batch-size", type=int, default=32)
    parser.add_argument("--tft-lr", type=float, default=2e-5)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--decision-weight", type=float, default=0.2)
    parser.add_argument("--decision-cost-weight", type=float, default=1.0)
    parser.add_argument(
        "--decision-cost-target",
        type=float,
        default=None,
        help="Absolute SRRL cost target. Defaults to the surrogate sample mean for the current round.",
    )
    parser.add_argument("--forecast-anchor-weight", type=float, default=0.02)

    parser.add_argument("--srrl-total-steps", type=int, default=4 * 20 * 40)
    parser.add_argument("--srrl-steps-per-epoch", type=int, default=4 * 20)
    parser.add_argument("--srrl-batch-size", type=int, default=40)
    parser.add_argument("--srrl-cost-limit", type=float, default=10.0)
    parser.add_argument("--srrl-gamma", type=float, default=0.5)
    parser.add_argument("--srrl-cost-gamma", type=float, default=0.5)
    parser.add_argument("--srrl-save-model-freq", type=int, default=10)
    parser.add_argument("--skip-srrl", action="store_true", help="Only run TFT/surrogate steps; useful for smoke tests.")
    parser.add_argument(
        "--no-post-tft-srrl",
        action="store_true",
        help="Do not retrain SRRL immediately after each TFT update. The next round will train it instead.",
    )

    parser.add_argument("--surrogate-samples", type=int, default=32)
    parser.add_argument("--surrogate-radius", type=float, default=0.03)
    parser.add_argument("--surrogate-epochs", type=int, default=500)
    parser.add_argument("--surrogate-batch-size", type=int, default=32)
    parser.add_argument("--surrogate-lr", type=float, default=1e-3)
    parser.add_argument("--surrogate-episodes", type=int, default=1)
    parser.add_argument("--surrogate-central-diff", action="store_true")
    parser.add_argument("--surrogate-central-step", type=float, default=0.01)
    parser.add_argument("--unclip-eval-cost", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Generate the first TFT profile and exit.")
    return parser


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.floating, float)):
        value_float = float(value)
        return value_float if np.isfinite(value_float) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, ensure_ascii=True, indent=2)


def export_current_profile(
    states: dict[str, Any],
    contexts: dict[str, torch.Tensor],
    output_dir: Path,
    time_indices: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    for state in states.values():
        state.model.eval()
    with torch.no_grad():
        raw96 = predict_raw96(states, contexts)
        flat_tensor = selected_raw_to_profile_flat(raw96, time_indices=time_indices)
    export_forecast_tables(output_dir, raw96, flat_tensor, time_indices=time_indices)
    flat = flat_tensor.detach().cpu().numpy()[0].astype(np.float32)
    save_profile_npz(output_dir / "dispatch_profile.npz", flat, time_count=len(time_indices))
    for state in states.values():
        state.model.train()
    return flat, raw96


def make_supervised_loaders(
    supervised: dict[str, tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    seed: int,
) -> dict[str, DataLoader]:
    loaders = {}
    generator = torch.Generator()
    generator.manual_seed(seed)
    for target, (x, y) in supervised.items():
        loaders[target] = DataLoader(
            TensorDataset(x, y),
            batch_size=min(batch_size, x.shape[0]),
            shuffle=True,
            generator=generator,
        )
    return loaders


def train_tft_with_decision_loss(
    states: dict[str, Any],
    supervised: dict[str, tuple[torch.Tensor, torch.Tensor]],
    contexts: dict[str, torch.Tensor],
    surrogate: SurrogateBundle | None,
    args: argparse.Namespace,
    time_indices: tuple[int, ...],
    round_dir: Path,
) -> dict[str, Any]:
    if args.tft_epochs <= 0:
        return {"skipped": True}
    if surrogate is not None:
        surrogate.model.eval()
        for param in surrogate.model.parameters():
            param.requires_grad_(False)

    loaders = make_supervised_loaders(supervised, args.tft_batch_size, args.seed)
    iterator_by_target = {target: iter(loader) for target, loader in loaders.items()}
    steps_per_epoch = max(len(loader) for loader in loaders.values())
    optimizer = torch.optim.AdamW(all_tft_parameters(states), lr=args.tft_lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()

    with torch.no_grad():
        anchor_raw = predict_raw96(states, contexts)
        anchor_flat = selected_raw_to_profile_flat(anchor_raw, time_indices=time_indices).detach()

    history: list[dict[str, float]] = []
    for epoch in range(1, args.tft_epochs + 1):
        supervised_losses: list[torch.Tensor] = []
        decision_losses: list[torch.Tensor] = []
        for _ in range(steps_per_epoch):
            sup_loss = torch.zeros((), dtype=torch.float32, device=torch.device(args.device))
            for target in TARGET_ORDER:
                try:
                    xb, yb = next(iterator_by_target[target])
                except StopIteration:
                    iterator_by_target[target] = iter(loaders[target])
                    xb, yb = next(iterator_by_target[target])
                pred = states[target].model(xb)
                sup_loss = sup_loss + loss_fn(pred, yb)
            sup_loss = sup_loss / len(TARGET_ORDER)

            raw96 = predict_raw96(states, contexts)
            flat_profile = selected_raw_to_profile_flat(raw96, time_indices=time_indices)
            anchor_loss = torch.mean((flat_profile - anchor_flat) ** 2)
            if surrogate is None:
                decision_loss = anchor_loss
            else:
                pred_decision = surrogate.predict(flat_profile)
                reward_scale = torch.clamp(
                    torch.maximum(torch.abs(surrogate.y_mean[0]), surrogate.y_std[0]),
                    min=1.0,
                )
                cost_target = (
                    torch.as_tensor(args.decision_cost_target, dtype=pred_decision.dtype, device=pred_decision.device)
                    if args.decision_cost_target is not None
                    else surrogate.y_mean[1].to(dtype=pred_decision.dtype, device=pred_decision.device)
                )
                cost_scale = torch.clamp(
                    torch.maximum(
                        torch.abs(cost_target),
                        surrogate.y_std[1],
                    ),
                    min=1.0,
                )
                reward_term = -(pred_decision[:, 0] / reward_scale).mean()
                cost_term = torch.relu((pred_decision[:, 1] - cost_target) / cost_scale).pow(2).mean()
                decision_loss = reward_term + args.decision_cost_weight * cost_term

            loss = (
                args.supervised_weight * sup_loss
                + args.decision_weight * decision_loss
                + args.forecast_anchor_weight * anchor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(all_tft_parameters(states), 1.0)
            optimizer.step()
            supervised_losses.append(sup_loss.detach())
            decision_losses.append(decision_loss.detach())

        row = {
            "epoch": float(epoch),
            "supervised_loss": float(torch.stack(supervised_losses).mean().cpu()),
            "decision_loss": float(torch.stack(decision_losses).mean().cpu()),
        }
        history.append(row)
        print(
            f"  TFT epoch {epoch}/{args.tft_epochs}: "
            f"supervised={row['supervised_loss']:.6g}, decision={row['decision_loss']:.6g}",
            flush=True,
        )

    save_tft_states(states, round_dir / "tft", metadata={"round_dir": str(round_dir), "history": history})
    return {"history": history, "checkpoint_dir": str(round_dir / "tft")}


def fit_round_surrogate(
    round_dir: Path,
    base_flat: np.ndarray,
    srrl_checkpoint: Path,
    args: argparse.Namespace,
) -> SurrogateBundle:
    sample_x = make_local_profile_samples(
        base_flat,
        sample_count=args.surrogate_samples,
        radius=args.surrogate_radius,
        seed=args.seed,
        include_central=args.surrogate_central_diff,
        central_step=args.surrogate_central_step,
    )
    print(f"  Evaluating {len(sample_x)} local profiles for surrogate fitting.", flush=True)
    sample_y = evaluate_flat_profiles(
        main_script=args.main_script,
        checkpoint_path=srrl_checkpoint,
        flat_profiles=sample_x,
        episodes=args.surrogate_episodes,
        seed=args.seed,
        device=args.device,
        unclip_cost=args.unclip_eval_cost,
    )
    save_samples_csv(round_dir / "surrogate" / "samples.csv", sample_x, sample_y)
    bundle = train_surrogate(
        x=sample_x,
        y=sample_y,
        output_path=round_dir / "surrogate" / "surrogate.pt",
        epochs=args.surrogate_epochs,
        batch_size=args.surrogate_batch_size,
        lr=args.surrogate_lr,
        seed=args.seed,
        device=args.device,
        metadata={
            "srrl_checkpoint": str(srrl_checkpoint),
            "sample_count": int(len(sample_x)),
            "radius": float(args.surrogate_radius),
            "episodes": int(args.surrogate_episodes),
        },
    )
    return bundle


def main() -> None:
    args = build_parser().parse_args()
    args.run_root = args.run_root.resolve()
    args.base_tft_dir = args.base_tft_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.main_script = args.main_script.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    time_indices = parse_time_indices(args.time_indices)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    states = load_tft_states(args.base_tft_dir, device)
    supervised = load_supervised_tensors(states, args.data_dir, device)
    contexts = load_decision_contexts(states, args.data_dir, args.decision_day_index, device)
    surrogate = load_surrogate(args.init_surrogate, device) if args.init_surrogate else None
    current_srrl_checkpoint = args.init_srrl_checkpoint.resolve() if args.init_srrl_checkpoint else None
    current_srrl_matches_current_tft = False
    rounds: list[dict[str, Any]] = []

    first_profile, _ = export_current_profile(states, contexts, args.run_root / "initial_forecast", time_indices)
    if args.dry_run:
        save_json(
            args.run_root / "manifest.json",
            {
                "status": "dry_run",
                "base_tft_dir": args.base_tft_dir,
                "initial_profile_npz": args.run_root / "initial_forecast" / "dispatch_profile.npz",
                "time_indices": time_indices,
            },
        )
        print(f"Dry run complete: {args.run_root / 'initial_forecast' / 'dispatch_profile.npz'}", flush=True)
        return

    for round_idx in range(1, args.rounds + 1):
        round_dir = args.run_root / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"Round {round_idx}/{args.rounds}", flush=True)

        base_flat, _ = export_current_profile(states, contexts, round_dir / "forecast_before_srrl", time_indices)
        profile_npz = round_dir / "forecast_before_srrl" / "dispatch_profile.npz"

        srrl_before_manifest: dict[str, Any] | None = None
        srrl_after_manifest: dict[str, Any] | None = None
        srrl_reused_aligned_checkpoint = False
        if args.skip_srrl:
            if current_srrl_checkpoint is None:
                raise ValueError("--skip-srrl requires --init-srrl-checkpoint so the surrogate can evaluate a policy.")
            srrl_checkpoint = current_srrl_checkpoint
        else:
            if current_srrl_checkpoint is None or not current_srrl_matches_current_tft:
                print("  Training SRRL on current TFT profile.", flush=True)
                srrl_checkpoint, srrl_before_manifest = run_srrl_training(
                    main_script=args.main_script,
                    output_dir=round_dir / "srrl_before_tft",
                    profile_npz=profile_npz,
                    total_steps=args.srrl_total_steps,
                    steps_per_epoch=args.srrl_steps_per_epoch,
                    batch_size=args.srrl_batch_size,
                    cost_limit=args.srrl_cost_limit,
                    gamma=args.srrl_gamma,
                    cost_gamma=args.srrl_cost_gamma,
                    save_model_freq=args.srrl_save_model_freq,
                    seed=args.seed,
                    init_model_path=current_srrl_checkpoint,
                    skip_plot=True,
                )
                current_srrl_checkpoint = srrl_checkpoint
                current_srrl_matches_current_tft = True
            else:
                srrl_checkpoint = current_srrl_checkpoint
                srrl_reused_aligned_checkpoint = True

        surrogate = fit_round_surrogate(round_dir, base_flat, srrl_checkpoint, args)
        tft_summary = train_tft_with_decision_loss(
            states=states,
            supervised=supervised,
            contexts=contexts,
            surrogate=surrogate,
            args=args,
            time_indices=time_indices,
            round_dir=round_dir,
        )
        after_flat, _ = export_current_profile(states, contexts, round_dir / "forecast_after_tft", time_indices)
        after_profile_npz = round_dir / "forecast_after_tft" / "dispatch_profile.npz"
        srrl_after_checkpoint: Path | None = None
        if not args.skip_srrl and args.tft_epochs > 0 and not args.no_post_tft_srrl:
            print("  Training SRRL on updated TFT profile.", flush=True)
            srrl_after_checkpoint, srrl_after_manifest = run_srrl_training(
                main_script=args.main_script,
                output_dir=round_dir / "srrl_after_tft",
                profile_npz=after_profile_npz,
                total_steps=args.srrl_total_steps,
                steps_per_epoch=args.srrl_steps_per_epoch,
                batch_size=args.srrl_batch_size,
                cost_limit=args.srrl_cost_limit,
                gamma=args.srrl_gamma,
                cost_gamma=args.srrl_cost_gamma,
                save_model_freq=args.srrl_save_model_freq,
                seed=args.seed,
                init_model_path=current_srrl_checkpoint,
                skip_plot=True,
            )
            current_srrl_checkpoint = srrl_after_checkpoint
            current_srrl_matches_current_tft = True
        elif not args.skip_srrl:
            current_srrl_matches_current_tft = args.tft_epochs <= 0

        rounds.append(
            {
                "round": round_idx,
                "profile_before_srrl": str(profile_npz),
                "profile_after_tft": str(after_profile_npz),
                "srrl_checkpoint_before_tft": str(srrl_checkpoint),
                "srrl_checkpoint_after_tft": str(srrl_after_checkpoint) if srrl_after_checkpoint is not None else None,
                "srrl_checkpoint": str(current_srrl_checkpoint) if current_srrl_checkpoint is not None else str(srrl_checkpoint),
                "srrl_reused_aligned_checkpoint": srrl_reused_aligned_checkpoint,
                "srrl_manifest_before_tft": srrl_before_manifest,
                "srrl_manifest_after_tft": srrl_after_manifest,
                "surrogate": str(round_dir / "surrogate" / "surrogate.pt"),
                "tft": tft_summary,
                "profile_l2_change": float(np.linalg.norm(after_flat - base_flat)),
                "srrl_aligned_after_round": current_srrl_matches_current_tft,
            }
        )
        save_json(args.run_root / "manifest.json", {"rounds": rounds, "latest_srrl_checkpoint": current_srrl_checkpoint})

    save_tft_states(states, args.run_root / "final_tft", metadata={"rounds": rounds})
    final_profile, _ = export_current_profile(states, contexts, args.run_root / "final_forecast", time_indices)
    save_json(
        args.run_root / "manifest.json",
        {
            "status": "complete",
            "base_tft_dir": args.base_tft_dir,
            "final_tft_dir": args.run_root / "final_tft",
            "final_profile_npz": args.run_root / "final_forecast" / "dispatch_profile.npz",
            "latest_srrl_checkpoint": current_srrl_checkpoint,
            "time_indices": time_indices,
            "initial_profile_l2": float(np.linalg.norm(first_profile)),
            "final_profile_l2": float(np.linalg.norm(final_profile)),
            "rounds": rounds,
        },
    )
    print(f"DFL training complete: {args.run_root}", flush=True)


if __name__ == "__main__":
    main()
