# 1corr Decision-Focused TFT + SRRL

This directory adds a new pipeline without modifying the existing TFT code in `1using`.

Main entry:

```powershell
python 1corr\joint_train.py --rounds 3
```

The pipeline uses the existing `1using/train_tft1.py` checkpoint format, predicts the
96-point TFT horizon, selects the default SRRL time points `0,24,48,72`, converts them
to the normalized `wind/sun/load` profile consumed by `1using/3main.py`, then alternates:

1. train SRRL on the current TFT profile through `3main.py --profile-npz`;
2. evaluate local profile perturbations with the current SRRL policy;
3. fit a differentiable surrogate for reward/cost;
4. update TFT with supervised loss plus surrogate decision loss.

Quick profile-only smoke run:

```powershell
python 1corr\joint_train.py --dry-run
```

To run only the TFT/surrogate side with an existing SRRL checkpoint:

```powershell
python 1corr\joint_train.py --skip-srrl --init-srrl-checkpoint path\to\final_model.pt
```

