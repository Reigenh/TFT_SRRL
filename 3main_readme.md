# 3main SRRL 风机首点扰动对比实验记录

本文档记录 `3main.py` 的 4 组风机首点小幅扰动训练实验。实验目标是在不修改原始风机数据集的前提下，考察 SRRL 电网优化运行脚本对风机数据微小扰动的训练结果差异。

## 实验说明

- 主训练脚本：`R:\A_code\SRRL\1robust\1using\3main.py`
- 批量实验脚本：`R:\A_code\SRRL\1robust\1using\run_wind_t0_percent_perturb_40epoch.py`
- 原始数据：
  - 风机：`R:\A_code\SRRL\1robust\1using\choseddata\wind4.csv`
  - 光伏：`R:\A_code\SRRL\1robust\1using\choseddata\sun4.csv`
  - 负荷：`R:\A_code\SRRL\1robust\1using\choseddata\load4.csv`
- 数据处理方式：先按 `3main.py` / `get_sun_wind_load_data_30.py` 原逻辑读取并归一化风光荷数据，然后只对 `wind[0]` 进行百分比扰动，保存为临时 `.npz` profile，通过 `--profile-npz` 传给 `3main.py` 训练。
- 原始 CSV 数据未被修改。
- 原始 `wind[0] = 25.0`。

## 训练参数

| 参数 | 值 |
|---|---:|
| algorithm | CPO |
| env | `OptimalPowerFlowEnv-v0` |
| total_steps | 3200 |
| steps_per_epoch | 80 |
| epochs | 40 |
| batch_size | 40 |
| cost_limit | 10.0 |
| gamma | 0.5 |
| cost_gamma | 0.5 |
| seed | 0 |
| save_model_freq | 10 |

## 输出目录

本次 4 组扰动实验统一输出到：

```text
R:\A_code\SRRL\1robust\1using\result\wind_t0_percent_perturb_40epoch_20260512_231408
```

关键汇总文件：

- `summary.json`
- `training_summary.csv`
- `training_curve_original_vs_percent_perturbations.png`
- `training_curve_delta_vs_original_percent_perturbations.png`
- `training_curve_original_vs_percent_perturbations_wide.csv`
- `training_curve_original_vs_percent_perturbations_long.csv`

## 不扰动基线模型

用于训练过程对比的不扰动模型来自之前同一 40 epoch 设置的训练：

```text
R:\A_code\SRRL\1robust\1using\result\wind_t0_perturb_40epoch_20260512_194930\train_original
```

| 项目 | 路径或结果 |
|---|---|
| final_model | `R:\A_code\SRRL\1robust\1using\result\wind_t0_perturb_40epoch_20260512_194930\train_original\models\final_model.pt` |
| best_model | `R:\A_code\SRRL\1robust\1using\result\wind_t0_perturb_40epoch_20260512_194930\train_original\models\best_model.pt` |
| progress.csv | `R:\A_code\SRRL\1robust\1using\result\wind_t0_perturb_40epoch_20260512_194930\train_original\logs\CPO-{OptimalPowerFlowEnv-v0}\seed-000-2026-05-12-19-49-41\progress.csv` |
| final train EpRet | -1870.74462890625 |
| final train EpCost | 36.93170928955078 |

## 4 组扰动实验结果

下表中的 `EpRet` 和 `EpCost` 为训练完成后 `run_manifest.json` / `training_summary.csv` 记录的最终训练指标，不是另行手工调度评估指标。

| 实验标签 | 扰动比例 | wind[0] | delta MW | final train EpRet | final train EpCost | 完成时间 |
|---|---:|---:|---:|---:|---:|---|
| `up_0p1pct` | +0.1% | 25.025 | +0.025 | -1869.7596435546875 | 30.99193572998047 | 2026-05-12 23:50:16 |
| `down_0p1pct` | -0.1% | 24.975 | -0.025 | -1864.4649658203125 | 20.19711685180664 | 2026-05-13 00:23:27 |
| `up_0p2pct` | +0.2% | 25.05 | +0.05 | -1874.771240234375 | 17.201589584350586 | 2026-05-13 00:47:48 |
| `down_0p2pct` | -0.2% | 24.95 | -0.05 | -1855.7353515625 | 13.08935832977295 | 2026-05-13 01:12:17 |

## 模型保存路径

| 实验标签 | final_model.pt | best_model.pt |
|---|---|---|
| `up_0p1pct` | `R:\A_code\SRRL\1robust\1using\result\wind_t0_percent_perturb_40epoch_20260512_231408\train_wind_t0_up_0p1pct\models\final_model.pt` | `R:\A_code\SRRL\1robust\1using\result\wind_t0_percent_perturb_40epoch_20260512_231408\train_wind_t0_up_0p1pct\models\best_model.pt` |
| `down_0p1pct` | `R:\A_code\SRRL\1robust\1using\result\wind_t0_percent_perturb_40epoch_20260512_231408\train_wind_t0_down_0p1pct\models\final_model.pt` | `R:\A_code\SRRL\1robust\1using\result\wind_t0_percent_perturb_40epoch_20260512_231408\train_wind_t0_down_0p1pct\models\best_model.pt` |
| `up_0p2pct` | `R:\A_code\SRRL\1robust\1using\result\wind_t0_percent_perturb_40epoch_20260512_231408\train_wind_t0_up_0p2pct\models\final_model.pt` | `R:\A_code\SRRL\1robust\1using\result\wind_t0_percent_perturb_40epoch_20260512_231408\train_wind_t0_up_0p2pct\models\best_model.pt` |
| `down_0p2pct` | `R:\A_code\SRRL\1robust\1using\result\wind_t0_percent_perturb_40epoch_20260512_231408\train_wind_t0_down_0p2pct\models\final_model.pt` | `R:\A_code\SRRL\1robust\1using\result\wind_t0_percent_perturb_40epoch_20260512_231408\train_wind_t0_down_0p2pct\models\best_model.pt` |

## 训练曲线对比

已生成两张训练过程对比图：

- `training_curve_original_vs_percent_perturbations.png`：不扰动基线与 4 个扰动模型的 `reward_mean` / `cost_mean` 训练曲线。
- `training_curve_delta_vs_original_percent_perturbations.png`：4 个扰动模型相对不扰动基线的逐 epoch 差值曲线。

第 40 epoch 的均值对比：

| 模型 | reward_mean | cost_mean | delta reward vs baseline | delta cost vs baseline |
|---|---:|---:|---:|---:|
| 不扰动 baseline | -1871.1236572265625 | 36.1960334777832 | 0.0 | 0.0 |
| `up_0p1pct` | -1870.5725911458333 | 29.91603660583496 | +0.5510660807292425 | -6.279996871948242 |
| `down_0p1pct` | -1865.8260091145837 | 20.133029301961265 | +5.297648111978788 | -16.063004175821938 |
| `up_0p2pct` | -1875.3689371744792 | 17.593828201293945 | -4.2452799479167425 | -18.602205276489258 |
| `down_0p2pct` | -1856.7165934244792 | 13.01150925954183 | +14.407063802083258 | -23.18452421824137 |

## 注意事项

- 本文档记录的是训练过程和训练日志指标，可追溯来源为 `summary.json`、`training_summary.csv`、各 run 的 `run_manifest.json` 与训练曲线 CSV。
- 之前手工编写的调度评估脚本曾出现 cost 口径不一致的问题，相关结果已撤回，不写入本实验结论。
- 若需要报告“最终模型在原始数据集上的真实调度 reward/cost”，应先实现并确认一个严格复现 OmniSafe 评估 wrapper、动作缩放、观测归一化和 cost 口径的评估脚本，再单独记录。
