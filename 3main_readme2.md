# 3main SRRL 第一个风机前四个时间点 ±1% 扰动实验

## 实验目的

在不修改原始风机 CSV 数据集的前提下，检查 `3main.py` 中 SRRL 电网优化运行训练对第一个风机前四个时间点出力微小扰动的响应。每个时间点分别进行 `+1%` 和 `-1%` 扰动，并为每个扰动 profile 完整训练一次模型。

## 实验实现

- 主训练脚本：`3main.py`。
- 批量实验脚本：`run_wind_first4_time_1pct_perturb_40epoch.py`。
- 数据集不改写：训练仍由 `3main.py` 读取 `wind4.csv` / `sun4.csv` / `load4.csv`；读取完成后在内存中复制风机 profile，并只对指定风机、指定时间点乘以 `1 + perturb_fraction`。
- 扰动位置：第一个风机，时间点 1-4；脚本内部对应 zero-based `wind[0]` 到 `wind[3]`。
- 扰动幅度：每个时间点分别 `+1%`、`-1%`，共 8 次完整训练。
- 每次训练独立输出 `run_manifest.json`、`models/final_model.pt`、`models/best_model.pt`、训练日志和按 epoch 汇总的训练曲线 CSV。

## 路径与状态

- 当前状态：`completed`
- 输出目录：`R:\A_code\SRRL\1robust\1using\result\wind_first4_time_1pct_40epoch`
- README：`R:\A_code\SRRL\1robust\1using\3main_readme2.md`
- 风机数据：`R:\A_code\SRRL\1robust\1using\choseddata\wind4.csv`
- 光伏数据：`R:\A_code\SRRL\1robust\1using\choseddata\sun4.csv`
- 负荷数据：`R:\A_code\SRRL\1robust\1using\choseddata\load4.csv`
- 原始 profile 备份：`R:\A_code\SRRL\1robust\1using\result\wind_first4_time_1pct_40epoch\profiles\original_profile.npz`

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

## 原始风机 profile

| 时间点 | zero-based index | 原始出力 |
|---:|---:|---:|
| 1 | 0 | 25 |
| 2 | 1 | 20.9659207555 |
| 3 | 2 | 17.3244713611 |
| 4 | 3 | 10.0415725724 |

## 8 组扰动训练

| 顺序 | 时间点 | 扰动 | 原始出力 | 扰动后出力 | delta | 状态 | final EpRet | final EpCost | final_model.pt |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1 | 1 | 1% | 25 | 25.25 | 0.25 | `completed` | -1861.0718 | 34.549835 | `R:\A_code\SRRL\1robust\1using\result\wind_first4_time_1pct_40epoch\train_wind_t01_up_1pct\models\final_model.pt` |
| 2 | 1 | -1% | 25 | 24.75 | -0.25 | `completed` | -1880.4542 | 47.567261 | `R:\A_code\SRRL\1robust\1using\result\wind_first4_time_1pct_40epoch\train_wind_t01_down_1pct\models\final_model.pt` |
| 3 | 2 | 1% | 20.9659207555 | 21.175579963 | 0.209659207555 | `completed` | -1873.7538 | 23.216475 | `R:\A_code\SRRL\1robust\1using\result\wind_first4_time_1pct_40epoch\train_wind_t02_up_1pct\models\final_model.pt` |
| 4 | 2 | -1% | 20.9659207555 | 20.7562615479 | -0.209659207555 | `completed` | -1867.038 | 27.354256 | `R:\A_code\SRRL\1robust\1using\result\wind_first4_time_1pct_40epoch\train_wind_t02_down_1pct\models\final_model.pt` |
| 5 | 3 | 1% | 17.3244713611 | 17.4977160747 | 0.173244713611 | `completed` | -1871.4158 | 35.114571 | `R:\A_code\SRRL\1robust\1using\result\wind_first4_time_1pct_40epoch\train_wind_t03_up_1pct\models\final_model.pt` |
| 6 | 3 | -1% | 17.3244713611 | 17.1512266475 | -0.173244713611 | `completed` | -1863.9747 | 26.011196 | `R:\A_code\SRRL\1robust\1using\result\wind_first4_time_1pct_40epoch\train_wind_t03_down_1pct\models\final_model.pt` |
| 7 | 4 | 1% | 10.0415725724 | 10.1419882981 | 0.100415725724 | `completed` | -1875.5298 | 63.064064 | `R:\A_code\SRRL\1robust\1using\result\wind_first4_time_1pct_40epoch\train_wind_t04_up_1pct\models\final_model.pt` |
| 8 | 4 | -1% | 10.0415725724 | 9.94115684664 | -0.100415725724 | `completed` | -1865.6016 | 22.2773 | `R:\A_code\SRRL\1robust\1using\result\wind_first4_time_1pct_40epoch\train_wind_t04_down_1pct\models\final_model.pt` |

## 复现实验命令

```powershell
cd R:\A_code\SRRL\1robust\1using
python .\run_wind_first4_time_1pct_perturb_40epoch.py --output-dir R:\A_code\SRRL\1robust\1using\result\wind_first4_time_1pct_40epoch --readme-path R:\A_code\SRRL\1robust\1using\3main_readme2.md
```

## 说明

- `profiles/` 中保存的 `.npz` 是每个扰动 profile 的审计备份；实际训练调用的是 `3main.py --wind-perturb-time-index ... --wind-perturb-fraction ...`，由 `3main.py` 在读取原始数据后做内存扰动。
- 若实验中断，使用相同 `--output-dir` 重新运行脚本；已存在且可找到模型 checkpoint 的 run 会被跳过，除非加 `--force-rerun`。
- `summary.json` 和本 README 会在每个 run 开始、完成后更新。

## 与未扰动 baseline 的训练过程对比

使用之前同一 40 epoch 设置下的未扰动模型作为 baseline：

```text
R:\A_code\SRRL\1robust\1using\result\wind_t0_perturb_40epoch_20260512_194930\train_original
```

baseline 也已核对为 `epoch 0-39`、`TotalEnvSteps = 3200`。

对比图和 CSV 已输出到：

```text
R:\A_code\SRRL\1robust\1using\result\wind_first4_time_1pct_40epoch\plots_vs_baseline
```

输出文件：

- `training_reward_cost_vs_baseline_all.png`：未扰动 baseline 与 8 组扰动的 reward/cost 总览曲线。
- `training_reward_cost_delta_vs_baseline_all.png`：8 组扰动相对 baseline 的 reward/cost 差值曲线。
- `training_reward_cost_vs_baseline_by_timepoint.png`：按时间点分组，分别比较 baseline、`+1%`、`-1%`。
- `training_curves_vs_baseline_long.csv`：所有训练曲线按 epoch 汇总后的长表。
- `training_curves_vs_baseline_delta_long.csv`：所有曲线相对 baseline 的逐 epoch 差值长表。
- `final_epoch_vs_baseline.csv`：第 40 个 epoch 的最终对比表。

第 40 个 epoch 的均值对比：

| 实验 | reward_mean | cost_mean | delta reward vs baseline | delta cost vs baseline |
|---|---:|---:|---:|---:|
| baseline_original | -1871.1237 | 36.1960 | 0.0000 | 0.0000 |
| `wind_t01_up_1pct` | -1861.9534 | 32.3855 | +9.1703 | -3.8105 |
| `wind_t01_down_1pct` | -1882.0104 | 47.3527 | -10.8868 | +11.1567 |
| `wind_t02_up_1pct` | -1873.5149 | 21.5776 | -2.3912 | -14.6185 |
| `wind_t02_down_1pct` | -1866.9399 | 27.3417 | +4.1838 | -8.8543 |
| `wind_t03_up_1pct` | -1872.5200 | 35.1702 | -1.3964 | -1.0258 |
| `wind_t03_down_1pct` | -1865.6024 | 24.5015 | +5.5213 | -11.6945 |
| `wind_t04_up_1pct` | -1875.1879 | 63.1303 | -4.0643 | +26.9343 |
| `wind_t04_down_1pct` | -1866.1689 | 22.4966 | +4.9548 | -13.6994 |
