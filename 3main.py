from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import omnisafe
import pandas as pd
import torch
from get_sun_wind_load_data_30 import wind_sun_load_bus
from gym_anm import ANMEnv
from gymnasium import spaces
from omnisafe.envs.core import CMDP, env_register

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = THIS_DIR / 'result' / 'train_30_version_2_1'
POINTS_PER_DAY = 4
WIND_CSV = THIS_DIR / 'choseddata' / 'wind4.csv'
SUN_CSV = THIS_DIR / 'choseddata' / 'sun4.csv'
LOAD_CSV = THIS_DIR / 'choseddata' / 'load4.csv'
WIND_PERTURB_DEVICE_ENV = 'SRRL_WIND_PERTURB_DEVICE_INDEX'
WIND_PERTURB_TIME_ENV = 'SRRL_WIND_PERTURB_TIME_INDEX'
WIND_PERTURB_FRACTION_ENV = 'SRRL_WIND_PERTURB_FRACTION'
network = {
    "baseMVA": 100,

    "bus" : np.array([[0, 0, 135, 1.0, 1.0], 
                [1, 1, 135, 1.1, 0.9], 
                [2, 1, 135, 1.1, 0.9], 
                [3, 1, 135, 1.05, 0.9], 
                [4, 1, 135, 1.1, 0.95],
                [5, 1, 135, 1.05, 0.9],
                [6, 1, 135, 1.1, 0.9],
                [7, 1, 135, 1.1, 0.9], 
                [8, 1, 135, 1.1, 0.9],
                [9, 1, 135, 1.1, 0.95],
                [10, 1, 135, 1.05, 0.95],
                [11, 1, 135, 1.1, 0.9],
                [12, 1, 135, 1.1, 0.9],
                [13, 1, 135, 1.05, 0.9],
                [14, 1, 135, 1.1, 0.95],
                [15, 1, 135, 1.05, 0.9],
                [16, 1, 135, 1.1, 0.9],
                [17, 1, 135, 1.05, 0.9],
                [18, 1, 135, 1.1, 0.9],
                [19, 1, 135, 1.05, 0.9],
                [20, 1, 135, 1.1, 0.9],
                [21, 1, 135, 1.1, 0.9],
                [22, 1, 135, 1.1, 0.9],
                [23, 1, 135, 1.05, 0.9],
                [24, 1, 135, 1.05, 0.9],
                [25, 1, 135, 1.1, 0.9],
                [26, 1, 135, 1.1, 0.9],
                [27, 1, 135, 1.1, 0.9],
                [28, 1, 135, 1.1, 0.9],
                [29, 1, 135, 1.1, 0.9]
               ]),


    "device": np.array(
        [
            [0, 0, 0, None, 50, -50, 50, -50, None, None, None, None, None, None, None],
            #与大电网连接
            [1, 1, -1, 0.2,  0,-25, None, None, None, None, None, None, None, None, None],
            [2, 2, -1, 0.5, 0,-5, None, None, None, None, None, None, None, None, None],
            [3, 3, -1, 0.2,  0,-10, None, None,  None, None, None, None, None, None,None],
            [4, 6, -1, 0.5, 0,-25, None, None,  None, None, None, None, None, None, None],
            [5, 7, -1, 0.5, 0, -30, None, None,  None, None, None, None, None, None, None],
            [6, 9, -1, 0.6, 0, -7, None, None,  None, None, None, None, None, None, None],
            [7, 11, -1, 0.5, 0,-15, None, None, None, None, None, None, None, None, None],
            [8, 13, -1, 0.2, 0,-8, None, None,  None, None, None, None, None, None, None],
            [9, 14, -1, 0.3, 0, -10, None, None,  None, None, None, None, None, None, None],
            [10, 15, -1, 0.2, 0,-5, None, None,  None, None, None, None, None, None, None],
            [11, 16, -1, 0.6, 0, -12, None, None,  None, None, None, None, None, None, None],
            [12, 17, -1, 0.2,  0,-4, None, None, None, None, None, None, None, None, None],
            [13, 18, -1, 0.5, 0,-10, None, None, None, None, None, None, None, None, None],
            [14, 19, -1, 0.2,  0,-3, None, None,  None, None, None, None, None, None,None],
            [15, 20, -1, 0.5, 0,-20, None, None,  None, None, None, None, None, None, None],
            [16, 22, -1, 0.2, 0, -4, None, None,  None, None, None, None, None, None, None],
            [17, 23, -1, 0.6, 0, -10, None, None,  None, None, None, None, None, None, None],
            [18, 25, -1, 0.5, 0,-4, None, None, None, None, None, None, None, None, None],
            [19, 28, -1, 0.2, 0,-3, None, None,  None, None, None, None, None, None, None],
            [20, 29, -1, 0.3, 0, -14, None, None,  None, None, None, None, None, None, None],
            #负荷
            [21, 1, 1, None, 50, 0, 30, -20, None, None, None, None, None, None, None],
            [22, 2, 1, None, 50, 0, 20, -20, None, None, None, None, None, None, None],
            [23, 21, 1, None, 50, 0, 32.5, -15, None, None, None, None, None, None, None],
            [24, 26, 1, None, 55, 0, 28, -15, None, None, None, None, None, None, None],
            [25, 22, 1, None, 30, 0, 20, -10, None, None, None, None, None, None, None],
            [26, 12, 1, None, 40, 0, 24.7, -15, None, None, None, None, None, None, None],
            #发电机
            [27, 17, 2, None, 30, 0, 20, -20, None, None, None, None, None, None, None],
            [28, 27, 2, None, 40, 0, 20, -20, None, None, None, None, None, None, None],
            #风电或光伏
            [29, 5, 3, None, 25, -20, 20, -15, None, None, None, None, 100, 0, 0.9],
            [30, 9, 3, None, 20, -20, 20, -20, None, None, None, None, 100, 0, 0.9],
            #储能
        ]
    ),
        "branch": np.array([
        [0, 1, 0.02, 0.06, 0.03, 50, 1, 0],
        [0, 2, 0.05, 0.19, 0.02, 50, 1, 0],
        [1, 3, 0.06, 0.17, 0.02, 50, 1, 0],
        [2, 3, 0.01, 0.04, 0, 50, 1, 0],
        [1, 4, 0.05, 0.2, 0.02, 30, 1, 0],
        [1, 5, 0.06, 0.18, 0.02, 30, 1, 0],
        [3, 5, 0.01, 0.04, 0, 40, 1, 0],
        [4, 6, 0.05, 0.12, 0.01, 30, 1, 0],
        [5, 6, 0.03, 0.08, 0.01, 40, 1, 0],
        [5, 7, 0.01, 0.04, 0, 30, 1, 0],
        [5, 8, 0, 0.21, 0, 40, 1, 0],
        [5, 9, 0, 0.56, 0, 40, 1, 0],
        [8, 10, 0, 0.21, 0, 50, 1, 0],
        [8, 9, 0, 0.11, 0, 30, 1, 0],
        [3, 11, 0, 0.26, 0, 40, 1, 0],
        [11, 12, 0, 0.14, 0, 50, 1, 0],
        [11, 13, 0.12, 0.26, 0, 40, 1, 0],
        [11, 14, 0.07, 0.13, 0, 30, 1, 0],
        [11, 15, 0.09, 0.2, 0, 50, 1, 0],
        [13, 14, 0.22, 0.2, 0, 60, 1, 0],
        [15, 16, 0.08, 0.19, 0, 30, 1, 0],
        [14, 17, 0.11, 0.22, 0, 40, 1, 0],
        [17, 18, 0.06, 0.13, 0, 50, 1, 0],
        [18, 19, 0.03, 0.07, 0, 50, 1, 0],
        [9, 19, 0.09, 0.21, 0, 30, 1, 0],
        [9, 16, 0.03, 0.08, 0, 40, 1, 0],
        [9, 20, 0.03, 0.07, 0, 50, 1, 0],
        [9, 21, 0.07, 0.15, 0, 50, 1, 0],
        [20, 21, 0.01, 0.02, 0, 50, 1, 0],
        [14, 22, 0.1, 0.2, 0, 50, 1, 0],
        [21, 23, 0.12, 0.18, 0, 50, 1, 0],
        [22, 23, 0.13, 0.27, 0, 50, 1, 0],
        [23, 24, 0.19, 0.33, 0, 50, 1, 0],
        [24, 25, 0.25, 0.38, 0, 50, 1, 0],
        [24, 26, 0.11, 0.21, 0, 50, 1, 0],
        [27, 26, 0, 0.4, 0, 50, 1, 0],
        [26, 28, 0.22, 0.42, 0, 50, 1, 0],
        [26, 29, 0.32, 0.6, 0, 50, 1, 0],
        [28, 29, 0.24, 0.45, 0, 50, 1, 0],
        [7, 27, 0.06, 0.2, 0.02, 30, 1, 0],
        [5, 27, 0.02, 0.06, 0.01, 40, 1, 0]
        ])

}
gencost = [
    [21, 0.02, 2, 0],
    [22, 0.0175, 1.75, 0],
    [23, 0.0625, 1, 0],
    [24, 0.00834, 3.25, 0],
    [25, 0.025, 3, 0],
    [26, 0.025, 3, 0]
]

k_wind_sun = 2.5
k_pcc = 4
p_pcc_max = 50
price_ess = 3.5
reward_min = -30000
p_delta = [12, 12, 12, 15, 8, 10]


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def apply_runtime_wind_perturbation(wind_data: np.ndarray) -> np.ndarray:
    time_index_value = os.getenv(WIND_PERTURB_TIME_ENV)
    fraction_value = os.getenv(WIND_PERTURB_FRACTION_ENV)
    if time_index_value is None and fraction_value is None:
        return np.asarray(wind_data, dtype=float)
    if time_index_value is None or fraction_value is None:
        raise ValueError(
            f'{WIND_PERTURB_TIME_ENV} and {WIND_PERTURB_FRACTION_ENV} must be set together.'
        )

    device_index = int(os.getenv(WIND_PERTURB_DEVICE_ENV, '0'))
    time_index = int(time_index_value)
    fraction = float(fraction_value)
    perturbed = np.asarray(wind_data, dtype=float).copy()

    if perturbed.ndim == 1:
        if device_index != 0:
            raise IndexError('1-D wind profile only supports device index 0.')
        if time_index < 0 or time_index >= perturbed.shape[0]:
            raise IndexError(
                f'Wind time index {time_index} is out of range for shape {perturbed.shape}.'
            )
        perturbed[time_index] *= 1.0 + fraction
        return perturbed

    if perturbed.ndim >= 2:
        if device_index < 0 or device_index >= perturbed.shape[0]:
            raise IndexError(
                f'Wind device index {device_index} is out of range for shape {perturbed.shape}.'
            )
        if time_index < 0 or time_index >= perturbed.shape[1]:
            raise IndexError(
                f'Wind time index {time_index} is out of range for shape {perturbed.shape}.'
            )
        perturbed[device_index, time_index] *= 1.0 + fraction
        return perturbed

    raise ValueError(f'Unsupported wind profile shape: {perturbed.shape}.')


def load_dispatch_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profile_npz = os.getenv('SRRL_PROFILE_NPZ')
    if profile_npz:
        profile = np.load(profile_npz)
        return (
            apply_runtime_wind_perturbation(np.asarray(profile['wind'], dtype=float)),
            np.asarray(profile['sun'], dtype=float),
            np.asarray(profile['load'], dtype=float),
        )

    wind_data, sun_data, load_data = wind_sun_load_bus(
        str(_env_path('SRRL_WIND_CSV', WIND_CSV)),
        str(_env_path('SRRL_SUN_CSV', SUN_CSV)),
        str(_env_path('SRRL_LOAD_CSV', LOAD_CSV)),
    )
    return apply_runtime_wind_perturbation(wind_data), sun_data, load_data
_count = 0  # 初始化步数计数器

@env_register  # 使用装饰器自动注册环境
class CustomExampleEnv(ANMEnv, CMDP):
    _support_envs: ClassVar[list[str]] = ['OptimalPowerFlowEnv-v0']
    need_auto_reset_wrapper = True  # 声明在达到终止条件时是否自动重置环境
    need_time_limit_wrapper = True  # 声明是否有时间限制

    def __init__(self, env_id: str, **kwargs: dict[str, Any]) -> None:
        self._num_envs = 1
        observation = "state"  
        K = 1  
        delta_t = 24 / POINTS_PER_DAY
        gamma = 0.995  
        lamb = 0  
        aux_bounds = np.array([[0, POINTS_PER_DAY - 1]])  
        costs_clipping = [0, 5]  
        seed = None  

        super().__init__(network, p_delta, _count, gencost, price_ess, k_wind_sun, k_pcc, p_pcc_max, reward_min, observation, K, delta_t, gamma, lamb, aux_bounds, costs_clipping, seed)

        action_low_float32 = self.action_space_anm.low.astype(np.float32)
        action_high_float32 = self.action_space_anm.high.astype(np.float32)
        self._action_space = spaces.Box(low=np.array(action_low_float32), high=np.array(action_high_float32), dtype=np.float32)

        obs_low_float32 = self.observation_space_anm.low.astype(np.float32)
        obs_low_list = obs_low_float32.tolist()

        # p_load_for, q_load_for, p_for, p_gen_old, soc, t
        p_traditional_gen_low = obs_low_list[self.simulator.N_load + 1: 1 + self.simulator.N_load + self.simulator.N_non_slack_gen - self.simulator.N_gen_rer]
        soc_low = obs_low_list[self.simulator.N_device * 2: self.simulator.N_des + self.simulator.N_device * 2]
        aux_low = obs_low_list[-1]
        P_load_low = obs_low_list[1: self.simulator.N_load + 1]
        Q_load_low = obs_low_list[1 + self.simulator.N_device: self.simulator.N_device + self.simulator.N_load + 1]
        P_w_s_for_low = obs_low_list[1 + self.simulator.N_load + self.simulator.N_non_slack_gen - self.simulator.N_gen_rer: 1 + self.simulator.N_load + self.simulator.N_non_slack_gen]

        obs_low_list = P_load_low + Q_load_low + P_w_s_for_low + p_traditional_gen_low + soc_low + [aux_low]

        for i in range(4):
            obs_low_list.append(-1.0)
        obs_low_float32 = np.array(obs_low_list, dtype=np.float32)

        obs_high_float32 = self.observation_space_anm.high.astype(np.float32)
        obs_high_list = obs_high_float32.tolist()

        p_traditional_gen_high = obs_high_list[self.simulator.N_load + 1: 1 + self.simulator.N_load + self.simulator.N_non_slack_gen - self.simulator.N_gen_rer]
        soc_high = obs_high_list[self.simulator.N_device * 2: self.simulator.N_des + self.simulator.N_device * 2]
        aux_high = obs_high_list[-1]
        P_load_high = obs_high_list[1: self.simulator.N_load + 1]
        Q_load_high = obs_high_list[1 + self.simulator.N_device: self.simulator.N_device + self.simulator.N_load + 1]
        P_w_s_for_high = obs_high_list[1 + self.simulator.N_load + self.simulator.N_non_slack_gen - self.simulator.N_gen_rer: 1 + self.simulator.N_load + self.simulator.N_non_slack_gen]

        obs_high_list = P_load_high + Q_load_high + P_w_s_for_high + p_traditional_gen_high + soc_high + [aux_high]

        for i in range(4):
            obs_high_list.append(1.0)

        obs_high_float32 = np.array(obs_high_list, dtype=np.float32)
        self._observation_space = spaces.Box(low=np.array(obs_low_float32), high=np.array(obs_high_float32), dtype=np.float32)

        print("obs:", self._observation_space)
        print("act:", self._action_space)

    @property
    def max_episode_steps(self) -> int:
        return POINTS_PER_DAY

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        self.seed(seed)

    def close(self) -> None:
        super().close()

    def render(self) -> Any:
        return None

    def init_state(self):
        # p_load_for, q_load_for, p_for, p_gen_old, soc, t
        wind_data, sun_data, load_data = load_dispatch_profile()
        self.P_loads = load_data
        P_maxs = np.zeros((6, POINTS_PER_DAY))
        P_maxs[0, :] = 50
        P_maxs[1, :] = 50
        P_maxs[2, :] = 50
        P_maxs[3, :] = 55
        P_maxs[4, :] = 30
        P_maxs[5, :] = 40
        P_maxs = np.vstack((P_maxs, wind_data, sun_data))

        self.P_maxs = P_maxs
        n_dev, n_load, n_tra_gen,n_rew_gen, n_des = 31,20, 6,2, 2  # 定义设备数、发电机数和储能装置数

        obs = np.zeros(2 * n_load + n_rew_gen + n_tra_gen + n_des+1)

        t_0 = 0
        obs[-1] = t_0
        load_bus=[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        for i in range(len(load_data)):
            obs[i] = load_data[i][t_0]
            obs[i+n_load] = load_data[i][t_0]* self.simulator.devices[load_bus[i]].qp_ratio
        

        
        obs[0+2*n_load]=wind_data[0]
        obs[1+2*n_load]=sun_data[0]
        des_bus=[29,30]
        for i in range(n_des):
            obs[2*n_load+n_rew_gen+n_tra_gen+i] =self.np_random.uniform(self.simulator.devices[des_bus[i]].soc_min * 100, self.simulator.devices[des_bus[i]].soc_max * 100)
        

        return obs

    def next_vars(self, t):
        aux = int(t + 1) 

        vars = []
        for p_load in self.P_loads:
            vars.append(p_load[aux-1])

        for p_max in self.P_maxs:
            vars.append(p_max[aux-1])

        vars.append(aux)
        return np.array(vars)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train CPO on the 30-bus OPF task.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument('--total-steps', type=int, default=4 * 20* 40)
    parser.add_argument('--steps-per-epoch', type=int, default= 4* 20)
    parser.add_argument('--batch-size', type=int, default=40)
    parser.add_argument('--cost-limit', type=float, default=10.0)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--cost-gamma', type=float, default=0.5)
    parser.add_argument('--save-model-freq', type=int, default=10)
    parser.add_argument('--plot-smooth', type=int, default=1)
    parser.add_argument('--skip-plot', action='store_true')
    parser.add_argument('--init-model-path', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wind-file', type=Path, default=None)
    parser.add_argument('--sun-file', type=Path, default=None)
    parser.add_argument('--load-file', type=Path, default=None)
    parser.add_argument('--profile-npz', type=Path, default=None)
    parser.add_argument('--wind-perturb-device-index', type=int, default=0)
    parser.add_argument('--wind-perturb-time-index', type=int, default=None)
    parser.add_argument('--wind-perturb-fraction', type=float, default=None)
    return parser


def prepare_run_dirs(run_root: Path) -> dict[str, Path]:
    run_root.mkdir(parents=True, exist_ok=True)
    paths = {
        'run_root': run_root,
        'logs': run_root / 'logs',
        'analysis': run_root / 'analysis',
        'models': run_root / 'models',
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_custom_cfgs(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    return {
        'seed': args.seed,
        'train_cfgs': {
            'total_steps': args.total_steps,
            'vector_env_nums': 1,
            'parallel': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': args.steps_per_epoch,
            'batch_size': args.batch_size,
            'target_kl': 0.01,
            'cost_limit': args.cost_limit,
            'obs_normalize': True,
            'gamma': args.gamma,
            'cost_gamma': args.cost_gamma,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'use_tensorboard': True,
            'save_model_freq': args.save_model_freq,
            'log_dir': str(paths['logs']),
        },
    }


def latest_checkpoint(torch_dir: Path) -> Path | None:
    checkpoints = list(torch_dir.glob('epoch-*.pt'))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.stem.split('-')[-1]))


def best_checkpoint(log_dir: Path) -> Path | None:
    progress_path = log_dir / 'progress.csv'
    torch_dir = log_dir / 'torch_save'
    if not progress_path.exists():
        return latest_checkpoint(torch_dir)
    progress = pd.read_csv(progress_path)
    if progress.empty or 'Train/Epoch' not in progress:
        return latest_checkpoint(torch_dir)
    if 'Metrics/EpRet' in progress:
        best_row = progress['Metrics/EpRet'].astype(float).idxmax()
    elif 'Metrics/EpCost' in progress:
        best_row = progress['Metrics/EpCost'].astype(float).idxmin()
    else:
        return latest_checkpoint(torch_dir)
    epoch = int(progress.loc[best_row, 'Train/Epoch'])
    checkpoint = torch_dir / f'epoch-{epoch}.pt'
    return checkpoint if checkpoint.exists() else latest_checkpoint(torch_dir)


def export_run_artifacts(
    agent: omnisafe.Agent,
    paths: dict[str, Path],
    metrics: tuple[float, float, float],
    init_model_path: Path | None,
    profile_overrides: dict[str, Any] | None,
) -> None:
    log_dir = Path(agent.agent.logger.log_dir)
    torch_dir = log_dir / 'torch_save'
    final_checkpoint = latest_checkpoint(torch_dir)
    selected_best_checkpoint = best_checkpoint(log_dir)
    exported_model = None
    exported_best_model = None
    if final_checkpoint is not None:
        exported_model = paths['models'] / 'final_model.pt'
        shutil.copy2(final_checkpoint, exported_model)
    if selected_best_checkpoint is not None:
        exported_best_model = paths['models'] / 'best_model.pt'
        shutil.copy2(selected_best_checkpoint, exported_best_model)

    manifest = {
        'run_root': str(paths['run_root']),
        'log_dir': str(log_dir),
        'progress_csv': str(log_dir / 'progress.csv'),
        'config_json': str(log_dir / 'config.json'),
        'analysis_dir': str(paths['analysis']),
        'final_model_path': str(exported_model) if exported_model is not None else None,
        'best_model_path': str(exported_best_model) if exported_best_model is not None else None,
        'init_model_path': str(init_model_path) if init_model_path is not None else None,
        'profile_overrides': profile_overrides or {},
        'metrics': {
            'ep_ret': float(metrics[0]),
            'ep_cost': float(metrics[1]),
            'ep_len': float(metrics[2]),
        },
    }
    with open(paths['run_root'] / 'run_manifest.json', mode='w', encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=True, indent=2)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if (args.wind_perturb_time_index is None) != (args.wind_perturb_fraction is None):
        parser.error('--wind-perturb-time-index and --wind-perturb-fraction must be set together.')
    launch_cwd = Path.cwd()
    os.chdir(THIS_DIR)

    run_root = args.output_dir
    if not run_root.is_absolute():
        run_root = (launch_cwd / run_root).resolve()
    paths = prepare_run_dirs(run_root)

    os.environ['SRRL_ANALYSIS_DIR'] = str(paths['analysis'])
    for env_name, arg_value in (
        ('SRRL_WIND_CSV', args.wind_file),
        ('SRRL_SUN_CSV', args.sun_file),
        ('SRRL_LOAD_CSV', args.load_file),
        ('SRRL_PROFILE_NPZ', args.profile_npz),
    ):
        if arg_value is None:
            os.environ.pop(env_name, None)
            continue
        resolved_path = arg_value if arg_value.is_absolute() else (launch_cwd / arg_value).resolve()
        os.environ[env_name] = str(resolved_path)

    if args.wind_perturb_time_index is None:
        for env_name in (
            WIND_PERTURB_DEVICE_ENV,
            WIND_PERTURB_TIME_ENV,
            WIND_PERTURB_FRACTION_ENV,
        ):
            os.environ.pop(env_name, None)
        profile_overrides = {}
    else:
        os.environ[WIND_PERTURB_DEVICE_ENV] = str(args.wind_perturb_device_index)
        os.environ[WIND_PERTURB_TIME_ENV] = str(args.wind_perturb_time_index)
        os.environ[WIND_PERTURB_FRACTION_ENV] = str(args.wind_perturb_fraction)
        profile_overrides = {
            'wind_perturb_device_index': args.wind_perturb_device_index,
            'wind_perturb_time_index': args.wind_perturb_time_index,
            'wind_perturb_fraction': args.wind_perturb_fraction,
            'wind_perturb_percent': args.wind_perturb_fraction * 100.0,
        }

    if args.init_model_path is not None:
        init_model_path = args.init_model_path
        if not init_model_path.is_absolute():
            init_model_path = (launch_cwd / init_model_path).resolve()
        os.environ['SRRL_INIT_MODEL_PATH'] = str(init_model_path)
    else:
        init_model_path = None
        os.environ.pop('SRRL_INIT_MODEL_PATH', None)
    custom_cfgs = build_custom_cfgs(args, paths)

    agent = omnisafe.Agent('CPO', 'OptimalPowerFlowEnv-v0', custom_cfgs=custom_cfgs)
    metrics = agent.learn()
    if not args.skip_plot:
        agent.plot(smooth=args.plot_smooth)
    export_run_artifacts(agent, paths, metrics, init_model_path, profile_overrides)


if __name__ == '__main__':
    main()
