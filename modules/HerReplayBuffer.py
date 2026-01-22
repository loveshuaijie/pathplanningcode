import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

class HerReplayBuffer(DictReplayBuffer):
    """
    适配 Stable-Baselines3 的 HER 回放缓冲区。
    继承自 DictReplayBuffer 以支持字典类型的 Observation。
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_sampled_goal: int = 4,
        goal_selection_strategy: str = "future",
        **kwargs,
    ):
        # 1. 调用父类 DictReplayBuffer 的初始化
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        # 2. HER 特有参数
        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        
        # 3. 提取维度信息
        self.obs_dim = observation_space.spaces['observation'].shape[0]
        self.goal_dim = observation_space.spaces['achieved_goal'].shape[0]
        self.action_dim = action_space.shape[0]

        # 注意：SB3 的 Buffer 存储逻辑与你原始的 Episode-based 存储略有不同
        # SB3 默认是扁平化存储，HER 的逻辑通常由 SB3 的适配层处理
        # 如果你希望完全手动控制采样逻辑，可以在此重写 sample() 方法
        
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[Any] = None) -> DictReplayBufferSamples:
        """
        重写采样逻辑，可以在这里实现自定义的 HER 目标替换逻辑。
        如果使用默认的 MultiInputPolicy 和内置处理，通常父类方法即可满足。
        """
        return super()._get_samples(batch_inds, env)

    # 如果你需要保留原始代码中的 reward_func 计算逻辑
    def set_reward_func(self, reward_func):
        self.reward_func = reward_func