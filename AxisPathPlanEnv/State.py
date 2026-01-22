import numpy as np
import torch
from AxisPathPlanEnv.util import calculate_distance
import fcl

class State:
    def __init__(self, env):
        # --- 1. 基础信息提取 (保留原逻辑) ---
        self.position = env.nowpos.copy()          # 当前位置
        self.gesture = env.nowges.copy()           # 当前姿态
        
        # 目标相对信息 (用于原始的 State 计算)
        self.targetpos_relative = env.targetpos - self.position
        self.targetgesture_relative = env.targetges - self.gesture
        self.distance_to_targetpos = calculate_distance(self.position, env.targetpos)
        self.distance_to_targetges = calculate_distance(self.gesture, env.targetges)
        
        # 本体运动信息
        self.now_vec = (self.position - env.lastpos) # 位置变化向量
        self.now_dir = (self.gesture - env.lastges)  # 姿态变化向量
        
        # --- 2. 障碍物感知 (保留原逻辑，耗时操作) ---
        self.obstacle_info = []
        for obs in env.obstacles:
            vec_to_obs = obs.centerPoint - self.position
            # 配置距离请求
            request = fcl.DistanceRequest(enable_nearest_points=True)
            result = fcl.DistanceResult()
            # 注意：这里需要确保 env.moving_tool 已经更新到当前位置
            dist = fcl.distance(obs.modelforfcl, env.moving_tool.modelforfcl, request, result)
            
            # 归一化方向向量并拼接距离
            norm_vec = vec_to_obs / (dist + 1e-6)
            self.obstacle_info.extend([dist, *norm_vec]) # [dist, dx, dy, dz]
            
        # --- 3. 原始状态向量 (兼容非 HER 模式) ---
        # 维度：[3, 3, 3, 3, 1, 1] + 障碍物维度
        self.states = np.concatenate([
            self.targetpos_relative,    # 3
            self.targetgesture_relative,# 3
            self.now_vec,               # 3
            self.now_dir,               # 3
            [self.distance_to_targetpos], # 1
            [self.distance_to_targetges], # 1
            # self.obstacle_info        # 如果你之前包含障碍物信息，这里应该加上
        ], dtype=np.float32)

        # --- 4. HER 专用纯净观测 (新增) ---
        # 剔除所有 target 相关的信息，因为 HER 回放时 Target 会变
        # 只保留：本体运动信息 + 障碍物信息 + (可选)绝对坐标
        self.her_observation = np.concatenate([
            self.now_vec,
            self.now_dir,
            self.position, # HER 中通常需要本体绝对坐标来推断与新 Goal 的关系
            self.gesture,
            np.array(self.obstacle_info) if self.obstacle_info else np.array([])
        ], dtype=np.float32)

    @staticmethod
    def makeBatch(states: list["State"])->torch.Tensor:
        return torch.stack([torch.tensor(state.states) for state in states])