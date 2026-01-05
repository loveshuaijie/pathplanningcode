import numpy as np
import torch
from AxisPathPlanEnv.util import calculate_distance
import fcl
# State.py
class State:
    def __init__(self, env):
        # 包含完整运动学信息
        self.position = env.nowpos.copy()          # 当前位置（必须使用copy避免引用问题）
        self.gesture = env.nowges.copy()  # 当前姿态（必须使用copy避免引用问题）
        #self.velocity = (env.nowpos - env.lastpos) / env.period  # 计算瞬时速度
        self.targetpos_relative = env.targetpos - self.position  # 目标相对位置
        self.targetgesture_relative = env.targetges - self.gesture  # 目标相对姿态
        self.distance_to_targetpos = calculate_distance(self.position, env.targetpos)
        self.distance_to_targetges = calculate_distance(self.gesture, env.targetges)
        self.now_vec=(self.position - env.lastpos) #当前方向
        self.now_dir=(self.gesture - env.lastges) #当前姿态
        # 障碍物信息（包含距离和方向）
        
        self.obstacle_info = []
        for obs in env.obstacles:
            vec_to_obs = obs.centerPoint - self.position
            # 配置距离请求
            request = fcl.DistanceRequest(enable_nearest_points=True)  # 启用最近点计算
            result = fcl.DistanceResult()
            dist = fcl.distance(obs.modelforfcl, env.moving_tool.modelforfcl, request, result)
            self.obstacle_info.extend([dist, *(vec_to_obs / (dist + 1e-6))])  # 归一化方向向量
            
        # 拼接状态向量（示例维度：[x,y,z, vx,vy,vz, dx,dy,dz, dist, obs1_dist, obs1_dir_x,...]） 
        self.states = np.concatenate([
            self.targetpos_relative,
            self.targetgesture_relative,
            self.now_vec,
            self.now_dir,
            [self.distance_to_targetpos],
            [self.distance_to_targetges],
        ], dtype=np.float32)
    @staticmethod
    def makeBatch(states: list["State"])->torch.Tensor:
        return torch.stack([torch.tensor(state) for state in states])