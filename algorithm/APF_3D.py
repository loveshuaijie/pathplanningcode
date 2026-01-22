import numpy as np
import fcl
import sys

sys.path.append('E:\pathplanning\pathplanningcode')

from AxisPathPlanEnv.MapEnv import MapEnv
from AxisPathPlanEnv.Prime import Cylinder
from AxisPathPlanEnv.util import calculate_distance, save_plot_3d_path

class APFPlanner:
    def __init__(self, env):
        self.env = env
        self.k_att = 5.0        # 引力增益
        self.k_rep = 10.0      # 斥力增益
        self.rho_0 = 3.0        # 斥力影响范围 (感知半径)
        self.step_size = 0.1    # 梯度下降步长
        self.max_iters = 2000   # 最大迭代次数
        
    def get_att_force(self, current_pos, target_pos):
        """计算引力: F_att = k * (target - current)"""
        vec = target_pos - current_pos
        dist = np.linalg.norm(vec)
        if dist < 1e-5:
            return np.zeros(3)
        # 这里的引力可以是线性的，也可以是二次的，这里用线性引导
        return self.k_att * vec 

    def get_rep_force(self, current_pos):
        """计算斥力: 基于 FCL 最近距离"""
        total_rep_force = np.zeros(3)
        
        # 创建一个临时的工具模型用于检测距离
        # 注意：这里简化处理，只考虑位置，忽略姿态对形状的影响（假设工具是球或直立圆柱）
        temp_tool = Cylinder(height=self.env.tool_size[0], radius=self.env.tool_size[1], centerPoint=current_pos)
        
        req = fcl.DistanceRequest(enable_nearest_points=True)
        
        
        for obs in self.env.obstacles:
            res = fcl.DistanceResult()
            dist = fcl.distance(temp_tool.modelforfcl, obs.modelforfcl, req, res)
            
            # 如果在斥力范围内
            if dist < self.rho_0:
                if dist <= 0.1: dist = 0.1 # 防止除零
                
                # 计算斥力向量方向：从障碍物指向工具
                # FCL nearest_points[0] 是 obs 上的点，[1] 是 tool 上的点
                # 我们需要 tool - obs 的方向
                obs_point = np.array(res.nearest_points[0])
                tool_point = np.array(res.nearest_points[1])
                
                rep_vec = tool_point - obs_point
                rep_vec_norm = np.linalg.norm(rep_vec)
                
                if rep_vec_norm > 1e-6:
                    unit_rep_vec = rep_vec / rep_vec_norm
                else:
                    unit_rep_vec = np.random.uniform(-1, 1, 3) # 随机扰动防止重合
                    unit_rep_vec /= np.linalg.norm(unit_rep_vec)

                # 斥力公式 (标准 APF)
                # F = k * (1/d - 1/rho0) * (1/d^2) * unit_vec
                force_val = self.k_rep * (1.0/dist - 1.0/self.rho_0) * (1.0/(dist**2))
                total_rep_force += force_val * unit_rep_vec
                
        return total_rep_force

    def plan(self, start_pos, target_pos):
        path = [start_pos.copy()]
        current_pos = start_pos.copy()
        
        print("APF Planning started...")
        for i in range(self.max_iters):
            # 1. 检查是否到达目标
            if np.linalg.norm(current_pos - target_pos) < 0.2:
                path.append(target_pos.copy())
                print(f"APF Reached Target in {i} steps!")
                return np.array(path), True
            
            # 2. 计算受力
            f_att = self.get_att_force(current_pos, target_pos)
            f_rep = self.get_rep_force(current_pos)
            
            f_total = f_att + f_rep
            
            # 3. 梯度下降更新位置
            # 归一化合力方向，防止步长过大
            f_norm = np.linalg.norm(f_total)
            if f_norm > 1e-5:
                direction = f_total / f_norm
                current_pos += direction * self.step_size
            else:
                # 陷入局部极小值，加随机扰动
                current_pos += np.random.uniform(-0.1, 0.1, 3)
            
            # 越界检查
            current_pos = np.clip(current_pos, 
                                  [self.env.x_range[0], self.env.y_range[0], self.env.z_range[0]],
                                  [self.env.x_range[1], self.env.y_range[1], self.env.z_range[1]])
            
            path.append(current_pos.copy())
            
        print("APF Timeout: Local Minima or too far.")
        return np.array(path), False

# --- 运行测试 ---
if __name__ == "__main__":
    # 配置
    config = {
        "envxrange": [-10, 10], "envyrange": [-10, 10], "envzrange": [-10, 10],
        "obstacles_num": 5, "safe_distance": 1.0, "tool_size": [2.0, 0.5],
        "maxstep": 100, "period": 0.1, "alpha_max": 1.0,
        "start": [0,0,0,0,0,0], "target": [8,8,8,0,0,0] # 距离远一点，测试避障
    }
    
    # 初始化环境
    env = MapEnv(config)
    env.reset() # 生成随机障碍物
    
    # 规划
    planner = APFPlanner(env)
    path, success = planner.plan(env.start_pos, env.target_pos)
    
    # 绘图
    bounds = np.concatenate([env.x_range, env.y_range, env.z_range])
    save_plot_3d_path(path, bounds, env.obstacles, "result_apf.png", 
                      info={"algorithm": "APF", "success": success})
    print("Result saved to result_apf.png")