import numpy as np
import fcl
import random
import sys

sys.path.append('E:\pathplanning\pathplanningcode')
from AxisPathPlanEnv.MapEnv import MapEnv
from AxisPathPlanEnv.Prime import Cylinder
from AxisPathPlanEnv.util import calculate_distance, save_plot_3d_path

class Node:
    def __init__(self, pos):
        self.pos = np.array(pos)
        self.parent = None

class RRTPlanner:
    def __init__(self, env):
        self.env = env
        self.step_size = 1.0      # 生长步长
        self.max_iter = 1000      # 最大采样次数
        self.goal_sample_rate = 0.1 # 有 10% 的概率直接向目标采样
        
    def get_random_node(self, target_pos):
        """采样：一定概率直接取目标点，否则随机"""
        if random.random() < self.goal_sample_rate:
            return Node(target_pos)
        
        rand_pos = np.array([
            random.uniform(self.env.x_range[0], self.env.x_range[1]),
            random.uniform(self.env.y_range[0], self.env.y_range[1]),
            random.uniform(self.env.z_range[0], self.env.z_range[1])
        ])
        return Node(rand_pos)
    
    def get_nearest_node(self, node_list, rnd_node):
        """在树中找到距离采样点最近的节点"""
        dists = [np.linalg.norm(node.pos - rnd_node.pos) for node in node_list]
        min_idx = dists.index(min(dists))
        return node_list[min_idx]
    
    def steer(self, from_node, to_node):
        """从 from 向 to 走一步，生成新节点"""
        vec = to_node.pos - from_node.pos
        dist = np.linalg.norm(vec)
        
        if dist <= self.step_size:
            return to_node
        
        # 截断
        new_pos = from_node.pos + (vec / dist) * self.step_size
        return Node(new_pos)
    
    def check_collision(self, node):
        """碰撞检测：生成一个虚拟工具判断是否碰撞"""
        # 注意：RRT通常只规划位置，姿态暂时设为默认（单位矩阵）
        # 如果你的障碍物很密集，姿态影响很大，这里需要更复杂的逻辑
        temp_tool = Cylinder(height=self.env.tool_size[0], radius=self.env.tool_size[1], 
                             centerPoint=node.pos)
        
        req = fcl.CollisionRequest()
        
        
        for obs in self.env.obstacles:
            res = fcl.CollisionResult()
            fcl.collide(temp_tool.modelforfcl, obs.modelforfcl, req, res)
            if res.is_collision:
                return True # 发生碰撞
        return False

    def plan(self, start_pos, target_pos):
        start_node = Node(start_pos)
        goal_node = Node(target_pos)
        
        self.node_list = [start_node]
        print("RRT Planning started...")
        
        for i in range(self.max_iter):
            # 1. 采样
            rnd_node = self.get_random_node(target_pos)
            
            # 2. 找最近点
            nearest_node = self.get_nearest_node(self.node_list, rnd_node)
            
            # 3. 延伸
            new_node = self.steer(nearest_node, rnd_node)
            
            # 4. 碰撞检测
            if not self.check_collision(new_node):
                new_node.parent = nearest_node
                self.node_list.append(new_node)
                
                # 5. 判断是否到达目标附近
                if np.linalg.norm(new_node.pos - target_pos) < self.step_size:
                    print(f"RRT Reached Target in {i} iters!")
                    goal_node.parent = new_node
                    return self.generate_final_path(goal_node), True
                    
        print("RRT Failed to find path.")
        return None, False

    def generate_final_path(self, goal_node):
        path = []
        node = goal_node
        while node is not None:
            path.append(node.pos)
            node = node.parent
        return np.array(path[::-1]) # 反转路径

# --- 运行测试 ---
if __name__ == "__main__":
    # 配置 (保持一致)
    config = {
        "envxrange": [-10, 10], "envyrange": [-10, 10], "envzrange": [-10, 10],
        "obstacles_num": 5, "safe_distance": 1.0, "tool_size": [2.0, 0.5],
        "maxstep": 100, "period": 0.1, "alpha_max": 1.0,
        "start": [0,0,0,0,0,0], "target": [8,8,8,0,0,0]
    }
    
    env = MapEnv(config)
    env.reset()
    
    planner = RRTPlanner(env)
    path, success = planner.plan(env.start_pos, env.target_pos)
    
    if success:
        bounds = np.concatenate([env.x_range, env.y_range, env.z_range])
        save_plot_3d_path(path, bounds, env.obstacles, "result_rrt.png",
                          info={"algorithm": "RRT", "success": success})
        print("Result saved to result_rrt.png")