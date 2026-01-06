import numpy as np
from AxisPathPlanEnv.Prime import *
from AxisPathPlanEnv.util import *
import fcl
from gym import spaces
from AxisPathPlanEnv.State import State
from AxisPathPlanEnv.Action import Action
import gym
import random
from typing import Tuple, List, Optional, Dict, Any, Union


class MapEnv(gym.Env):
    """通用路径规划环境
    
    该环境既可用于强化学习，也可用于传统路径规划算法。
    支持3D空间中的路径规划，包含障碍物避障、方向约束等功能。
    """
    
    def __init__(self, env_config: Dict[str, Any], render_mode: str = None) -> None:
        """初始化环境
        
        Args:
            env_config: 环境配置字典
            render_mode: 渲染模式，支持None（不渲染）或'human'
        """
        super().__init__()
        self.render_mode = render_mode
        
        # 环境边界
        self.xrange = env_config["envxrange"]
        self.yrange = env_config["envyrange"]
        self.zrange = env_config["envzrange"]
        
        # 障碍物
        self.obstacles = []
        self.obstacles_num = env_config["obstacles_num"]
        
        # 起点和终点
        self.start = np.array(env_config["start"])
        self.target = np.array(env_config["target"])
        self.startpos = self.start[0:3]
        self.startges = self.start[3:6]
        self.targetpos = self.target[0:3]
        self.targetges = self.target[3:6]
        
        # 状态变量
        self.totalreward = 0
        self.done = False
        self.nowpos = self.startpos.copy()
        self.nowges = self.startges.copy()
        self.lastpos = self.startpos.copy()
        self.lastges = self.startges.copy()
        self.last_dis_to_target = 0
        self.last_dir_to_target = 0
        self.goal_distance = 0
        self.last_dis_to_obstacle = [0 for _ in range(self.obstacles_num)]
        self.lastpath = np.array([0, 0, 0])
        self.timestep = 0
        self.trajectory = []
        
        # 工具参数
        self.tool_size = env_config["tool_size"]
        self.moving_tool = Cylinder(
            height=self.tool_size[0], 
            radius=self.tool_size[1], 
            centerPoint=self.nowpos
        )
        
        # 环境参数
        self.maxstep = env_config["maxstep"]
        self.period = env_config["period"]
        self.safe_distance = env_config["safe_distance"]
        self.alpha_max = env_config["alpha_max"]
        self.reachpos_scale = env_config["reachpos_scale"]
        self.reachges_scale = env_config["reachges_scale"]
        self.Vmax = env_config.get("Vmax", 1.0)
        
        # 计算初始目标距离和角度
        self.goal_distance = calculate_distance(self.startpos, self.targetpos)
        self.goal_dir_to_target = calculate_distance(self.startges, self.targetges)
        self.reach_distance = self.goal_distance / self.reachpos_scale
        self.reach_ges = self.goal_dir_to_target / self.reachges_scale
        
        # 确保到达阈值不为零
        if self.reach_distance < 0.1:
            self.reach_distance = 0.1
        if self.reach_ges < 0.1:
            self.reach_ges = 0.1
        
        # 观察空间和动作空间（强化学习专用）
        state = State(self)
        state_dim = len(state.states)
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32
        )
        
        # 用于非强化学习算法的规划状态
        self._planning_state = {
            'path': [],
            'collision_checks': 0,
            'planning_time': 0,
            'nodes_expanded': 0
        }
        
        self.reset()
    
    # ==================== 基础环境操作 ====================
    
    def reset(self, 
              start: Optional[np.ndarray] = None,
              target: Optional[np.ndarray] = None,
              generate_obstacles: bool = True,
              obstacle_list: Optional[List] = None) -> np.ndarray:
        """重置环境
        
        Args:
            start: 起始点坐标和姿态，形状为(6,)。如果为None则使用配置中的起点或随机生成
            target: 目标点坐标和姿态，形状为(6,)。如果为None则使用配置中的目标点或随机生成
            generate_obstacles: 是否随机生成障碍物
            obstacle_list: 预定义的障碍物列表，如果提供则使用此列表而不随机生成
            
        Returns:
            初始状态向量
        """
        # 设置起点
        if start is not None:
            self.start = np.array(start)
            self.startpos = self.start[0:3]
            self.startges = self.start[3:6]
        
        # 设置终点
        if target is not None:
            self.target = np.array(target)
            self.targetpos = self.target[0:3]
            self.targetges = self.target[3:6]
        else:
            # 如果未指定目标且配置允许随机生成，则随机生成目标
            if generate_obstacles:
                targetpos = np.random.uniform(low=-10, high=10, size=(3,))
                targetges = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
                self.target = np.concatenate((targetpos, targetges))
                self.targetpos = self.target[0:3]
                self.targetges = self.target[3:6]
        
        # 重置状态变量
        self.nowpos = self.startpos.copy()
        self.nowges = self.startges.copy()
        self.totalreward = 0
        self.done = False
        self.timestep = 0
        self.trajectory = [self.nowpos.copy()]
        self._planning_state = {
            'path': [],
            'collision_checks': 0,
            'planning_time': 0,
            'nodes_expanded': 0
        }
        
        # 设置或生成障碍物
        if obstacle_list is not None:
            self.obstacles = obstacle_list
        elif generate_obstacles:
            self.generate_obstacles()
        
        # 更新工具位置
        rotation_matrix = euler_to_rotation_matrix(
            self.nowges[0], self.nowges[1], self.nowges[2]
        )
        self.moving_tool = Cylinder(
            height=self.tool_size[0],
            radius=self.tool_size[1],
            centerPoint=self.nowpos,
            orientation=rotation_matrix
        )
        
        # 更新距离计算
        self.goal_distance = calculate_distance(self.startpos, self.targetpos)
        self.goal_dir_to_target = calculate_distance(self.startges, self.targetges)
        self.reach_distance = self.goal_distance / self.reachpos_scale
        self.reach_ges = self.goal_dir_to_target / self.reachges_scale
        
        if self.reach_distance < 0.1:
            self.reach_distance = 0.1
        if self.reach_ges < 0.1:
            self.reach_ges = 0.1
        
        # 创建并返回初始状态
        self.state = State(self)
        return self.state.states
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作（强化学习接口）
        
        Args:
            action: 动作向量，形状为(6,)，前3个是位置变化，后3个是姿态变化
            
        Returns:
            tuple: (观测状态, 奖励, 是否终止, 额外信息)
        """
        return self._execute_action(action, compute_reward=True)
    
    # ==================== 路径规划接口 ====================
    
    def plan_step(self, new_position: np.ndarray, new_orientation: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """为传统路径规划算法设计的步进函数
        
        Args:
            new_position: 新位置坐标 (x, y, z)
            new_orientation: 新姿态 (rx, ry, rz)，如果为None则保持当前姿态
            
        Returns:
            包含以下信息的字典:
            - 'success': 是否成功执行移动
            - 'collision': 是否发生碰撞
            - 'out_of_bounds': 是否超出边界
            - 'position': 实际到达的位置
            - 'orientation': 实际到达的姿态
            - 'distance_to_target': 到目标的距离
            - 'distance_to_obstacles': 到各个障碍物的距离列表
        """
        # 记录规划步骤
        self._planning_state['nodes_expanded'] += 1
        
        # 计算新姿态
        if new_orientation is None:
            new_orientation = self.nowges.copy()
        
        # 检查边界
        if not self._check_bounds(new_position):
            return {
                'success': False,
                'collision': False,
                'out_of_bounds': True,
                'position': self.nowpos.copy(),
                'orientation': self.nowges.copy(),
                'distance_to_target': calculate_distance(self.nowpos, self.targetpos),
                'distance_to_obstacles': self._get_distances_to_obstacles()
            }
        
        # 检查碰撞
        rotation_matrix = euler_to_rotation_matrix(
            new_orientation[0], new_orientation[1], new_orientation[2]
        )
        temp_tool = Cylinder(
            height=self.tool_size[0],
            radius=self.tool_size[1],
            centerPoint=new_position,
            orientation=rotation_matrix
        )
        
        self._planning_state['collision_checks'] += 1
        
        if self._check_collision(temp_tool):
            return {
                'success': False,
                'collision': True,
                'out_of_bounds': False,
                'position': self.nowpos.copy(),
                'orientation': self.nowges.copy(),
                'distance_to_target': calculate_distance(self.nowpos, self.targetpos),
                'distance_to_obstacles': self._get_distances_to_obstacles()
            }
        
        # 执行移动
        self.lastpos = self.nowpos.copy()
        self.lastges = self.nowges.copy()
        self.nowpos = new_position.copy()
        self.nowges = new_orientation.copy()
        self.moving_tool = temp_tool
        self.trajectory.append(self.nowpos.copy())
        
        # 更新状态
        self.state = State(self)
        self.timestep += 1
        
        # 检查终止条件
        reached_target = self.judgeTarget()
        collision = False  # 已经检查过碰撞
        timeout = self.judgeTime()
        
        if reached_target:
            self.done = True
            self.trajectory.append(self.targetpos.copy())
        
        return {
            'success': True,
            'collision': False,
            'out_of_bounds': False,
            'position': self.nowpos.copy(),
            'orientation': self.nowges.copy(),
            'distance_to_target': calculate_distance(self.nowpos, self.targetpos),
            'distance_to_obstacles': self._get_distances_to_obstacles(),
            'reached_target': reached_target,
            'timeout': timeout
        }
    
    def get_valid_neighbors(self, position: np.ndarray, orientation: np.ndarray, 
                           step_size: float = 1.0) -> List[Dict[str, Any]]:
        """获取当前位置的有效邻居位置（用于图搜索算法）
        
        Args:
            position: 当前位置 (x, y, z)
            orientation: 当前姿态 (rx, ry, rz)
            step_size: 搜索步长
            
        Returns:
            有效邻居位置列表，每个元素包含位置、姿态和移动成本
        """
        neighbors = []
        directions = [
            (step_size, 0, 0),
            (-step_size, 0, 0),
            (0, step_size, 0),
            (0, -step_size, 0),
            (0, 0, step_size),
            (0, 0, -step_size)
        ]
        
        for dx, dy, dz in directions:
            new_pos = position + np.array([dx, dy, dz])
            
            # 检查边界
            if not self._check_bounds(new_pos):
                continue
            
            # 检查碰撞
            temp_tool = Cylinder(
                height=self.tool_size[0],
                radius=self.tool_size[1],
                centerPoint=new_pos,
                orientation=euler_to_rotation_matrix(
                    orientation[0], orientation[1], orientation[2]
                )
            )
            
            if not self._check_collision(temp_tool):
                # 计算启发式成本（到目标的欧氏距离）
                cost = calculate_distance(new_pos, position)  # 移动距离作为成本
                heuristic = calculate_distance(new_pos, self.targetpos)
                
                neighbors.append({
                    'position': new_pos,
                    'orientation': orientation,
                    'cost': cost,
                    'heuristic': heuristic,
                    'total_cost': cost + heuristic
                })
        
        return neighbors
    
    def evaluate_path(self, path: List[np.ndarray]) -> Dict[str, Any]:
        """评估给定路径的质量
        
        Args:
            path: 路径点列表，每个点为(x, y, z, rx, ry, rz)或(x, y, z)
            
        Returns:
            包含路径评估指标的字典
        """
        if len(path) == 0:
            return {'valid': False, 'length': 0, 'collisions': 0}
        
        # 保存当前状态
        original_state = {
            'nowpos': self.nowpos.copy(),
            'nowges': self.nowges.copy(),
            'trajectory': self.trajectory.copy(),
            'timestep': self.timestep
        }
        
        # 评估路径
        collisions = 0
        path_length = 0
        valid = True
        
        # 重置到起点
        self.nowpos = self.startpos.copy()
        self.nowges = self.startges.copy()
        
        for i, point in enumerate(path):
            if len(point) == 3:
                # 只有位置，使用当前姿态
                result = self.plan_step(point, self.nowges)
            else:
                # 包含姿态
                result = self.plan_step(point[:3], point[3:6])
            
            if i > 0:
                prev_point = path[i-1]
                path_length += calculate_distance(
                    prev_point[:3] if len(prev_point) > 3 else prev_point,
                    point[:3] if len(point) > 3 else point
                )
            
            if result['collision']:
                collisions += 1
                valid = False
                break
            
            if result['out_of_bounds']:
                valid = False
                break
        
        # 恢复原始状态
        self.nowpos = original_state['nowpos']
        self.nowges = original_state['nowges']
        self.trajectory = original_state['trajectory']
        self.timestep = original_state['timestep']
        
        return {
            'valid': valid,
            'length': path_length,
            'collisions': collisions,
            'distance_to_target': calculate_distance(self.nowpos, self.targetpos) 
            if valid else float('inf'),
            'smoothness': self._calculate_path_smoothness(path) if len(path) > 2 else 0
        }
    
    # ==================== 环境信息获取 ====================
    
    def get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        return {
            'bounds': {
                'x': self.xrange,
                'y': self.yrange,
                'z': self.zrange
            },
            'start': {
                'position': self.startpos.tolist(),
                'orientation': self.startges.tolist()
            },
            'target': {
                'position': self.targetpos.tolist(),
                'orientation': self.targetges.tolist()
            },
            'obstacles': len(self.obstacles),
            'tool_size': self.tool_size,
            'safe_distance': self.safe_distance,
            'reach_distance': self.reach_distance,
            'reach_angle': self.reach_ges
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'position': self.nowpos.tolist(),
            'orientation': self.nowges.tolist(),
            'distance_to_target': calculate_distance(self.nowpos, self.targetpos),
            'distance_to_goal_orientation': calculate_distance(self.nowges, self.targetges),
            'timestep': self.timestep,
            'trajectory_length': len(self.trajectory)
        }
    
    def get_planning_stats(self) -> Dict[str, Any]:
        """获取规划统计信息"""
        return self._planning_state.copy()
    
    # ==================== 障碍物管理 ====================
    
    def append_obstacle(self, obstacle) -> None:
        """添加障碍物"""
        self.obstacles.append(obstacle)
    
    def clear_obstacles(self) -> None:
        """清空所有障碍物"""
        self.obstacles = []
    
    def set_obstacles(self, obstacles: List) -> None:
        """设置障碍物列表"""
        self.obstacles = obstacles
    
    def generate_obstacles(self) -> None:
        """生成多种类型的障碍物"""
        self.obstacles = []
        
        for _ in range(self.obstacles_num):
            pos = np.array([
                np.random.uniform(self.xrange[0], self.xrange[1]),
                np.random.uniform(self.yrange[0], self.yrange[1]),
                np.random.uniform(self.zrange[0], self.zrange[1])
            ])
            
            # 障碍物类型：球体、圆柱体、长方体
            obs_type = random.choice(['sphere', 'cylinder', 'box'])
            
            if obs_type == 'sphere':
                radius = np.random.uniform(2, 5)
                obstacle = Sphere(radius, pos)
            elif obs_type == 'cylinder':
                radius = np.random.uniform(1, 3)
                height = np.random.uniform(2, 6)
                obstacle = Cylinder(height, radius, pos)
            else:  # box
                size = np.random.uniform(3, 8, size=3)
                obstacle = Cuboid(size, pos)
            
            self.obstacles.append(obstacle)
    
    # ==================== 内部辅助方法 ====================
    
    def _execute_action(self, action: np.ndarray, compute_reward: bool = False) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作的内部方法"""
        self.lastpos = self.nowpos.copy()
        self.lastges = self.nowges.copy()
        self.lastpath = self.nowpos - self.lastpos
        
        # 应用动作
        scaled_action = action[0:3] * self.Vmax
        self.nowpos = self.nowpos + scaled_action * self.period
        self.nowges = self.nowges + action[3:6] * self.period
        
        # 更新工具
        rotation_matrix = euler_to_rotation_matrix(
            self.nowges[0], self.nowges[1], self.nowges[2]
        )
        self.moving_tool = Cylinder(
            height=self.tool_size[0],
            radius=self.tool_size[1],
            centerPoint=self.nowpos,
            orientation=rotation_matrix
        )
        self.trajectory.append(self.nowpos.copy())
        self.timestep += 1
        
        # 更新状态
        self.state = State(self)
        
        # 计算奖励（仅强化学习使用）
        reward = 0
        if compute_reward:
            reward += self.stepReward()
            reward += self.obstacleAwayReward()
            reward += self.angleReward()
        
        # 终止判断
        done = False
        info = {
            "nowpos": self.nowpos.copy(),
            "nowges": self.nowges.copy(),
            "target": self.target.copy(),
            "planning_stats": self._planning_state.copy()
        }
        
        if self.judgeTarget():
            self.trajectory.append(self.targetpos.copy())
            if compute_reward:
                reward += 100
            done = True
            info["terminal"] = "reached_target"
        elif self.judgeObstacle():
            if compute_reward:
                reward += -50
            done = True
            info["terminal"] = "collision"
        elif self.judgeTime():
            if compute_reward:
                reward += -5
            done = True
            info["terminal"] = "timeout"
        
        return self.state.states, reward, done, info
    
    def _check_bounds(self, position: np.ndarray) -> bool:
        """检查位置是否在边界内"""
        x, y, z = position
        return (self.xrange[0] <= x <= self.xrange[1] and
                self.yrange[0] <= y <= self.yrange[1] and
                self.zrange[0] <= z <= self.zrange[1])
    
    def _check_collision(self, tool) -> bool:
        """检查工具是否与障碍物碰撞"""
        for obstacle in self.obstacles:
            request = fcl.CollisionRequest()
            result = fcl.CollisionResult()
            fcl.collide(tool.modelforfcl, obstacle.modelforfcl, request, result)
            if result.is_collision:
                return True
        return False
    
    def _get_distances_to_obstacles(self) -> List[float]:
        """获取到所有障碍物的距离"""
        distances = []
        for obstacle in self.obstacles:
            distance = calculate_distance(self.nowpos, obstacle.centerPoint)
            distances.append(distance)
        return distances
    
    def _calculate_path_smoothness(self, path: List[np.ndarray]) -> float:
        """计算路径平滑度（角度变化总和）"""
        if len(path) < 3:
            return 0
        
        total_angle = 0
        for i in range(1, len(path)-1):
            # 计算三个连续点形成的角度
            p1 = path[i-1][:3] if len(path[i-1]) > 3 else path[i-1]
            p2 = path[i][:3] if len(path[i]) > 3 else path[i]
            p3 = path[i+1][:3] if len(path[i+1]) > 3 else path[i+1]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 1e-5 and np.linalg.norm(v2) > 1e-5:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                total_angle += angle
        
        return total_angle
    
    # ==================== 原有功能保持兼容 ====================
    
    def stepReward(self) -> float:
        """时间步奖励（强化学习专用）"""
        step_reward = 0
        
        # 计算当前点与终点的距离和角度差
        dis = calculate_distance(self.nowpos, self.targetpos)
        dir_diff = calculate_distance(self.nowges, self.targetges)
        base_reward = -dis/self.goal_distance - dir_diff/self.goal_dir_to_target
        
        # 方向一致性奖励
        direction = self.targetpos - self.nowpos
        movement = self.nowpos - self.lastpos
        if np.linalg.norm(direction) > 1e-5 and np.linalg.norm(movement) > 1e-5:
            cos_sim = np.dot(direction, movement) / (np.linalg.norm(direction) * np.linalg.norm(movement))
            direction_reward = 2 * cos_sim
        else:
            direction_reward = 0
        
        # 进步奖励
        if dis < self.last_dis_to_target and dir_diff < self.last_dir_to_target:
            step_reward += 1
        else:
            step_reward += -1
        
        step_reward += direction_reward
        step_reward += base_reward
        
        self.last_dis_to_target = dis
        self.last_dir_to_target = dir_diff
        
        return step_reward
    
    def obstacleAwayReward(self) -> float:
        """远离障碍物奖励（强化学习专用）"""
        obstacle_away_reward = 0
        for obstacle in self.obstacles:
            dis_to_obstacle = calculate_distance(self.nowpos, obstacle.centerPoint)
            if dis_to_obstacle < self.safe_distance:
                obstacle_away_reward -= (self.safe_distance - dis_to_obstacle) / self.safe_distance
        return obstacle_away_reward
    
    def angleReward(self) -> float:
        """转角奖励（强化学习专用）"""
        angle_reward = 0
        now_path = self.nowpos - self.lastpos
        
        if np.linalg.norm(self.lastpath) > 1e-5 and np.linalg.norm(now_path) > 1e-5:
            angle = calculate_angle(now_path, self.lastpath)
            if angle > self.alpha_max:
                angle_reward += -5
        
        return angle_reward
    
    def judgeTarget(self) -> bool:
        """判断是否到达目标点"""
        pos_distance = calculate_distance(self.nowpos, self.targetpos)
        ges_distance = calculate_distance(self.nowges, self.targetges)
        return pos_distance < self.reach_distance and ges_distance < self.reach_ges
    
    def judgeObstacle(self) -> bool:
        """判断是否碰撞或超出边界"""
        # 检查边界
        x, y, z = self.nowpos
        if not (self.xrange[0] <= x <= self.xrange[1] and
                self.yrange[0] <= y <= self.yrange[1] and
                self.zrange[0] <= z <= self.zrange[1]):
            return True
        
        # 检查障碍物碰撞
        for obstacle in self.obstacles:
            request = fcl.CollisionRequest()
            result = fcl.CollisionResult()
            fcl.collide(self.moving_tool.modelforfcl, obstacle.modelforfcl, request, result)
            if result.is_collision:
                return True
        
        return False
    
    def judgeTime(self) -> bool:
        """判断是否超时"""
        return self.timestep >= self.maxstep
    
    def render(self, pic_path: str = None, info: Dict = None) -> None:
        """渲染环境
        
        Args:
            pic_path: 图片保存路径，如果为None则不保存
            info: 额外信息
        """
        if pic_path is not None:
            save_plot_3d_path(
                self.trajectory,
                np.concatenate((self.xrange, self.yrange, self.zrange), axis=0),
                self.obstacles,
                pic_path,
                info
            )
    
    # ==================== 便捷属性 ====================
    
    @property
    def is_done(self) -> bool:
        """环境是否终止"""
        return self.done
    
    @property
    def current_position(self) -> np.ndarray:
        """当前位置"""
        return self.nowpos.copy()
    
    @property
    def current_orientation(self) -> np.ndarray:
        """当前姿态"""
        return self.nowges.copy()
    
    @property
    def path_taken(self) -> List[np.ndarray]:
        """已经走过的路径"""
        return self.trajectory.copy()


# ==================== 使用示例 ====================

def example_rl_usage():
    """强化学习使用示例"""
    env_config = {
        "envxrange": [-10, 10],
        "envyrange": [-10, 10],
        "envzrange": [-10, 10],
        "obstacles_num": 5,
        "start": [0, 0, 0, 0, 0, 0],
        "target": [5, 5, 5, 0, 0, 0],
        "tool_size": [2.0, 0.5],
        "maxstep": 1000,
        "period": 0.1,
        "safe_distance": 1.0,
        "alpha_max": np.pi / 4,
        "reachpos_scale": 10.0,
        "reachges_scale": 10.0,
        "Vmax": 1.0
    }
    
    env = MapEnv(env_config)
    
    # 标准的RL训练循环
    state = env.reset()
    done = False
    
    while not done:
        # 这里应该用RL算法产生动作
        action = np.random.uniform(-1, 1, size=(6,))
        
        next_state, reward, done, info = env.step(action)
        
        # 训练RL算法...
        state = next_state
    
    print(f"Episode finished with reward: {env.totalreward}")
    env.render("path.png", info)


def example_path_planning_usage():
    """传统路径规划算法使用示例"""
    env_config = {
        "envxrange": [-10, 10],
        "envyrange": [-10, 10],
        "envzrange": [-10, 10],
        "obstacles_num": 0,  # 不自动生成障碍物
        "start": [0, 0, 0, 0, 0, 0],
        "target": [8, 8, 8, 0, 0, 0],
        "tool_size": [2.0, 0.5],
        "maxstep": 1000,
        "period": 0.1,
        "safe_distance": 1.0,
        "alpha_max": np.pi / 4,
        "reachpos_scale": 10.0,
        "reachges_scale": 10.0,
        "Vmax": 1.0
    }
    
    env = MapEnv(env_config)
    
    # 手动设置障碍物
    obstacles = [
        Sphere(radius=2.0, centerPoint=np.array([3, 3, 3])),
        Cuboid(size=np.array([2, 2, 2]), centerPoint=np.array([6, 6, 6]))
    ]
    env.set_obstacles(obstacles)
    
    # 重置环境（使用手动设置的障碍物）
    env.reset(generate_obstacles=False)
    
    # 传统规划算法示例（如A*）
    path = []
    current_pos = env.current_position
    current_ori = env.current_orientation
    
    # 简单的直线规划（实际应使用更复杂的算法）
    for t in np.linspace(0, 1, 10):
        new_pos = current_pos + t * (env.targetpos - current_pos)
        result = env.plan_step(new_pos, current_ori)
        
        if not result['success']:
            print(f"Collision at step {t}")
            break
        
        path.append(np.concatenate([result['position'], result['orientation']]))
    
    # 评估路径
    evaluation = env.evaluate_path(path)
    print(f"Path evaluation: {evaluation}")
    
    # 获取环境信息
    env_info = env.get_environment_info()
    print(f"Environment info: {env_info}")
    
    # 渲染结果
    env.render("planned_path.png", {"planning_algorithm": "straight_line"})


if __name__ == "__main__":
    print("=== RL Usage Example ===")
    example_rl_usage()
    
    print("\n=== Path Planning Usage Example ===")
    example_path_planning_usage()