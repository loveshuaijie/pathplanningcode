import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional
import time


class APF3D:
    """
    三维空间人工势场法路径规划器
    
    特点：
    - 三维引力场和斥力场计算
    - 支持障碍物排斥
    - 局部极小值检测与逃逸
    - 路径平滑优化
    - 自适应步长调整
    """
    
    def __init__(self, env):
        """
        初始化三维APF规划器
        
        Args:
            env: MapEnv环境实例
        """
        self.env = env
        self.path = []  # 规划路径
        self.stats = {}  # 统计信息
        
        # 默认参数配置
        self.config = {
            'max_iterations': 2000,      # 最大迭代次数
            'step_size': 0.5,           # 基础步长
            'goal_threshold': 0.1,      # 目标到达阈值
            'influence_radius': 5.0,    # 障碍物影响半径
            'attractive_gain': 2.0,     # 引力增益
            'repulsive_gain': 5.0,      # 斥力增益
            'damping': 0.1,             # 阻尼系数
            'smoothing_iterations': 100, # 平滑迭代次数
            'oscillation_threshold': 5,  # 振荡检测阈值
            'adaptive_step': True,      # 是否自适应步长
            'random_escape': True,      # 是否随机逃逸局部极小值
            'escape_attempts': 10,      # 逃逸尝试次数
            'escape_step': 2.0,         # 逃逸步长
        }
    
    def plan(self, start: Optional[np.ndarray] = None,
             goal: Optional[np.ndarray] = None,
             obstacles: Optional[List] = None) -> Dict[str, Any]:
        """
        执行三维路径规划
        
        Args:
            start: 起始点 [x, y, z] 或 [x, y, z, rx, ry, rz]
            goal: 目标点 [x, y, z] 或 [x, y, z, rx, ry, rz]
            obstacles: 障碍物列表，如果为None则使用环境的障碍物
            
        Returns:
            规划结果字典
        """
        start_time = time.time()
        
        # 设置起点和目标点
        if start is not None:
            if len(start) == 3:
                self.env.startpos = start.copy()
                self.env.startges = np.zeros(3)
            else:
                self.env.startpos = start[:3].copy()
                self.env.startges = start[3:6].copy()
        
        if goal is not None:
            if len(goal) == 3:
                self.env.targetpos = goal.copy()
                self.env.targetges = np.zeros(3)
            else:
                self.env.targetpos = goal[:3].copy()
                self.env.targetges = goal[3:6].copy()
        
        # 设置障碍物
        if obstacles is not None:
            self.env.obstacles = obstacles
        
        # 重置环境
        self.env.reset(target_random_flag=False,obstacles_random_flag=False)
        
        # 初始化路径
        self.path = [self.env.nowpos.copy()]
        current_pos = self.env.nowpos.copy()
        current_ori = self.env.nowges.copy()
        
        # 统计信息
        iterations = 0
        oscillations = 0
        escape_attempts = 0
        success = False
        
        # 主规划循环
        for i in range(self.config['max_iterations']):
            iterations = i + 1
            
            # 检查是否到达目标
            if self._is_goal_reached(current_pos):
                print(f"APF: 到达目标，迭代次数: {iterations}")
                success = True
                break
            
            # 计算合力
            total_force = self._calculate_total_force_3d(current_pos)
            
            # 自适应步长调整
            step_size = self._get_adaptive_step_size(current_pos, total_force)
            
            # 计算新位置
            new_pos = current_pos + total_force * step_size
            
            # 边界检查
            new_pos = self._clamp_to_bounds(new_pos)
            
            # 尝试移动到新位置
            result = self.env.plan_step(new_pos, current_ori)
            
            if result['success']:
                current_pos = result['position'].copy()
                self.path.append(current_pos.copy())
                
                # 检测振荡（局部极小值）
                if self._detect_oscillation():
                    oscillations += 1
                    print(f"APF: 检测到振荡，次数: {oscillations}")
                    
                    if self.config['random_escape'] and oscillations > self.config['oscillation_threshold']:
                        escaped = self._escape_local_minimum(current_pos, current_ori)
                        if escaped:
                            oscillations = 0  # 重置振荡计数
                        else:
                            escape_attempts += 1
                            if escape_attempts >= self.config['escape_attempts']:
                                print("APF: 无法逃逸局部极小值")
                                break
            else:
                # 碰撞或边界，增加斥力并重试
                print("APF: 检测到碰撞，调整斥力")
                self.config['repulsive_gain'] *= 1.2
                
                # 尝试小步移动
                for _ in range(3):
                    test_force = self._calculate_total_force_3d(current_pos)
                    test_force = test_force / np.linalg.norm(test_force) if np.linalg.norm(test_force) > 0 else test_force
                    test_pos = current_pos + test_force * (step_size * 0.3)
                    test_pos = self._clamp_to_bounds(test_pos)
                    test_result = self.env.plan_step(test_pos, current_ori)
                    if test_result['success']:
                        current_pos = test_result['position'].copy()
                        self.path.append(current_pos.copy())
                        break
        
        # 路径平滑
        if len(self.path) > 3 and success:
            self.path = self._smooth_path_3d(self.path)
        
        # 计算统计信息
        planning_time = time.time() - start_time
        path_length = self._calculate_path_length_3d(self.path)
        
        self.stats = {
            'success': success,
            'iterations': iterations,
            'path_length': path_length,
            'planning_time': planning_time,
            'path_points': len(self.path),
            'oscillations_detected': oscillations,
            'escape_attempts': escape_attempts,
        }
        
        return {
            'success': success,
            'path': self.path,
            'stats': self.stats,
            'message': '成功到达目标' if success else '未到达目标'
        }
    
    def _calculate_total_force_3d(self, position: np.ndarray) -> np.ndarray:
        """
        计算三维空间中的总力（引力 + 斥力）
        
        Args:
            position: 当前位置 [x, y, z]
            
        Returns:
            总力向量 [Fx, Fy, Fz]
        """
        # 引力（指向目标）
        attractive_force = self._calculate_attractive_force_3d(position)
        
        # 斥力（远离障碍物）
        repulsive_force = self._calculate_repulsive_force_3d(position)
        
        # 合力
        total_force = attractive_force + repulsive_force
        
        # 归一化
        force_norm = np.linalg.norm(total_force)
        if force_norm > 0:
            total_force = total_force / force_norm
        
        return total_force
    
    def _calculate_attractive_force_3d(self, position: np.ndarray) -> np.ndarray:
        """
        计算三维引力
        
        Args:
            position: 当前位置
            
        Returns:
            引力向量
        """
        # 目标方向
        direction = self.env.targetpos - position
        distance = np.linalg.norm(direction)
        
        if distance < 1e-10:
            return np.zeros(3)
        
        # 线性引力场
        attractive_force = self.config['attractive_gain'] * direction
        
        return attractive_force
    
    def _calculate_repulsive_force_3d(self, position: np.ndarray) -> np.ndarray:
        """
        计算三维斥力
        
        Args:
            position: 当前位置
            
        Returns:
            斥力向量
        """
        repulsive_force = np.zeros(3)
        
        for obstacle in self.env.obstacles:
            # 获取障碍物信息
            obs_pos = obstacle.centerPoint
            obs_type = type(obstacle).__name__
            
            # 计算到障碍物的距离和方向
            direction = position - obs_pos
            distance = np.linalg.norm(direction)
            
            if distance < 1e-10:
                continue
            
            # 根据障碍物类型计算有效距离
            if obs_type == 'Sphere':
                effective_distance = max(0, distance - obstacle.radius)
            elif obs_type == 'Cylinder':
                # 计算到圆柱体中心线的距离
                cylinder_axis = np.array([0, 0, 1])  # 假设圆柱体沿Z轴
                if hasattr(obstacle, 'orientation') and obstacle.orientation is not None:
                    cylinder_axis = obstacle.orientation[:, 2]  # 使用实际方向
                
                # 计算到轴线的垂直距离
                to_center = position - obs_pos
                parallel_component = np.dot(to_center, cylinder_axis) * cylinder_axis
                perpendicular_distance = np.linalg.norm(to_center - parallel_component)
                
                # 考虑圆柱体半径
                effective_distance = max(0, perpendicular_distance - obstacle.radius)
            elif obs_type == 'Cuboid':
                # 计算到长方体表面的最小距离
                half_size = obstacle.size / 2
                dx = max(abs(position[0] - obs_pos[0]) - half_size[0], 0)
                dy = max(abs(position[1] - obs_pos[1]) - half_size[1], 0)
                dz = max(abs(position[2] - obs_pos[2]) - half_size[2], 0)
                effective_distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            else:
                effective_distance = distance
            
            # 如果在影响半径内，计算斥力
            if effective_distance < self.config['influence_radius']:
                # 斥力大小与距离成反比
                if effective_distance < 0.1:
                    effective_distance = 0.1  # 防止除零
                
                repulsive_magnitude = self.config['repulsive_gain'] * \
                                    (1/effective_distance - 1/self.config['influence_radius']) * \
                                    (1/(effective_distance**2))
                
                # 斥力方向
                if distance > 0:
                    force_direction = direction / distance
                else:
                    force_direction = np.zeros(3)
                
                repulsive_force += repulsive_magnitude * force_direction
        
        return repulsive_force
    
    def _get_adaptive_step_size(self, position: np.ndarray, force: np.ndarray) -> float:
        """
        获取自适应步长
        
        Args:
            position: 当前位置
            force: 当前受力
            
        Returns:
            自适应步长
        """
        if not self.config['adaptive_step']:
            return self.config['step_size']
        
        # 基础步长
        step_size = self.config['step_size']
        
        # 计算到目标的距离
        to_goal = self.env.targetpos - position
        distance_to_goal = np.linalg.norm(to_goal)
        
        # 根据目标距离调整步长
        if distance_to_goal < 2.0:
            step_size *= 0.5  # 接近目标时减小步长
        elif distance_to_goal > 10.0:
            step_size *= 1.5  # 远离目标时增大步长
        
        # 根据受力大小调整步长
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > 0:
            step_size *= min(2.0, 1.0 / force_magnitude)
        
        # 限制步长范围
        step_size = max(0.1, min(2.0, step_size))
        
        return step_size
    
    def _detect_oscillation(self, window_size: int = 10) -> bool:
        """
        检测路径振荡（局部极小值）
        
        Args:
            window_size: 检测窗口大小
            
        Returns:
            是否检测到振荡
        """
        if len(self.path) < window_size * 2:
            return False
        
        # 获取最近的路径点
        recent_points = self.path[-window_size:]
        
        # 计算路径点之间的总距离
        total_distance = 0
        for i in range(1, len(recent_points)):
            total_distance += np.linalg.norm(recent_points[i] - recent_points[i-1])
        
        # 计算起点到终点的直线距离
        direct_distance = np.linalg.norm(recent_points[-1] - recent_points[0])
        
        # 如果路径长度远大于直线距离，可能存在振荡
        if total_distance > direct_distance * 3.0:
            return True
        
        # 检查位置变化方差
        positions = np.array(recent_points)
        variances = np.var(positions, axis=0)
        
        # 如果三个方向的变化都很小，可能陷入局部极小值
        if np.all(variances < 0.1):
            return True
        
        return False
    
    def _escape_local_minimum(self, current_pos: np.ndarray, current_ori: np.ndarray) -> bool:
        """
        尝试逃逸局部极小值
        
        Args:
            current_pos: 当前位置
            current_ori: 当前姿态
            
        Returns:
            是否成功逃逸
        """
        print("APF: 尝试逃逸局部极小值")
        
        # 尝试多个随机方向
        for attempt in range(5):
            # 生成随机方向
            random_direction = np.random.randn(3)
            random_direction = random_direction / np.linalg.norm(random_direction)
            
            # 计算逃逸位置
            escape_pos = current_pos + random_direction * self.config['escape_step']
            escape_pos = self._clamp_to_bounds(escape_pos)
            
            # 尝试移动
            result = self.env.plan_step(escape_pos, current_ori)
            
            if result['success']:
                self.path.append(result['position'].copy())
                print(f"APF: 成功逃逸局部极小值，尝试次数: {attempt+1}")
                return True
        
        # 尝试向目标方向移动
        to_goal = self.env.targetpos - current_pos
        if np.linalg.norm(to_goal) > 0:
            to_goal = to_goal / np.linalg.norm(to_goal)
            escape_pos = current_pos + to_goal * self.config['escape_step']
            escape_pos = self._clamp_to_bounds(escape_pos)
            
            result = self.env.plan_step(escape_pos, current_ori)
            if result['success']:
                self.path.append(result['position'].copy())
                print("APF: 使用目标方向逃逸成功")
                return True
        
        print("APF: 逃逸局部极小值失败")
        return False
    
    def _smooth_path_3d(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """
        平滑三维路径
        
        Args:
            path: 原始路径
            
        Returns:
            平滑后的路径
        """
        if len(path) <= 2:
            return path
        
        # 使用梯度下降法平滑路径
        smoothed_path = path.copy()
        alpha = 0.3  # 平滑系数
        beta = 0.1   # 保真系数
        
        for _ in range(self.config['smoothing_iterations']):
            new_path = smoothed_path.copy()
            
            for i in range(1, len(smoothed_path) - 1):
                # 平滑项：使点靠近相邻点的中心
                smooth_term = (smoothed_path[i-1] + smoothed_path[i+1]) / 2 - smoothed_path[i]
                
                # 保真项：保持接近原始路径
                fidelity_term = path[i] - smoothed_path[i]
                
                # 碰撞惩罚项
                collision_penalty = np.zeros(3)
                for obstacle in self.env.obstacles:
                    obs_pos = obstacle.centerPoint
                    distance = np.linalg.norm(smoothed_path[i] - obs_pos)
                    if distance < self.config['influence_radius']:
                        direction = smoothed_path[i] - obs_pos
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                        collision_penalty += direction * (1.0 / max(distance, 0.1))
                
                # 更新位置
                new_path[i] = smoothed_path[i] + alpha * smooth_term + beta * fidelity_term - 0.1 * collision_penalty
            
            smoothed_path = new_path
        
        return smoothed_path
    
    def _is_goal_reached(self, position: np.ndarray) -> bool:
        """检查是否到达目标"""
        distance = np.linalg.norm(position - self.env.targetpos)
        return distance < self.config['goal_threshold']
    
    def _clamp_to_bounds(self, position: np.ndarray) -> np.ndarray:
        """限制位置在边界内"""
        x = np.clip(position[0], self.env.xrange[0], self.env.xrange[1])
        y = np.clip(position[1], self.env.yrange[0], self.env.yrange[1])
        z = np.clip(position[2], self.env.zrange[0], self.env.zrange[1])
        return np.array([x, y, z])
    
    def _calculate_path_length_3d(self, path: List[np.ndarray]) -> float:
        """计算三维路径长度"""
        if len(path) < 2:
            return 0
        
        length = 0
        for i in range(1, len(path)):
            length += np.linalg.norm(path[i] - path[i-1])
        
        return length
    
    def visualize(self, save_path: str = "apf_3d_path.png"):
        """可视化规划结果"""
        info = {
            'algorithm': 'APF-3D',
            'iterations': self.stats.get('iterations', 0),
            'path_length': self.stats.get('path_length', 0),
            'planning_time': self.stats.get('planning_time', 0),
            'success': self.stats.get('success', False),
        }
        
        self.env.render(save_path, info)
    
    def get_force_field(self, resolution: float = 1.0) -> Dict[str, Any]:
        """
        获取力场可视化数据
        
        Args:
            resolution: 网格分辨率
            
        Returns:
            力场数据
        """
        # 创建网格
        x = np.arange(self.env.xrange[0], self.env.xrange[1], resolution)
        y = np.arange(self.env.yrange[0], self.env.yrange[1], resolution)
        z = np.arange(self.env.zrange[0], self.env.zrange[1], resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 计算每个网格点的力
        forces = []
        positions = []
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    pos = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                    force = self._calculate_total_force_3d(pos)
                    forces.append(force)
                    positions.append(pos)
        
        return {
            'positions': np.array(positions),
            'forces': np.array(forces),
            'resolution': resolution,
            'bounds': {
                'x': self.env.xrange,
                'y': self.env.yrange,
                'z': self.env.zrange
            }
        }