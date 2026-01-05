import numpy as np
import random
import math
from typing import List, Tuple, Dict, Any, Optional
import time
from collections import deque


class RRTNode3D:
    """三维RRT树节点"""
    
    def __init__(self, position: np.ndarray, orientation: Optional[np.ndarray] = None, parent=None):
        self.position = position.copy()  # 位置 [x, y, z]
        self.orientation = orientation.copy() if orientation is not None else None  # 姿态 [rx, ry, rz]
        self.parent = parent  # 父节点
        self.children = []  # 子节点
        self.cost = 0.0  # 从根节点到本节点的累计成本
        
        # 用于RRT*的邻居信息
        self.neighbors = []
    
    def add_child(self, child_node):
        """添加子节点"""
        self.children.append(child_node)
    
    def distance_to(self, other_node) -> float:
        """计算到另一个节点的欧氏距离"""
        return np.linalg.norm(self.position - other_node.position)
    
    def __repr__(self) -> str:
        return f"RRTNode3D(pos={self.position}, cost={self.cost:.2f})"


class RRT3D:
    """
    三维空间快速随机树路径规划器
    
    特点：
    - 三维空间随机采样
    - 支持RRT和RRT*变体
    - 双向搜索（Bi-RRT）
    - 路径平滑优化
    - 动态步长调整
    """
    
    def __init__(self, env):
        """
        初始化三维RRT规划器
        
        Args:
            env: MapEnv环境实例
        """
        self.env = env
        self.tree = []  # 树节点列表
        self.start_node = None  # 起点节点
        self.goal_node = None  # 目标节点
        self.path = []  # 规划路径
        self.stats = {}  # 统计信息
        
        # 配置参数
        self.config = {
            'max_iterations': 5000,      # 最大迭代次数
            'step_size': 2.0,           # 扩展步长
            'goal_bias': 0.1,           # 目标偏向概率
            'goal_threshold': 1.0,      # 目标到达阈值
            'neighbor_radius': 3.0,     # 邻居搜索半径
            'path_smoothing': True,     # 是否平滑路径
            'smoothing_iterations': 50, # 平滑迭代次数
            'use_rrt_star': True,       # 是否使用RRT*
            'use_bidirectional': True,  # 是否使用双向RRT
            'rewire_radius': 4.0,       # RRT*重连半径
            'adaptive_step': True,      # 是否自适应步长
            'collision_check_points': 10, # 碰撞检查点数
            'goal_sample_rate': 0.05,   # 目标采样率
        }
        
        # 双向RRT相关
        self.start_tree = []  # 从起点生长的树
        self.goal_tree = []   # 从目标生长的树
        self.connection_node = None  # 连接节点
    
    def plan(self, start: Optional[np.ndarray] = None,
             goal: Optional[np.ndarray] = None,
             obstacles: Optional[List] = None) -> Dict[str, Any]:
        """
        执行三维RRT路径规划
        
        Args:
            start: 起始点 [x, y, z] 或 [x, y, z, rx, ry, rz]
            goal: 目标点 [x, y, z] 或 [x, y, z, rx, ry, rz]
            obstacles: 障碍物列表
            
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
        self.env.reset(generate_obstacles=False)
        
        # 初始化树
        self._initialize_trees()
        
        # 执行规划
        if self.config['use_bidirectional']:
            success = self._bidirectional_rrt()
        else:
            success = self._standard_rrt()
        
        # 提取路径
        if success:
            self._extract_path()
            
            # 路径平滑
            if self.config['path_smoothing'] and len(self.path) > 3:
                self.path = self._smooth_path_3d(self.path)
        
        # 计算统计信息
        planning_time = time.time() - start_time
        path_length = self._calculate_path_length_3d(self.path)
        
        self.stats = {
            'success': success,
            'iterations': len(self.tree),
            'path_length': path_length,
            'planning_time': planning_time,
            'path_points': len(self.path),
            'tree_nodes': len(self.tree),
            'collision_checks': self.env.get_planning_stats()['collision_checks'],
        }
        
        return {
            'success': success,
            'path': self.path,
            'raw_tree': self.tree,
            'stats': self.stats,
            'message': '成功找到路径' if success else '未找到路径'
        }
    
    def _initialize_trees(self):
        """初始化RRT树"""
        self.tree = []
        self.start_tree = []
        self.goal_tree = []
        self.connection_node = None
        
        # 创建起点节点
        self.start_node = RRTNode3D(
            position=self.env.startpos.copy(),
            orientation=self.env.startges.copy(),
            parent=None
        )
        self.start_node.cost = 0.0
        
        # 创建目标节点
        self.goal_node = RRTNode3D(
            position=self.env.targetpos.copy(),
            orientation=self.env.targetges.copy(),
            parent=None
        )
        
        # 添加到树中
        self.tree.append(self.start_node)
        
        if self.config['use_bidirectional']:
            self.start_tree.append(self.start_node)
            self.goal_tree.append(self.goal_node)
        else:
            # 单方向RRT，目标节点只用于检查
            pass
    
    def _standard_rrt(self) -> bool:
        """执行标准RRT算法"""
        for iteration in range(self.config['max_iterations']):
            # 1. 采样随机点
            random_point = self._sample_random_point_3d()
            
            # 2. 寻找最近邻节点
            nearest_node = self._find_nearest_neighbor_3d(random_point)
            
            # 3. 向随机点方向扩展
            new_node = self._extend_towards_3d(nearest_node, random_point)
            
            if new_node is not None:
                # 4. 添加新节点到树中
                self.tree.append(new_node)
                nearest_node.add_child(new_node)
                
                # 5. RRT*重连（如果启用）
                if self.config['use_rrt_star']:
                    self._rewire_tree_3d(new_node)
                
                # 6. 检查是否到达目标
                if self._check_goal_reached_3d(new_node):
                    # 连接到目标节点
                    if self._connect_to_goal(new_node):
                        print(f"RRT: 在迭代 {iteration} 时到达目标")
                        return True
            
            # 进度显示
            if iteration % 1000 == 0 and iteration > 0:
                print(f"RRT: 迭代 {iteration}/{self.config['max_iterations']}, 节点数: {len(self.tree)}")
        
        print(f"RRT: 达到最大迭代次数 {self.config['max_iterations']}，未找到路径")
        return False
    
    def _bidirectional_rrt(self) -> bool:
        """执行双向RRT算法"""
        for iteration in range(self.config['max_iterations']):
            # 交替扩展两棵树
            if iteration % 2 == 0:
                # 扩展起点树
                tree_from = self.start_tree
                tree_to = self.goal_tree
                is_start_tree = True
            else:
                # 扩展目标树
                tree_from = self.goal_tree
                tree_to = self.start_tree
                is_start_tree = False
            
            # 1. 采样随机点
            random_point = self._sample_random_point_3d()
            
            # 2. 寻找最近邻节点
            nearest_node = self._find_nearest_neighbor_in_tree(random_point, tree_from)
            
            # 3. 向随机点方向扩展
            new_node = self._extend_towards_3d(nearest_node, random_point)
            
            if new_node is not None:
                # 4. 添加新节点到树中
                tree_from.append(new_node)
                self.tree.append(new_node)
                nearest_node.add_child(new_node)
                
                # 5. 尝试连接到另一棵树
                nearest_in_other = self._find_nearest_neighbor_in_tree(new_node.position, tree_to)
                
                if nearest_in_other is not None:
                    # 尝试直接连接
                    if self._try_connect_nodes_3d(new_node, nearest_in_other):
                        # 记录连接节点
                        self.connection_node = new_node
                        
                        print(f"Bi-RRT: 在迭代 {iteration} 时连接两棵树")
                        return True
                
                # 6. RRT*重连
                if self.config['use_rrt_star']:
                    self._rewire_tree_in_tree(new_node, tree_from)
            
            # 进度显示
            if iteration % 1000 == 0 and iteration > 0:
                print(f"Bi-RRT: 迭代 {iteration}/{self.config['max_iterations']}, 总节点数: {len(self.tree)}")
        
        print(f"Bi-RRT: 达到最大迭代次数 {self.config['max_iterations']}，未找到路径")
        return False
    
    def _sample_random_point_3d(self) -> np.ndarray:
        """
        三维空间随机采样
        
        Returns:
            随机点坐标 [x, y, z]
        """
        # 目标偏向采样
        if random.random() < self.config['goal_bias']:
            return self.env.targetpos.copy()
        
        # 目标区域采样（以一定概率在目标附近采样）
        if random.random() < self.config['goal_sample_rate']:
            # 在目标附近采样
            radius = 3.0
            offset = np.random.uniform(-radius, radius, 3)
            return self.env.targetpos + offset
        
        # 在环境边界内均匀采样
        x = random.uniform(self.env.xrange[0], self.env.xrange[1])
        y = random.uniform(self.env.yrange[0], self.env.yrange[1])
        z = random.uniform(self.env.zrange[0], self.env.zrange[1])
        
        return np.array([x, y, z])
    
    def _find_nearest_neighbor_3d(self, point: np.ndarray) -> RRTNode3D:
        """
        在树中寻找最近邻节点
        
        Args:
            point: 查询点
            
        Returns:
            最近邻节点
        """
        if len(self.tree) == 0:
            return None
        
        min_distance = float('inf')
        nearest_node = None
        
        for node in self.tree:
            distance = np.linalg.norm(node.position - point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _find_nearest_neighbor_in_tree(self, point: np.ndarray, tree: List[RRTNode3D]) -> Optional[RRTNode3D]:
        """
        在指定树中寻找最近邻节点
        
        Args:
            point: 查询点
            tree: 树节点列表
            
        Returns:
            最近邻节点
        """
        if len(tree) == 0:
            return None
        
        min_distance = float('inf')
        nearest_node = None
        
        for node in tree:
            distance = np.linalg.norm(node.position - point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _extend_towards_3d(self, from_node: RRTNode3D, to_point: np.ndarray) -> Optional[RRTNode3D]:
        """
        从节点向目标点扩展
        
        Args:
            from_node: 起始节点
            to_point: 目标点
            
        Returns:
            新节点（如果扩展成功）
        """
        # 计算方向
        direction = to_point - from_node.position
        distance = np.linalg.norm(direction)
        
        if distance < 1e-10:
            return None
        
        # 归一化方向
        direction = direction / distance
        
        # 自适应步长
        step_size = self.config['step_size']
        if self.config['adaptive_step']:
            # 根据到目标的距离调整步长
            to_goal = self.env.targetpos - from_node.position
            distance_to_goal = np.linalg.norm(to_goal)
            
            if distance_to_goal < 5.0:
                step_size *= 0.5
            elif distance_to_goal > 20.0:
                step_size *= 2.0
        
        # 限制步长
        step_size = min(step_size, distance)
        
        # 计算新位置
        new_position = from_node.position + direction * step_size
        
        # 检查路径是否可行
        if self._is_path_clear_3d(from_node.position, new_position):
            # 创建新节点
            new_node = RRTNode3D(
                position=new_position,
                orientation=from_node.orientation,  # 使用父节点姿态
                parent=from_node
            )
            new_node.cost = from_node.cost + step_size
            
            return new_node
        
        return None
    
    def _is_path_clear_3d(self, start: np.ndarray, end: np.ndarray) -> bool:
        """
        检查三维路径段是否无碰撞
        
        Args:
            start: 起点 [x, y, z]
            end: 终点 [x, y, z]
            
        Returns:
            是否无碰撞
        """
        # 计算路径长度
        path_length = np.linalg.norm(end - start)
        
        # 确定检查点数
        num_points = max(2, int(path_length * self.config['collision_check_points']))
        
        for i in range(num_points + 1):
            t = i / num_points
            test_point = start * (1 - t) + end * t
            
            # 检查边界
            if not self.env._check_bounds(test_point):
                return False
            
            # 检查碰撞（使用当前姿态）
            temp_tool = self.env.moving_tool.__class__(
                height=self.env.tool_size[0],
                radius=self.env.tool_size[1],
                centerPoint=test_point,
                orientation=self.env.moving_tool.orientation
            )
            
            if self.env._check_collision(temp_tool):
                return False
        
        return True
    
    def _rewire_tree_3d(self, new_node: RRTNode3D):
        """
        RRT*重连：优化树结构
        
        Args:
            new_node: 新添加的节点
        """
        # 寻找附近节点
        nearby_nodes = self._find_nearby_nodes_3d(new_node.position, self.config['rewire_radius'])
        
        # 检查是否可以通过新节点优化路径
        for node in nearby_nodes:
            if node == new_node or node == new_node.parent:
                continue
            
            # 计算通过新节点的成本
            new_cost = new_node.cost + new_node.distance_to(node)
            
            # 如果新路径成本更低
            if new_cost < node.cost:
                # 检查新路径是否可行
                if self._is_path_clear_3d(new_node.position, node.position):
                    # 从原父节点移除
                    if node.parent:
                        node.parent.children.remove(node)
                    
                    # 重连到新节点
                    node.parent = new_node
                    new_node.add_child(node)
                    
                    # 更新成本
                    cost_diff = new_cost - node.cost
                    node.cost = new_cost
                    
                    # 递归更新子节点成本
                    self._update_costs_recursive(node, cost_diff)
    
    def _rewire_tree_in_tree(self, new_node: RRTNode3D, tree: List[RRTNode3D]):
        """
        在指定树中执行RRT*重连
        
        Args:
            new_node: 新节点
            tree: 树节点列表
        """
        # 寻找附近节点（在同一棵树中）
        nearby_nodes = []
        for node in tree:
            if node != new_node and np.linalg.norm(node.position - new_node.position) < self.config['rewire_radius']:
                nearby_nodes.append(node)
        
        # 优化连接
        for node in nearby_nodes:
            if node == new_node.parent:
                continue
            
            new_cost = new_node.cost + new_node.distance_to(node)
            
            if new_cost < node.cost:
                if self._is_path_clear_3d(new_node.position, node.position):
                    if node.parent:
                        node.parent.children.remove(node)
                    
                    node.parent = new_node
                    new_node.add_child(node)
                    
                    cost_diff = new_cost - node.cost
                    node.cost = new_cost
                    self._update_costs_recursive(node, cost_diff)
    
    def _find_nearby_nodes_3d(self, position: np.ndarray, radius: float) -> List[RRTNode3D]:
        """
        寻找指定位置附近的节点
        
        Args:
            position: 中心位置
            radius: 搜索半径
            
        Returns:
            附近节点列表
        """
        nearby_nodes = []
        
        for node in self.tree:
            if np.linalg.norm(node.position - position) < radius:
                nearby_nodes.append(node)
        
        return nearby_nodes
    
    def _update_costs_recursive(self, node: RRTNode3D, cost_diff: float):
        """
        递归更新节点及其子节点的成本
        
        Args:
            node: 起始节点
            cost_diff: 成本变化量
        """
        for child in node.children:
            child.cost += cost_diff
            self._update_costs_recursive(child, cost_diff)
    
    def _check_goal_reached_3d(self, node: RRTNode3D) -> bool:
        """
        检查节点是否到达目标
        
        Args:
            node: 待检查节点
            
        Returns:
            是否到达目标
        """
        distance = np.linalg.norm(node.position - self.env.targetpos)
        return distance < self.config['goal_threshold']
    
    def _connect_to_goal(self, node: RRTNode3D) -> bool:
        """
        尝试将节点连接到目标
        
        Args:
            node: 当前节点
            
        Returns:
            是否成功连接
        """
        if self._is_path_clear_3d(node.position, self.env.targetpos):
            # 创建目标节点并连接到当前节点
            goal_node = RRTNode3D(
                position=self.env.targetpos.copy(),
                orientation=self.env.targetges.copy(),
                parent=node
            )
            goal_node.cost = node.cost + node.distance_to(goal_node)
            
            node.add_child(goal_node)
            self.tree.append(goal_node)
            
            return True
        
        return False
    
    def _try_connect_nodes_3d(self, node1: RRTNode3D, node2: RRTNode3D) -> bool:
        """
        尝试连接两个节点
        
        Args:
            node1: 节点1
            node2: 节点2
            
        Returns:
            是否成功连接
        """
        return self._is_path_clear_3d(node1.position, node2.position)
    
    def _extract_path(self):
        """从树中提取路径"""
        # 找到最接近目标的节点
        goal_node = None
        min_distance = float('inf')
        
        for node in self.tree:
            distance = np.linalg.norm(node.position - self.env.targetpos)
            if distance < min_distance:
                min_distance = distance
                goal_node = node
        
        if goal_node is None:
            self.path = []
            return
        
        # 双向RRT需要特殊处理
        if self.config['use_bidirectional'] and self.connection_node is not None:
            # 从起点到连接节点
            path_from_start = []
            current = self.connection_node
            while current is not None:
                path_from_start.append(current)
                current = current.parent
            
            path_from_start.reverse()
            
            # 从目标到连接节点
            path_from_goal = []
            # 需要在目标树中找到连接节点的对应节点
            connection_in_goal_tree = None
            for node in self.goal_tree:
                if np.linalg.norm(node.position - self.connection_node.position) < 0.1:
                    connection_in_goal_tree = node
                    break
            
            if connection_in_goal_tree:
                current = connection_in_goal_tree.parent
                while current is not None:
                    path_from_goal.append(current)
                    current = current.parent
            
            # 合并路径
            self.path = []
            for node in path_from_start:
                self.path.append(np.concatenate([node.position, node.orientation if node.orientation is not None else np.zeros(3)]))
            
            for node in reversed(path_from_goal):
                self.path.append(np.concatenate([node.position, node.orientation if node.orientation is not None else np.zeros(3)]))
        else:
            # 标准RRT路径提取
            self.path = []
            current = goal_node
            
            while current is not None:
                if current.orientation is not None:
                    path_point = np.concatenate([current.position, current.orientation])
                else:
                    path_point = current.position
                self.path.append(path_point)
                current = current.parent
            
            self.path.reverse()
    
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
        
        # 将路径转换为位置列表
        positions = []
        for point in path:
            if len(point) > 3:
                positions.append(point[:3])
            else:
                positions.append(point)
        
        # 路径拉直算法
        smoothed_positions = [positions[0]]
        i = 0
        
        while i < len(positions) - 1:
            # 寻找可以从当前点直接到达的最远点
            j = len(positions) - 1
            found = False
            
            while j > i + 1:
                if self._is_path_clear_3d(positions[i], positions[j]):
                    smoothed_positions.append(positions[j])
                    i = j
                    found = True
                    break
                j -= 1
            
            if not found:
                smoothed_positions.append(positions[i + 1])
                i += 1
        
        # 转换回完整路径点
        smoothed_path = []
        for i, pos in enumerate(smoothed_positions):
            # 使用原始路径中最近点的姿态
            nearest_idx = self._find_nearest_point_index(pos, positions)
            if nearest_idx < len(path) and len(path[nearest_idx]) > 3:
                smoothed_path.append(np.concatenate([pos, path[nearest_idx][3:6]]))
            else:
                smoothed_path.append(pos)
        
        return smoothed_path
    
    def _find_nearest_point_index(self, point: np.ndarray, points_list: List[np.ndarray]) -> int:
        """在点列表中寻找最近点的索引"""
        min_distance = float('inf')
        nearest_idx = 0
        
        for i, p in enumerate(points_list):
            distance = np.linalg.norm(p - point)
            if distance < min_distance:
                min_distance = distance
                nearest_idx = i
        
        return nearest_idx
    
    def _calculate_path_length_3d(self, path: List[np.ndarray]) -> float:
        """计算三维路径长度"""
        if len(path) < 2:
            return 0
        
        length = 0
        for i in range(1, len(path)):
            if len(path[i]) > 3:
                p1 = path[i-1][:3]
                p2 = path[i][:3]
            else:
                p1 = path[i-1]
                p2 = path[i]
            
            length += np.linalg.norm(p2 - p1)
        
        return length
    
    def visualize(self, save_path: str = "rrt_3d_path.png", show_tree: bool = False):
        """可视化规划结果"""
        info = {
            'algorithm': 'RRT-3D',
            'iterations': self.stats.get('iterations', 0),
            'path_length': self.stats.get('path_length', 0),
            'planning_time': self.stats.get('planning_time', 0),
            'success': self.stats.get('success', False),
            'tree_nodes': self.stats.get('tree_nodes', 0),
        }
        
        if show_tree:
            # 添加树结构信息用于可视化
            tree_edges = []
            for node in self.tree:
                if node.parent:
                    tree_edges.append({
                        'start': node.parent.position.tolist(),
                        'end': node.position.tolist()
                    })
            info['tree_edges'] = tree_edges
        
        self.env.render(save_path, info)
    
    def get_tree_data(self) -> Dict[str, Any]:
        """获取树结构数据"""
        nodes = []
        edges = []
        
        for node in self.tree:
            nodes.append({
                'position': node.position.tolist(),
                'cost': node.cost,
                'parent': node.parent.position.tolist() if node.parent else None
            })
            
            if node.parent:
                edges.append({
                    'start': node.parent.position.tolist(),
                    'end': node.position.tolist(),
                    'cost': node.cost - node.parent.cost
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'start': self.env.startpos.tolist(),
            'goal': self.env.targetpos.tolist()
        }