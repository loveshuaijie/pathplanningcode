import numpy as np
import yaml
from typing import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import random
import math
import os
import re


# ==================== 配置文件加载 ====================

def loadYaml(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


# ==================== 三维几何计算 ====================

def calculate_distance(point1, point2):
    """计算三维空间两点间的欧氏距离"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)


def calculate_angle(vec1, vec2):
    """计算两个三维向量之间的夹角（弧度）"""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # 防止数值误差导致cos_angle超出[-1, 1]范围
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def calculate_path_length(path: List[np.ndarray]) -> float:
    """计算三维路径的总长度"""
    if len(path) < 2:
        return 0
    
    length = 0
    for i in range(1, len(path)):
        # 支持带有姿态的路径点
        if len(path[i]) > 3:
            p1 = path[i-1][:3]
            p2 = path[i][:3]
        else:
            p1 = path[i-1]
            p2 = path[i]
        length += calculate_distance(p1, p2)
    
    return length


def calculate_path_smoothness(path: List[np.ndarray]) -> float:
    """计算路径的平滑度（角度变化总和）"""
    if len(path) < 3:
        return 0
    
    total_angle = 0
    for i in range(1, len(path) - 1):
        # 获取三个连续点
        if len(path[i-1]) > 3:
            p1 = path[i-1][:3]
            p2 = path[i][:3]
            p3 = path[i+1][:3]
        else:
            p1 = path[i-1]
            p2 = path[i]
            p3 = path[i+1]
        
        # 计算两个向量
        v1 = p2 - p1
        v2 = p3 - p2
        
        # 计算夹角
        if np.linalg.norm(v1) > 1e-5 and np.linalg.norm(v2) > 1e-5:
            angle = calculate_angle(v1, v2)
            total_angle += angle
    
    return total_angle


def calculate_path_safety(path: List[np.ndarray], obstacles: List, safe_distance: float) -> float:
    """计算路径的安全性（平均距离障碍物的距离）"""
    if len(path) == 0:
        return 0
    
    total_safety = 0
    points_checked = 0
    
    for point in path:
        # 提取位置
        if len(point) > 3:
            pos = point[:3]
        else:
            pos = point
        
        # 计算到所有障碍物的最小距离
        min_distance = float('inf')
        for obstacle in obstacles:
            if hasattr(obstacle, 'centerPoint'):
                distance = calculate_distance(pos, obstacle.centerPoint)
                if hasattr(obstacle, 'radius'):
                    distance -= obstacle.radius  # 考虑障碍物半径
                min_distance = min(min_distance, distance)
        
        if min_distance < float('inf'):
            # 使用安全距离进行归一化
            safety_score = min(1.0, min_distance / safe_distance)
            total_safety += safety_score
            points_checked += 1
    
    return total_safety / points_checked if points_checked > 0 else 0


# ==================== 路径平滑算法 ====================

def smooth_path_gradient_descent(path: List[np.ndarray], 
                                 alpha: float = 0.3, 
                                 beta: float = 0.1,
                                 iterations: int = 100) -> List[np.ndarray]:
    """
    使用梯度下降法平滑路径
    
    Args:
        path: 原始路径
        alpha: 平滑系数（控制路径平滑程度）
        beta: 保真系数（控制与原始路径的接近程度）
        iterations: 迭代次数
        
    Returns:
        平滑后的路径
    """
    if len(path) <= 2:
        return path.copy()
    
    # 将路径转换为位置数组
    positions = []
    orientations = []
    
    for point in path:
        if len(point) > 3:
            positions.append(point[:3])
            orientations.append(point[3:6])
        else:
            positions.append(point)
            orientations.append(None)
    
    smoothed_positions = np.array(positions, dtype=np.float32).copy()
    original_positions = np.array(positions, dtype=np.float32)
    
    for _ in range(iterations):
        new_positions = smoothed_positions.copy()
        
        for i in range(1, len(smoothed_positions) - 1):
            # 平滑项：使点靠近相邻点的中心
            smooth_term = (smoothed_positions[i-1] + smoothed_positions[i+1]) / 2 - smoothed_positions[i]
            
            # 保真项：保持接近原始路径
            fidelity_term = original_positions[i] - smoothed_positions[i]
            
            # 更新位置
            new_positions[i] = smoothed_positions[i] + alpha * smooth_term + beta * fidelity_term
        
        smoothed_positions = new_positions
    
    # 重建路径点（保持原始姿态）
    smoothed_path = []
    for i, pos in enumerate(smoothed_positions):
        if orientations[i] is not None:
            smoothed_path.append(np.concatenate([pos, orientations[i]]))
        else:
            smoothed_path.append(pos)
    
    return smoothed_path


def smooth_path_straightening(path: List[np.ndarray], 
                             collision_check_func: Callable = None,
                             max_iterations: int = 50) -> List[np.ndarray]:
    """
    使用路径拉直法平滑路径
    
    Args:
        path: 原始路径
        collision_check_func: 碰撞检查函数，接受起点和终点返回是否碰撞
        max_iterations: 最大迭代次数
        
    Returns:
        平滑后的路径
    """
    if len(path) <= 2 or collision_check_func is None:
        return path.copy()
    
    # 提取位置
    positions = []
    for point in path:
        if len(point) > 3:
            positions.append(point[:3])
        else:
            positions.append(point)
    
    smoothed_positions = [positions[0]]
    i = 0
    
    while i < len(positions) - 1:
        # 寻找可以从当前点直接到达的最远点
        j = len(positions) - 1
        found = False
        
        while j > i + 1:
            # 检查直接连接是否可行
            if not collision_check_func(positions[i], positions[j]):
                smoothed_positions.append(positions[j])
                i = j
                found = True
                break
            j -= 1
        
        if not found:
            smoothed_positions.append(positions[i + 1])
            i += 1
    
    # 重建路径点
    smoothed_path = []
    for pos in smoothed_positions:
        # 找到原始路径中最近点的姿态
        nearest_idx = np.argmin([calculate_distance(pos, p[:3] if len(p) > 3 else p) for p in path])
        if len(path[nearest_idx]) > 3:
            smoothed_path.append(np.concatenate([pos, path[nearest_idx][3:6]]))
        else:
            smoothed_path.append(pos)
    
    return smoothed_path


# ==================== 坐标转换 ====================

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角（ZYX 顺序，弧度）转换为旋转矩阵
    
    参数:
    roll (float): 绕 Z 轴的 Roll 角（弧度）
    pitch (float): 绕 Y 轴的 Pitch 角（弧度）
    yaw (float): 绕 X 轴的 Yaw 角（弧度）
    
    返回:
    3x3 numpy.ndarray: 旋转矩阵
    """
    # 计算各轴旋转矩阵
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(yaw), -np.sin(yaw)],
        [0, np.sin(yaw), np.cos(yaw)]
    ])
    
    # 总旋转矩阵（ZYX 顺序）
    # 注意：矩阵相乘顺序为逆序（Rx * Ry * Rz）
    return Rx @ Ry @ Rz


def rotation_matrix_to_euler(R):
    """
    将旋转矩阵转换为欧拉角（ZYX顺序）
    
    参数:
    R: 3x3旋转矩阵
    
    返回:
    (roll, pitch, yaw): 欧拉角（弧度）
    """
    # 检查旋转矩阵是否有效
    if np.abs(R[2, 0]) != 1:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    else:
        # 万向锁情况
        yaw = 0
        if R[2, 0] == -1:
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    
    return roll, pitch, yaw


def quaternion_to_rotation_matrix(q):
    """
    四元数转换为旋转矩阵
    
    参数:
    q: 四元数 [w, x, y, z]
    
    返回:
    3x3旋转矩阵
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


# ==================== 可视化函数 ====================

def plot_3d_path(points, bounds, obstacles, title="3D Path", show=True):
    """
    绘制三维路径（包含起点、终点和中间点）并设置边界
    
    参数:
    points -- 路径点列表，格式为 [(x1, y1, z1), (x2, y2, z2), ...]
            第一个点为起点，最后一个点为终点，中间为路径点
    bounds -- 三维边界 [xmin, xmax, ymin, ymax, zmin, zmax]
    obstacles -- 障碍物列表
    title -- 图表标题
    show -- 是否显示图表
    """
    # 检查是否有足够的点
    if len(points) < 2:
        print("需要至少两个点才能绘制路径")
        return
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取坐标
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    
    # 绘制路径
    ax.plot(x, y, z, 
            'o-', color='b', markersize=8, linewidth=0.5, 
            label=f'Path with {len(points)} points')
    
    # 标记起点、终点和中间点
    ax.text(x[0], y[0], z[0], 'Start', color='g', fontsize=12, weight='bold')
    ax.text(x[-1], y[-1], z[-1], 'End', color='r', fontsize=12, weight='bold')
    
    # # 标记中间点
    # for i, point in enumerate(points[1:-1]):
    #     ax.text(point[0], point[1], point[2], 
    #             f'P{i+1}', color='purple', fontsize=10)
    
    # 设置边界
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])
    
    # 设置标签和标题
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # 添加图例和网格
    ax.legend(loc='best')
    ax.grid(True)
    
    # 画出障碍物
    for obstacle in obstacles:
        if hasattr(obstacle, 'draw'):
            obstacle.draw(ax)
        else:
            draw_obstacle_3d(ax, obstacle)
    
    # 显示图形
    plt.tight_layout()
    if show:
        plt.show()
    
    return fig, ax


def plot_3d_path_with_tree(points, tree_data, bounds, obstacles, title="3D Path with RRT Tree", show=True):
    """
    绘制三维路径和RRT树结构
    
    参数:
    points -- 路径点列表
    tree_data -- 树数据，包含edges和nodes
    bounds -- 三维边界
    obstacles -- 障碍物列表
    title -- 图表标题
    show -- 是否显示图表
    """
    if len(points) < 2:
        print("需要至少两个点才能绘制路径")
        return
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取路径坐标
    x = [p[0] if len(p) > 3 else p[0] for p in points]
    y = [p[1] if len(p) > 3 else p[1] for p in points]
    z = [p[2] if len(p) > 3 else p[2] for p in points]
    
    # 绘制树结构
    if tree_data and 'edges' in tree_data:
        for edge in tree_data['edges']:
            if 'start' in edge and 'end' in edge:
                start = edge['start']
                end = edge['end']
                ax.plot([start[0], end[0]], 
                        [start[1], end[1]], 
                        [start[2], end[2]], 
                        'gray', alpha=0.2, linewidth=0.5)
    
    # 绘制路径
    ax.plot(x, y, z, 
            'o-', color='b', markersize=8, linewidth=2, 
            label=f'Path with {len(points)} points')
    
    # 标记起点、终点
    ax.text(x[0], y[0], z[0], 'Start', color='g', fontsize=12, weight='bold')
    ax.text(x[-1], y[-1], z[-1], 'End', color='r', fontsize=12, weight='bold')
    ax.plot([x[0]], [y[0]], [z[0]], 'g*', markersize=15, label='Start')
    ax.plot([x[-1]], [y[-1]], [z[-1]], 'r*', markersize=15, label='End')
    
    # 设置边界
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])
    
    # 设置标签和标题
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # 添加图例和网格
    ax.legend(loc='best')
    ax.grid(True)
    
    # 画出障碍物
    for obstacle in obstacles:
        if hasattr(obstacle, 'draw'):
            obstacle.draw(ax)
        else:
            draw_obstacle_3d(ax, obstacle)
    
    # 显示图形
    plt.tight_layout()
    if show:
        plt.show()
    
    return fig, ax


def draw_obstacle_3d(ax, obstacle, alpha=0.3):
    """绘制三维障碍物"""
    if hasattr(obstacle, 'centerPoint'):
        center = obstacle.centerPoint
        
        if hasattr(obstacle, 'radius'):
            # 绘制球体
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + obstacle.radius * np.cos(u) * np.sin(v)
            y = center[1] + obstacle.radius * np.sin(u) * np.sin(v)
            z = center[2] + obstacle.radius * np.cos(v)
            ax.plot_surface(x, y, z, color='gray', alpha=alpha)
        
        elif hasattr(obstacle, 'size'):
            # 绘制长方体
            size = obstacle.size
            half_size = size / 2
            
            # 定义长方体的8个顶点
            vertices = np.array([
                [center[0]-half_size[0], center[1]-half_size[1], center[2]-half_size[2]],
                [center[0]+half_size[0], center[1]-half_size[1], center[2]-half_size[2]],
                [center[0]+half_size[0], center[1]+half_size[1], center[2]-half_size[2]],
                [center[0]-half_size[0], center[1]+half_size[1], center[2]-half_size[2]],
                [center[0]-half_size[0], center[1]-half_size[1], center[2]+half_size[2]],
                [center[0]+half_size[0], center[1]-half_size[1], center[2]+half_size[2]],
                [center[0]+half_size[0], center[1]+half_size[1], center[2]+half_size[2]],
                [center[0]-half_size[0], center[1]+half_size[1], center[2]+half_size[2]]
            ])
            
            # 定义长方体的6个面
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
                [vertices[1], vertices[2], vertices[6], vertices[5]]   # 右面
            ]
            
            # 绘制每个面
            for face in faces:
                poly = Poly3DCollection([face], alpha=alpha, color='gray')
                ax.add_collection3d(poly)
        
        elif hasattr(obstacle, 'height') and hasattr(obstacle, 'radius'):
            # 绘制圆柱体（近似为多面体）
            height = obstacle.height
            radius = obstacle.radius
            
            # 圆柱体底部和顶部的圆心
            bottom_center = np.array([center[0], center[1], center[2] - height/2])
            top_center = np.array([center[0], center[1], center[2] + height/2])
            
            # 生成圆柱体的侧面
            theta = np.linspace(0, 2*np.pi, 20)
            x_bottom = bottom_center[0] + radius * np.cos(theta)
            y_bottom = bottom_center[1] + radius * np.sin(theta)
            z_bottom = np.full_like(theta, bottom_center[2])
            
            x_top = top_center[0] + radius * np.cos(theta)
            y_top = top_center[1] + radius * np.sin(theta)
            z_top = np.full_like(theta, top_center[2])
            
            # 绘制侧面
            for i in range(len(theta)-1):
                x = [x_bottom[i], x_top[i], x_top[i+1], x_bottom[i+1]]
                y = [y_bottom[i], y_top[i], y_top[i+1], y_bottom[i+1]]
                z = [z_bottom[i], z_top[i], z_top[i+1], z_bottom[i+1]]
                verts = [list(zip(x, y, z))]
                poly = Poly3DCollection(verts, alpha=alpha, color='gray')
                ax.add_collection3d(poly)
            
            # 绘制顶部和底部
            ax.plot_surface(x_top.reshape(1, -1), y_top.reshape(1, -1), z_top.reshape(1, -1), 
                           color='gray', alpha=alpha)
            ax.plot_surface(x_bottom.reshape(1, -1), y_bottom.reshape(1, -1), z_bottom.reshape(1, -1), 
                           color='gray', alpha=alpha)


def save_plot_3d_path(points, bounds, obstacles, save_path, info=None):
    """
    保存三维路径图到指定路径（包含起点、终点和中间点）并设置边界
    
    参数:
    points -- 路径点列表，格式为 [(x1, y1, z1), (x2, y2, z2), ...]
            第一个点为起点，最后一个点为终点，中间为路径点
    bounds -- 三维边界 [xmin, xmax, ymin, ymax, zmin, zmax]
    obstacles -- 障碍物列表
    save_path -- 图片保存路径（含文件名及扩展名，如 'path.png'）
    info -- 额外信息字典，用于显示规划结果
    """
    if len(points) < 2:
        print("起点处存在障碍物，不符合初始条件，无法绘制路径")
        return
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取坐标（支持带姿态的点）
    x = [p[0] if len(p) > 3 else p[0] for p in points]
    y = [p[1] if len(p) > 3 else p[1] for p in points]
    z = [p[2] if len(p) > 3 else p[2] for p in points]
    
    # 绘制路径
    ax.plot(x[1:-1], y[1:-1], z[1:-1], 
            color='b', linewidth=2, 
            label=f'Path with {len(points)} points')
    
    # 绘制起点和终点
    ax.plot([x[0]], [y[0]], [z[0]], 'g*', markersize=15, label='Start')
    ax.plot([x[-1]], [y[-1]], [z[-1]], 'r*', markersize=15, label='End')
    
    # 添加文本标注
    ax.text(x[0], y[0], z[0], 'Start', color='g', fontsize=12, weight='bold')
    
    if info and "terminal" in info:
        terminal = info["terminal"]
        if terminal == "reached_target":
            ax.text(x[-1], y[-1], z[-1], 'End', color='r', fontsize=12, weight='bold')
        elif terminal == "collision":
            ax.text(x[-1], y[-1], z[-1], 'Collision', color='r', fontsize=12, weight='bold')
        elif terminal == "timeout":
            ax.text(x[-1], y[-1], z[-1], 'Timeout', color='r', fontsize=12, weight='bold')
    
    # 添加算法信息
    if info and "algorithm" in info:
        algorithm = info["algorithm"]
        stats_text = f"Algorithm: {algorithm}"
        
        if "path_length" in info:
            stats_text += f"\nPath Length: {info['path_length']:.2f}"
        if "planning_time" in info:
            stats_text += f"\nTime: {info['planning_time']:.2f}s"
        if "iterations" in info:
            stats_text += f"\nIterations: {info['iterations']}"
        
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 设置边界
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])
    
    # 设置标签和标题
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    
    title = "3D Path Planning Result"
    if info and "algorithm" in info:
        title = f"3D Path Planning - {info['algorithm']}"
    ax.set_title(title, fontsize=14)
    
    # 添加图例和网格
    ax.legend(loc='best')
    ax.grid(True)
    
    # 画出障碍物
    for obstacle in obstacles:
        if hasattr(obstacle, 'draw'):
            obstacle.draw(ax)
        else:
            draw_obstacle_3d(ax, obstacle)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_path_comparison(paths, labels, bounds, obstacles, title="Path Comparison", save_path=None):
    """
    绘制多个路径的比较图
    
    参数:
    paths -- 路径列表，每个路径是点的列表
    labels -- 每个路径的标签
    bounds -- 三维边界
    obstacles -- 障碍物列表
    title -- 图表标题
    save_path -- 保存路径（可选）
    """
    if len(paths) != len(labels):
        print("路径数量与标签数量不匹配")
        return
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 颜色列表
    colors = ['b', 'r', 'g', 'm', 'c', 'y']
    
    # 绘制每个路径
    for i, (path, label) in enumerate(zip(paths, labels)):
        if len(path) < 2:
            continue
        
        # 提取坐标
        x = [p[0] if len(p) > 3 else p[0] for p in path]
        y = [p[1] if len(p) > 3 else p[1] for p in path]
        z = [p[2] if len(p) > 3 else p[2] for p in path]
        
        # 计算路径长度
        path_length = calculate_path_length(path)
        
        # 绘制路径
        ax.plot(x, y, z, 
                color=colors[i % len(colors)], 
                linewidth=2, 
                label=f'{label} (Length: {path_length:.2f})')
    
    # 设置边界
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])
    
    # 设置标签和标题
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # 添加图例和网格
    ax.legend(loc='best')
    ax.grid(True)
    
    # 画出障碍物
    for obstacle in obstacles:
        if hasattr(obstacle, 'draw'):
            obstacle.draw(ax)
        else:
            draw_obstacle_3d(ax, obstacle, alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ==================== 文件操作 ====================

def write_str_to_file(file_path: str, content: str):
    """将字符串写入文件（追加模式）"""
    with open(file_path, 'a') as f:
        f.write(content)


def save_path_to_file(path: List[np.ndarray], file_path: str):
    """将路径保存到文件"""
    with open(file_path, 'w') as f:
        f.write("# Path points (x, y, z, roll, pitch, yaw)\n")
        for point in path:
            if len(point) == 3:
                f.write(f"{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}\n")
            else:
                f.write(f"{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}, "
                       f"{point[3]:.6f}, {point[4]:.6f}, {point[5]:.6f}\n")


def load_path_from_file(file_path: str) -> List[np.ndarray]:
    """从文件加载路径"""
    path = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            values = [float(v) for v in line.split(',')]
            if len(values) == 3:
                path.append(np.array(values))
            elif len(values) == 6:
                path.append(np.array(values))
            else:
                print(f"Warning: Ignoring line with {len(values)} values: {line}")
    return path


# ==================== 目录管理 ====================

def create_record_dir(base_path):
    """
    在指定路径下创建新的记录文件夹，文件夹格式为"logs数字"
    Args:
        base_path: 基础路径
    Returns:
        str: 新创建的文件夹名称
    """
    # 确保基础路径存在
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # 获取所有符合"logs数字"格式的文件夹
    pattern = re.compile(r"logs(\d+)")
    existing_dirs = []
    for d in os.listdir(base_path):
        match = pattern.match(d)
        if match:
            existing_dirs.append(int(match.group(1)))
    
    # 如果没有已存在的文件夹，从1开始
    if not existing_dirs:
        new_dir_num = 1
    else:
        # 获取最大编号并加1
        new_dir_num = max(existing_dirs) + 1
    
    # 创建新文件夹，格式为"logs数字"
    new_dir_name = f"logs{new_dir_num}"
    new_dir_path = os.path.join(base_path, new_dir_name)
    os.makedirs(new_dir_path)
    
    return new_dir_name


def create_experiment_dir(base_path, experiment_name):
    """
    创建实验目录
    
    Args:
        base_path: 基础路径
        experiment_name: 实验名称
        
    Returns:
        str: 实验目录路径
    """
    # 创建时间戳
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建目录
    exp_dir = os.path.join(base_path, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # 创建子目录
    os.makedirs(os.path.join(exp_dir, "paths"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "configs"), exist_ok=True)
    
    return exp_dir


# ==================== 碰撞检测辅助函数 ====================

def is_point_safe(position, obstacles, safe_distance, tool_radius=0):
    """
    检查点是否安全（不与障碍物碰撞且保持安全距离）
    
    Args:
        position: 位置 [x, y, z]
        obstacles: 障碍物列表
        safe_distance: 安全距离
        tool_radius: 工具半径
        
    Returns:
        bool: 点是否安全
    """
    for obstacle in obstacles:
        # 计算到障碍物的距离
        if hasattr(obstacle, 'centerPoint'):
            distance = calculate_distance(position, obstacle.centerPoint)
            
            # 考虑障碍物半径
            if hasattr(obstacle, 'radius'):
                distance -= obstacle.radius
            
            # 考虑工具半径
            distance -= tool_radius
            
            # 检查是否安全
            if distance < safe_distance:
                return False
    
    return True


def is_path_safe(path, obstacles, safe_distance, tool_radius=0):
    """
    检查路径是否安全
    
    Args:
        path: 路径点列表
        obstacles: 障碍物列表
        safe_distance: 安全距离
        tool_radius: 工具半径
        
    Returns:
        (bool, float): (是否安全, 最小安全距离)
    """
    if len(path) == 0:
        return True, float('inf')
    
    min_safe_distance = float('inf')
    
    for point in path:
        # 提取位置
        if len(point) > 3:
            pos = point[:3]
        else:
            pos = point
        
        for obstacle in obstacles:
            if hasattr(obstacle, 'centerPoint'):
                distance = calculate_distance(pos, obstacle.centerPoint)
                
                if hasattr(obstacle, 'radius'):
                    distance -= obstacle.radius
                
                distance -= tool_radius
                
                if distance < safe_distance:
                    return False, distance
                
                min_safe_distance = min(min_safe_distance, distance)
    
    return True, min_safe_distance


# ==================== 路径评估 ====================

def evaluate_path(path, obstacles, start_pos, target_pos):
    """
    评估路径质量
    
    Args:
        path: 路径点列表
        obstacles: 障碍物列表
        start_pos: 起点位置
        target_pos: 目标位置
        
    Returns:
        dict: 包含各项评估指标
    """
    if len(path) == 0:
        return {
            'valid': False,
            'length': 0,
            'smoothness': 0,
            'safety': 0,
            'completion': 0
        }
    
    # 提取位置
    positions = []
    for point in path:
        if len(point) > 3:
            positions.append(point[:3])
        else:
            positions.append(point)
    
    # 路径长度
    length = calculate_path_length(path)
    
    # 路径平滑度
    smoothness = calculate_path_smoothness(path)
    
    # 路径安全性
    safety = calculate_path_safety(path, obstacles, 1.0)  # 使用1.0作为参考安全距离
    
    # 路径完成度（距离目标有多近）
    if len(positions) > 0:
        final_pos = positions[-1]
        completion = 1.0 / (1.0 + calculate_distance(final_pos, target_pos))
    else:
        completion = 0
    
    return {
        'valid': len(path) > 0,
        'length': length,
        'smoothness': smoothness,
        'safety': safety,
        'completion': completion,
        'num_points': len(path)
    }


def print_path_evaluation(evaluation):
    """打印路径评估结果"""
    print("\n" + "="*60)
    print("路径评估结果")
    print("="*60)
    print(f"有效性: {'有效' if evaluation['valid'] else '无效'}")
    print(f"路径长度: {evaluation['length']:.2f}")
    print(f"平滑度: {evaluation['smoothness']:.2f} rad")
    print(f"安全性: {evaluation['safety']:.2%}")
    print(f"完成度: {evaluation['completion']:.2%}")
    print(f"路径点数: {evaluation['num_points']}")
    print("="*60)