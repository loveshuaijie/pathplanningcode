import numpy as np
import yaml
from typing import *
import matplotlib.pyplot as plt

def loadYaml(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def calculate_angle(vec1, vec2):
    return np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def write_str_to_file(file_path: str, content: str):
    with open(file_path, 'a') as f:
        f.write(content)
    return

def plot_3d_path(points, bounds,obstacles):
    """
    绘制三维路径（包含起点、终点和中间点）并设置边界
    
    参数:
    points -- 路径点列表，格式为 [(x1, y1, z1), (x2, y2, z2), ...]
            第一个点为起点，最后一个点为终点，中间为路径点
    bounds -- 三维边界 [xmin, xmax, ymin, ymax, zmin, zmax]
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
    ax.set_title('3D Path with Intermediate Points', fontsize=14)
    
    # 添加图例和网格
    ax.legend(loc='best')
    ax.grid(True)
    
    # 添加坐标平面投影
    # X-Y平面投影
    # ax.plot(x, y, zmin*np.ones_like(z), 'k--', alpha=0.3)
    # # X-Z平面投影
    # ax.plot(x, ymin*np.ones_like(y), z, 'k--', alpha=0.3)
    # # Y-Z平面投影
    # ax.plot(xmin*np.ones_like(x), y, z, 'k--', alpha=0.3)
    #画出障碍物

    for obstacle in obstacles:
        obstacle.draw(ax)

    # 显示图形
    plt.tight_layout()
    plt.show()

def save_plot_3d_path(points, bounds, obstacles, save_path,info):
    """
    保存三维路径图到指定路径（包含起点、终点和中间点）并设置边界
    
    参数:
    points -- 路径点列表，格式为 [(x1, y1, z1), (x2, y2, z2), ...]
            第一个点为起点，最后一个点为终点，中间为路径点
    bounds -- 三维边界 [xmin, xmax, ymin, ymax, zmin, zmax]
    save_path -- 图片保存路径（含文件名及扩展名，如 'path.png'）
    """

    if len(points) < 2:
        print("起点处存在障碍物，不符合初始条件，无法绘制路径")
        return
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    
    ax.plot(x[1:-1], y[1:-1], z[1:-1], 
            color='b',  linewidth=2, 
            label=f'Path with {len(points)} points')
    
    ax.text(x[0], y[0], z[0], 'Start', color='g', fontsize=12, weight='bold')
    if info["terminal"]=="reached_target":
        ax.text(x[-1], y[-1], z[-1], 'End', color='r', fontsize=12, weight='bold')                    
    if info["terminal"]=="collision":
        ax.text(x[-1], y[-1], z[-1], 'collision', color='r', fontsize=12, weight='bold')
    if info["terminal"]=="timeout":
        ax.text(x[-1], y[-1], z[-1], 'timeout', color='r', fontsize=12, weight='bold')
    ax.plot([x[0]], [y[0]], [z[0]], 'g*', markersize=12, label='Start')
    ax.plot([x[-1]], [y[-1]], [z[-1]], 'r*', markersize=12, label='End')

    # 设置边界
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])
    
    # 设置标签和标题
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    ax.set_title('3D Path with Intermediate Points', fontsize=14)
    
    ax.legend(loc='best')
    ax.grid(True)
    
    for obstacle in obstacles:
        obstacle.draw(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存图片
    plt.close(fig)  # 关闭图形对象释放内存

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

def create_record_dir(base_path):
    """
    在指定路径下创建新的记录文件夹，文件夹格式为"logs数字"
    Args:
        base_path: 基础路径
    Returns:
        str: 新创建的文件夹名称
    """
    import os
    import re
    
    # 确保基础路径存在
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # 获取所有符合"logs数字"格式的文件夹
    # 使用正则表达式匹配"logs"开头后跟数字的文件夹
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
