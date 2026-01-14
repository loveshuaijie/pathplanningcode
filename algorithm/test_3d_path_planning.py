
import sys

sys.path.append('E:\pathplanning\pathplanningcode')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from AxisPathPlanEnv.MapEnv import MapEnv
from APF_3D import APF3D
from RRT_3D import RRT3D
from AxisPathPlanEnv.Prime import Sphere, Cuboid, Cylinder


def create_3d_test_environment(complexity: str = 'medium'):
    """
    创建三维测试环境
    
    Args:
        complexity: 环境复杂度 ('simple', 'medium', 'complex')
    
    Returns:
        MapEnv环境实例
    """
    # 基础环境配置
    env_config = {
        "envxrange": [-15, 15],    # 环境在X轴的范围
        "envyrange": [-15, 15],    # 环境在Y轴的范围
        "envzrange": [0, 15],
        "obstacles_num": 0,  # 手动设置障碍物
        "start": [-10, -10, 2, 0, 0, 0],
        "target": [12, 12, 10, 0, 0, 0],
        "tool_size": [3.0, 0.5],
        "maxstep": 1000,
        "period": 0.1,
        "safe_distance": 1.0,
        "alpha_max": np.pi / 3,
        "reachpos_scale": 10.0,
        "reachges_scale": 10.0,
        "Vmax": 1.0
    }
    
    env = MapEnv(env_config)
    
    # 创建障碍物
    obstacles = []
    
    if complexity == 'simple':
        # 简单环境：少量障碍物
        obstacles = [
            Sphere(radius=2.0, centerPoint=np.array([0, 0, 5])),
            Sphere(radius=1.5, centerPoint=np.array([-5, -5, 3])),
            Sphere(radius=2.0, centerPoint=np.array([5, 5, 7])),
        ]
    
    elif complexity == 'medium':
        # 中等环境：混合障碍物
        obstacles = [
            # 中央大型障碍物
            Sphere(radius=3.0, centerPoint=np.array([0, 0, 5])),
            
            # 周围障碍物
            Cuboid(size=np.array([4, 2, 6]), centerPoint=np.array([-6, -4, 3])),
            Cuboid(size=np.array([2, 4, 5]), centerPoint=np.array([6, 4, 2.5])),
            
            # 圆柱体障碍物
            Cylinder(height=8, radius=1.5, centerPoint=np.array([-8, 8, 4])),
            Cylinder(height=6, radius=2.0, centerPoint=np.array([8, -8, 3])),
            
            # 浮动障碍物
            Sphere(radius=1.5, centerPoint=np.array([-3, 6, 9])),
            Sphere(radius=1.0, centerPoint=np.array([6, -3, 11])),
        ]
    
    elif complexity == 'complex':
        # 复杂环境：密集障碍物
        obstacles = [
            # 中央障碍物群
            Sphere(radius=2.5, centerPoint=np.array([0, 0, 5])),
            Cuboid(size=np.array([3, 3, 8]), centerPoint=np.array([-2, -2, 4])),
            Cylinder(height=7, radius=1.8, centerPoint=np.array([2, 2, 3.5])),
            
            # 高空障碍物
            Sphere(radius=1.5, centerPoint=np.array([-8, 0, 9])),
            Sphere(radius=1.5, centerPoint=np.array([8, 0, 11])),
            Sphere(radius=1.5, centerPoint=np.array([0, -8, 8])),
            Sphere(radius=1.5, centerPoint=np.array([0, 8, 10])),
            
            # 狭窄通道
            Cuboid(size=np.array([2, 8, 5]), centerPoint=np.array([-4, 0, 2.5])),
            Cuboid(size=np.array([8, 2, 5]), centerPoint=np.array([0, 4, 2.5])),
        ]
    
        # 障碍物墙
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i != 0 or j != 0:
                    obstacles.append(
                        Sphere(radius=0.8, centerPoint=np.array([i*3, j*3, 2 + abs(i*j)*0.5]))
                    )
    # 设置障碍物
    for obstacle in obstacles:
        env.append_obstacle(obstacle)
        env.reset(generate_obstacles=False)
    
    return env, obstacles


def visualize_obstacle_environments():
    """可视化三种不同复杂度的障碍物环境"""
    fig = plt.figure(figsize=(18, 6))
    complexities = ['simple', 'medium', 'complex']
    titles = ['简单环境 (3个障碍物)', '中等环境 (7个障碍物)', '复杂环境 (33个障碍物)']
    
    for idx, complexity in enumerate(complexities):
        # 创建环境获取障碍物列表
        _, obstacles = create_3d_test_environment(complexity)
        
        # 创建3D子图
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        
        # 绘制障碍物
        for obstacle in obstacles:
            if hasattr(obstacle, 'centerPoint'):
                center = obstacle.centerPoint
                
                # 绘制球体障碍物
                if hasattr(obstacle, 'radius'):
                    u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:15j]
                    x = center[0] + obstacle.radius * np.cos(u) * np.sin(v)
                    y = center[1] + obstacle.radius * np.sin(u) * np.sin(v)
                    z = center[2] + obstacle.radius * np.cos(v)
                    ax.plot_surface(x, y, z, color='red', alpha=0.6, edgecolor='darkred')
                
                # 绘制长方体障碍物
                elif hasattr(obstacle, 'size'):
                    size = obstacle.size
                    corners = np.array([
                        [center[0]-size[0]/2, center[1]-size[1]/2, center[2]-size[2]/2],
                        [center[0]+size[0]/2, center[1]-size[1]/2, center[2]-size[2]/2],
                        [center[0]+size[0]/2, center[1]+size[1]/2, center[2]-size[2]/2],
                        [center[0]-size[0]/2, center[1]+size[1]/2, center[2]-size[2]/2],
                        [center[0]-size[0]/2, center[1]-size[1]/2, center[2]+size[2]/2],
                        [center[0]+size[0]/2, center[1]-size[1]/2, center[2]+size[2]/2],
                        [center[0]+size[0]/2, center[1]+size[1]/2, center[2]+size[2]/2],
                        [center[0]-size[0]/2, center[1]+size[1]/2, center[2]+size[2]/2],
                    ])
                    
                    # 绘制长方体的边
                    edges = [
                        [0,1], [1,2], [2,3], [3,0],
                        [4,5], [5,6], [6,7], [7,4],
                        [0,4], [1,5], [2,6], [3,7]
                    ]
                    
                    for edge in edges:
                        ax.plot(
                            [corners[edge[0],0], corners[edge[1],0]],
                            [corners[edge[0],1], corners[edge[1],1]],
                            [corners[edge[0],2], corners[edge[1],2]],
                            'blue', alpha=0.7, linewidth=2
                        )
                
                # 绘制圆柱体障碍物
                elif hasattr(obstacle, 'height') and hasattr(obstacle, 'radius'):
                    height = obstacle.height
                    radius = obstacle.radius
                    
                    # 绘制侧面
                    theta = np.linspace(0, 2*np.pi, 20)
                    z_side = np.linspace(center[2]-height/2, center[2]+height/2, 5)
                    theta_grid, z_grid = np.meshgrid(theta, z_side)
                    x_side = center[0] + radius * np.cos(theta_grid)
                    y_side = center[1] + radius * np.sin(theta_grid)
                    ax.plot_surface(x_side, y_side, z_grid, color='green', alpha=0.6, edgecolor='darkgreen')
                    
                    # 绘制顶部和底部圆盘
                    for z_offset in [-height/2, height/2]:
                        theta = np.linspace(0, 2*np.pi, 30)
                        r = np.linspace(0, radius, 10)
                        theta_grid, r_grid = np.meshgrid(theta, r)
                        x_disk = center[0] + r_grid * np.cos(theta_grid)
                        y_disk = center[1] + r_grid * np.sin(theta_grid)
                        z_disk = np.full_like(x_disk, center[2] + z_offset)
                        ax.plot_surface(x_disk, y_disk, z_disk, color='green', alpha=0.6, edgecolor='darkgreen')
        
        # 绘制起点和终点
        start_pos = np.array([-10, -10, 2])
        target_pos = np.array([12, 12, 10])
        
        ax.scatter(start_pos[0], start_pos[1], start_pos[2], 
                  c='green', s=150, marker='o', label='起点', edgecolors='black', linewidth=2)
        ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                  c='yellow', s=200, marker='*', label='终点', edgecolors='black', linewidth=2)
        
        # 设置坐标轴
        ax.set_xlabel('X轴', fontsize=10)
        ax.set_ylabel('Y轴', fontsize=10)
        ax.set_zlabel('Z轴', fontsize=10)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_zlim(0, 15)
        ax.set_title(titles[idx], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=25, azim=45)
        
        # 添加图例
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9)
    
    plt.suptitle('三维路径规划测试环境 - 障碍物可视化', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('obstacle_environments_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印障碍物统计信息
    print("\n" + "="*60)
    print("障碍物环境统计信息")
    print("="*60)
    
    for complexity in complexities:
        _, obstacles = create_3d_test_environment(complexity)
        
        # 统计不同类型障碍物的数量
        sphere_count = 0
        cuboid_count = 0
        cylinder_count = 0
        
        for obstacle in obstacles:
            if hasattr(obstacle, 'radius') and not hasattr(obstacle, 'height'):
                sphere_count += 1
            elif hasattr(obstacle, 'size'):
                cuboid_count += 1
            elif hasattr(obstacle, 'height') and hasattr(obstacle, 'radius'):
                cylinder_count += 1
        
        print(f"\n{complexity.capitalize()}环境:")
        print(f"  总障碍物数量: {len(obstacles)}")
        print(f"  球体障碍物: {sphere_count}")
        print(f"  长方体障碍物: {cuboid_count}")
        print(f"  圆柱体障碍物: {cylinder_count}")


def test_3d_apf():
    """测试三维APF算法"""
    print("\n" + "="*60)
    print("测试三维APF算法")
    print("="*60)
    
    # 创建环境
    env, obstacles = create_3d_test_environment('medium')
    
    # 创建APF规划器
    apf = APF3D(env)
    
    # 配置APF参数
    apf.config.update({
        'max_iterations': 1500,
        'step_size': 0.8,
        'attractive_gain': 2.5,
        'repulsive_gain': 6.0,
        'influence_radius': 6.0,
        'adaptive_step': True,
        'random_escape': True,
    })
    
    # 执行规划
    print("开始APF路径规划...")
    result = apf.plan()
    
    # 显示结果
    print(f"\nAPF规划结果:")
    print(f"  成功: {result['success']}")
    print(f"  迭代次数: {result['stats']['iterations']}")
    print(f"  路径长度: {result['stats']['path_length']:.2f}")
    print(f"  规划时间: {result['stats']['planning_time']:.2f}秒")
    print(f"  路径点数: {result['stats']['path_points']}")
    
    # 可视化
    apf.visualize("apf_3d_result.png")
    
    # 力场可视化（可选）
    if result['success']:
        force_field = apf.get_force_field(resolution=2.0)
        print(f"  力场分辨率: {force_field['resolution']}")
        print(f"  力场点数: {len(force_field['positions'])}")
    
    return result


def test_3d_rrt():
    """测试三维RRT算法"""
    print("\n" + "="*60)
    print("测试三维RRT算法")
    print("="*60)
    
    # 创建环境
    env, obstacles = create_3d_test_environment('medium')
    
    # 创建RRT规划器
    rrt = RRT3D(env)
    
    # 配置RRT参数
    rrt.config.update({
        'max_iterations': 3000,
        'step_size': 2.5,
        'goal_bias': 0.15,
        'use_rrt_star': True,
        'use_bidirectional': True,
        'rewire_radius': 5.0,
        'adaptive_step': True,
    })
    
    # 执行规划
    print("开始RRT路径规划...")
    result = rrt.plan()
    
    # 显示结果
    print(f"\nRRT规划结果:")
    print(f"  成功: {result['success']}")
    print(f"  树节点数: {result['stats']['tree_nodes']}")
    print(f"  路径长度: {result['stats']['path_length']:.2f}")
    print(f"  规划时间: {result['stats']['planning_time']:.2f}秒")
    print(f"  路径点数: {result['stats']['path_points']}")
    print(f"  碰撞检查次数: {result['stats']['collision_checks']}")
    
    # 可视化
    rrt.visualize("rrt_3d_result.png", show_tree=True)
    
    # 树结构可视化（可选）
    if result['success']:
        tree_data = rrt.get_tree_data()
        print(f"  树节点数: {len(tree_data['nodes'])}")
        print(f"  树边数: {len(tree_data['edges'])}")
    
    return result


def compare_3d_algorithms():
    """比较三维路径规划算法"""
    print("\n" + "="*60)
    print("比较三维路径规划算法")
    print("="*60)
    
    complexities = ['simple', 'medium', 'complex']
    results = {}
    
    for complexity in complexities:
        print(f"\n测试环境复杂度: {complexity}")
        print("-" * 40)
        
        # 创建相同环境
        env1, obstacles = create_3d_test_environment(complexity)
        env2, _ = create_3d_test_environment(complexity)
        
        # 测试APF
        print("执行APF规划...")
        apf = APF3D(env1)
        apf.config['max_iterations'] = 2000
        apf_result = apf.plan()
        
        # 测试RRT
        print("执行RRT规划...")
        rrt = RRT3D(env2)
        rrt.config['max_iterations'] = 4000
        rrt_result = rrt.plan()
        
        # 保存结果
        results[complexity] = {
            'APF': apf_result,
            'RRT': rrt_result
        }
        
        # 显示比较结果
        print(f"\n{complexity}环境比较结果:")
        print(f"{'指标':<20} {'APF':<15} {'RRT':<15}")
        print("-" * 50)
        print(f"{'成功':<20} {str(apf_result['success']):<15} {str(rrt_result['success']):<15}")
        print(f"{'路径长度':<20} {apf_result['stats']['path_length']:<15.2f} {rrt_result['stats']['path_length']:<15.2f}")
        print(f"{'规划时间':<20} {apf_result['stats']['planning_time']:<15.2f} {rrt_result['stats']['planning_time']:<15.2f}")
        print(f"{'迭代/节点':<20} {apf_result['stats']['iterations']:<15} {rrt_result['stats']['tree_nodes']:<15}")
    
    return results


def visualize_3d_comparison(apf_path, rrt_path, obstacles, bounds):
    """
    可视化APF和RRT路径比较
    
    Args:
        apf_path: APF路径
        rrt_path: RRT路径
        obstacles: 障碍物列表
        bounds: 环境边界
    """
    fig = plt.figure(figsize=(20, 10))
    
    # 1. 3D路径比较图
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 绘制障碍物
    for obstacle in obstacles:
        if hasattr(obstacle, 'centerPoint'):
            center = obstacle.centerPoint
            if hasattr(obstacle, 'radius'):
                # 绘制球体
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = center[0] + obstacle.radius * np.cos(u) * np.sin(v)
                y = center[1] + obstacle.radius * np.sin(u) * np.sin(v)
                z = center[2] + obstacle.radius * np.cos(v)
                ax1.plot_surface(x, y, z, color='gray', alpha=0.3)
            elif hasattr(obstacle, 'size'):
                # 绘制长方体
                size = obstacle.size
                corners = np.array([
                    [center[0]-size[0]/2, center[1]-size[1]/2, center[2]-size[2]/2],
                    [center[0]+size[0]/2, center[1]-size[1]/2, center[2]-size[2]/2],
                    [center[0]+size[0]/2, center[1]+size[1]/2, center[2]-size[2]/2],
                    [center[0]-size[0]/2, center[1]+size[1]/2, center[2]-size[2]/2],
                    [center[0]-size[0]/2, center[1]-size[1]/2, center[2]+size[2]/2],
                    [center[0]+size[0]/2, center[1]-size[1]/2, center[2]+size[2]/2],
                    [center[0]+size[0]/2, center[1]+size[1]/2, center[2]+size[2]/2],
                    [center[0]-size[0]/2, center[1]+size[1]/2, center[2]+size[2]/2],
                ])
                
                # 绘制长方体的边
                edges = [
                    [0,1], [1,2], [2,3], [3,0],
                    [4,5], [5,6], [6,7], [7,4],
                    [0,4], [1,5], [2,6], [3,7]
                ]
                
                for edge in edges:
                    ax1.plot(
                        [corners[edge[0],0], corners[edge[1],0]],
                        [corners[edge[0],1], corners[edge[1],1]],
                        [corners[edge[0],2], corners[edge[1],2]],
                        'gray', alpha=0.5
                    )
    
    # 绘制APF路径
    if apf_path and len(apf_path) > 0:
        apf_points = np.array(apf_path)
        ax1.plot(apf_points[:,0], apf_points[:,1], apf_points[:,2], 
                'b-', linewidth=3, label='APF路径', alpha=0.8)
        ax1.scatter(apf_points[0,0], apf_points[0,1], apf_points[0,2], 
                   c='green', s=200, marker='o', label='起点')
        ax1.scatter(apf_points[-1,0], apf_points[-1,1], apf_points[-1,2], 
                   c='red', s=200, marker='*', label='终点')
    
    # 绘制RRT路径
    if rrt_path and len(rrt_path) > 0:
        rrt_points = []
        for point in rrt_path:
            if len(point) > 3:
                rrt_points.append(point[:3])
            else:
                rrt_points.append(point)
        rrt_points = np.array(rrt_points)
        ax1.plot(rrt_points[:,0], rrt_points[:,1], rrt_points[:,2], 
                'r-', linewidth=3, label='RRT路径', alpha=0.8)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(bounds[0], bounds[1])
    ax1.set_ylim(bounds[2], bounds[3])
    ax1.set_zlim(bounds[4], bounds[5])
    ax1.set_title('三维路径规划比较')
    ax1.legend()
    ax1.grid(True)
    ax1.view_init(elev=30, azim=45)
    
    # 2. 2D投影图
    ax2 = fig.add_subplot(122)
    
    # 绘制障碍物投影
    for obstacle in obstacles:
        center = obstacle.centerPoint
        if hasattr(obstacle, 'radius'):
            circle = plt.Circle((center[0], center[1]), obstacle.radius, 
                               color='gray', alpha=0.3)
            ax2.add_patch(circle)
    
    # 绘制路径投影
    if apf_path and len(apf_path) > 0:
        apf_points = np.array(apf_path)
        ax2.plot(apf_points[:,0], apf_points[:,1], 'b-', linewidth=2, label='APF路径')
    
    if rrt_path and len(rrt_path) > 0:
        rrt_points = []
        for point in rrt_path:
            if len(point) > 3:
                rrt_points.append(point[:3])
            else:
                rrt_points.append(point)
        rrt_points = np.array(rrt_points)
        ax2.plot(rrt_points[:,0], rrt_points[:,1], 'r-', linewidth=2, label='RRT路径')
    
    # 绘制起点和终点
    if apf_path and len(apf_path) > 0:
        ax2.scatter(apf_points[0,0], apf_points[0,1], c='green', s=100, marker='o', label='起点')
        ax2.scatter(apf_points[-1,0], apf_points[-1,1], c='red', s=100, marker='*', label='终点')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_xlim(bounds[0], bounds[1])
    ax2.set_ylim(bounds[2], bounds[3])
    ax2.set_title('XY平面投影')
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('3d_path_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_3d_path_planning():
    """三维路径规划演示"""
    print("\n" + "="*60)
    print("三维路径规划算法演示")
    print("="*60)
    
    # 创建测试环境
    env, obstacles = create_3d_test_environment('medium')
    bounds = [env.xrange[0], env.xrange[1], 
              env.yrange[0], env.yrange[1],
              env.zrange[0], env.zrange[1]]
    
    # 测试APF
    print("\n1. 测试APF算法...")
    apf = APF3D(env)
    apf_result = apf.plan()
    
    if apf_result['success']:
        print(f"APF规划成功!")
        print(f"  路径长度: {apf_result['stats']['path_length']:.2f}")
        print(f"  规划时间: {apf_result['stats']['planning_time']:.2f}秒")
    else:
        print("APF规划失败")
    
    # 创建新环境测试RRT
    env2, _ = create_3d_test_environment('medium')
    
    print("\n2. 测试RRT算法...")
    rrt = RRT3D(env2)
    rrt_result = rrt.plan()
    
    if rrt_result['success']:
        print(f"RRT规划成功!")
        print(f"  路径长度: {rrt_result['stats']['path_length']:.2f}")
        print(f"  规划时间: {rrt_result['stats']['planning_time']:.2f}秒")
        print(f"  树节点数: {rrt_result['stats']['tree_nodes']}")
    else:
        print("RRT规划失败")
    
    # 可视化比较
    print("\n3. 生成可视化结果...")
    visualize_3d_comparison(
        apf_result['path'], 
        rrt_result['path'], 
        obstacles, 
        bounds
    )
    
    # 算法比较
    print("\n4. 算法性能比较:")
    print(f"{'算法':<10} {'成功':<8} {'路径长度':<12} {'规划时间':<12} {'效率':<10}")
    print("-" * 55)
    
    apf_success = apf_result['success']
    rrt_success = rrt_result['success']
    
    if apf_success:
        apf_efficiency = apf_result['stats']['path_length'] / max(apf_result['stats']['planning_time'], 0.1)
        print(f"{'APF':<10} {str(apf_success):<8} {apf_result['stats']['path_length']:<12.2f} {apf_result['stats']['planning_time']:<12.2f} {apf_efficiency:<10.2f}")
    
    if rrt_success:
        rrt_efficiency = rrt_result['stats']['path_length'] / max(rrt_result['stats']['planning_time'], 0.1)
        print(f"{'RRT':<10} {str(rrt_success):<8} {rrt_result['stats']['path_length']:<12.2f} {rrt_result['stats']['planning_time']:<12.2f} {rrt_efficiency:<10.2f}")
    
    print("\n演示完成!")
    return apf_result, rrt_result


if __name__ == "__main__":
    print("开始三维路径规划测试...")
    
    # 首先可视化障碍物环境
    visualize_obstacle_environments()
    
    # 运行演示
    #demo_3d_path_planning()
    
    # 分别测试算法
    #apf_result = test_3d_apf()
    #rrt_result = test_3d_rrt()
    
    # 比较算法性能
    comparison_results = compare_3d_algorithms()
    
    print("\n测试完成!")
