import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 引用你的环境和工具
from AxisPathPlanEnv.MapEnv import MapEnv
from AxisPathPlanEnv.util import save_plot_3d_path

# ================= 配置参数 (需与训练时一致) =================
ENV_CONFIG = {
    "envxrange": [-10, 10],
    "envyrange": [-10, 10],
    "envzrange": [-10, 10],
    "obstacles_num": 5, 
    "safe_distance": 1.5,
    "tool_size": [2.0, 0.5],
    "Vmax": 2.0,
    "maxstep": 100, 
    "period": 0.1, 
    "alpha_max": np.pi/4,
    "goal_conditioned": True,
    "reward_type": "sparse",
    "start": [0, 0, 0, 0, 0, 0],
    "target": [5, 5, 5, 0, 0, 0],
    "reachpos_scale": 20.0,
    "reachges_scale": 10.0,
}

# 模型路径
MODEL_PATH = "./models/sac_her_axis/sac_axis_final"
# 如果你使用了 VecNormalize，还需要加载统计文件
NORM_PATH = "./models/sac_her_axis/vec_normalize.pkl"
# 结果保存路径
RESULT_DIR = "./test_results/"
os.makedirs(RESULT_DIR, exist_ok=True)

def run_test(num_episodes=10, use_norm=False):
    # 1. 创建环境
    # 测试时通常不需要 Vectorized 环境，但为了加载模型方便，我们还是包一层 DummyVecEnv
    # 如果训练时用了 VecNormalize，这里必须保持一致
    env = MapEnv(ENV_CONFIG)
    
    # 包装环境以适配 SB3 加载逻辑
    vec_env = DummyVecEnv([lambda: env])

    # 2. 处理归一化 (如果在训练中使用了 VecNormalize)
    if use_norm and os.path.exists(NORM_PATH):
        print(f"Loading normalization stats from {NORM_PATH}...")
        vec_env = VecNormalize.load(NORM_PATH, vec_env)
        vec_env.training = False  # 测试模式：不再更新均值和方差
        vec_env.norm_reward = False
    else:
        print("Running without VecNormalize.")

    # 3. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = SAC.load(MODEL_PATH, env=vec_env)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. 开始测试循环
    success_count = 0
    collision_count = 0

    for ep in range(num_episodes):
        obs = vec_env.reset()
        
        # 获取底层环境实例 (用于访问 obstacles 和 plotting)
        # vec_env -> envs[0] -> (如果有Monitor/TimeLimit) -> unwrapped
        raw_env = vec_env.envs[0].unwrapped
        
        done = False
        truncated = False
        trajectory = [] # 用于绘图
        
        print(f"\n--- Episode {ep+1} ---")
        print(f"Target: {raw_env.target_pos}")

        step_cnt = 0
        while not (done or truncated):
            # 预测动作 (deterministic=True 意味着不加噪声，纯贪婪策略)
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, done, info = vec_env.step(action)
            
            single_info = info[0]
            # 记录轨迹 (注意 vec_env 返回的是 list，我们取第0个)
            current_pos = single_info['pos']
            trajectory.append(current_pos)
            target_pos = single_info['target']
            
            # 提取 Info
            # vec_env 返回的 info 是一个列表
            
            #print(single_info)

            # 简单的 Debug 打印
            env_instance = vec_env.envs[0].unwrapped
            dist = np.linalg.norm(current_pos - target_pos)
            print(f"Step {step_cnt:02d} | Action: {action[0][:3]} | Pos: {current_pos} | Dist: {dist:.4f}")
            
            step_cnt += 1

        # 5. 结果统计与绘图
        is_success = single_info.get('is_success', False)
        is_collision = single_info.get('is_collision', False)
        
        if is_success:
            print(f"Result: SUCCESS ✅ (Steps: {step_cnt})")
            success_count += 1
        elif is_collision:
            print(f"Result: COLLISION 💥 (Steps: {step_cnt})")
            collision_count += 1
        else:
            print(f"Result: TIMEOUT ⏳ (Dist: {dist:.2f})")

        # 6. 调用 util.py 画图
        # 构造环境范围数组 [xmin, xmax, ymin, ymax, zmin, zmax]
        env_ranges = np.concatenate((raw_env.x_range, raw_env.y_range, raw_env.z_range))
        
        pic_path = os.path.join(RESULT_DIR, f"ep_{ep+1}_result.png")
        
        try:
            # 补充 info 信息用于标题显示
            plot_info = {
                'valid': is_success,
                'length': len(trajectory),
                'smoothness': 0, # 暂时填0
                'safety': 0 if is_collision else 1
            }
            
            print(f"Saving trajectory plot to {pic_path}...")
            save_plot_3d_path(
                trajectory,
                env_ranges,
                raw_env.obstacles,
                pic_path,
                plot_info
            )
        except Exception as e:
            print(f"Plotting failed: {e}")
            # 有可能是 trajectory 格式问题，确保它是 list of numpy arrays

    print("\n================ TEST REPORT ================")
    print(f"Total Episodes: {num_episodes}")
    print(f"Success Rate:   {success_count/num_episodes*100:.1f}%")
    print(f"Collision Rate: {collision_count/num_episodes*100:.1f}%")
    print("=============================================")

if __name__ == "__main__":
    # 如果你在训练代码中使用了 VecNormalize，请把 use_norm 设为 True
    run_test(num_episodes=10, use_norm=False)