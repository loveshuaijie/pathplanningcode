import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

# 引用你的环境
from AxisPathPlanEnv.MapEnv import MapEnv

# ================= 配置参数 =================
ENV_CONFIG = {
    # 空间范围
    "envxrange": [-10, 10],
    "envyrange": [-10, 10],
    "envzrange": [-10, 10],
    
    # 障碍物
    "obstacles_num": 10, 
    "safe_distance": 0.5, # 安全距离
    
    # 工具参数
    "tool_size": [1.0, 0.2], # [height, radius]
    "Vmax": 2.0,             # 最大速度
    
    # 训练逻辑参数
    "maxstep": 100,          # 单个 Episode 最大步数
    "period": 0.1,           # dt
    "alpha_max": np.pi/4,    # 角度限制(如需)
    
    # HER 核心配置
    "goal_conditioned": True,  # 【重要】开启 HER 模式
    "reward_type": "sparse",   # HER 配合稀疏奖励通常更稳定，也可以试 "dense"
    
    # 初始状态 (会被 reset 覆盖，但初始化需要)
    "start": [0, 0, 0, 0, 0, 0],
    "target": [5, 5, 5, 0, 0, 0],
    
    # 缩放因子
    "reachpos_scale": 20.0,
    "reachges_scale": 10.0,
}

LOG_DIR = "./logs/sac_her_axis/"
MODEL_DIR = "./models/sac_her_axis/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env():
    """创建并包装环境"""
    env = MapEnv(ENV_CONFIG, render_mode=None) # 训练时不渲染
    # 使用 Monitor 记录每一局的 Reward 和 Length
    env = Monitor(env, LOG_DIR) 
    return env

def main():
    # 1. 环境检查 (Sanity Check)
    # 确保之前重构的 MapEnv 符合 Gymnasium 标准
    # print("Checking environment compatibility...")
    # temp_env = MapEnv(ENV_CONFIG)
    # check_env(temp_env)
    # print("Environment check passed!")

    # 2. 创建向量化环境 (SB3 推荐)
    # 即使是单进程，DummyVecEnv 也能处理数据维度的对齐
    vec_env = DummyVecEnv([make_env])

    # 3. 定义 SAC 模型
    # 如果你想用你上传的 HerReplayBuffer.py，请确保它完整实现了 HER 采样逻辑
    # 这里我们使用 SB3 官方经过大量测试的 HerReplayBuffer
    model = SAC(
        "MultiInputPolicy",           # 支持 Dict Observation
        vec_env,
        replay_buffer_class=HerReplayBuffer, # 激活 HER
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,         # k=4: 每一个真实经历生成 4 个虚拟经历
            goal_selection_strategy="future", # 经典的 HER 策略
        ),
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=100000,           # 根据内存调整
        gamma=0.95,                   # 折扣因子
        tau=0.05,                     # 软更新系数
        tensorboard_log=LOG_DIR,
        device="auto"                 # 自动选择 CUDA 或 CPU
    )

    # 4. 设置回调函数
    # 每 5000 步保存一次模型
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path=MODEL_DIR, 
        name_prefix="sac_axis_model"
    )
    
    # (可选) 评估回调：定期在一个独立的测试环境中测试模型性能
    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR + "/best_model",
        log_path=LOG_DIR,
        eval_freq=2000,
        deterministic=True,
        render=False
    )

    # 5. 开始训练
    print("Start training...")
    try:
        model.learn(
            total_timesteps=1000000, # 训练总步数，根据任务难度增加
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True       # 显示进度条
        )
    except KeyboardInterrupt:
        print("Training interrupted manually.")

    # 6. 保存最终模型
    model.save(f"{MODEL_DIR}/sac_axis_final")
    print(f"Model saved to {MODEL_DIR}")

    # 7. 简单的测试与可视化
    print("Testing loaded model...")
    # 加载模型 (注意指定 custom_objects，或者重新传参)
    loaded_model = SAC.load(f"{MODEL_DIR}/sac_axis_final", env=vec_env)
    
    obs = vec_env.reset()
    for i in range(200):
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        
        # 打印部分信息
        if i % 10 == 0:
            print(f"Step {i}, Reward: {rewards[0]:.4f}")
            
        if dones[0]:
            print("Episode finished.")
            obs = vec_env.reset()
            break

if __name__ == "__main__":
    main()