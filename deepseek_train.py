
import os
import yaml
import numpy as np
import threading
import time
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from AxisPathPlanEnv.MapEnv import MapEnv
from AxisPathPlanEnv.util import *
from AxisPathPlanEnv.Prime import *
import keyboard


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class RandomEnvWrapper(gym.Env):
    """
    随机环境包装器，用于随机起点和终点的训练
    
    这个类继承了gym.Env，确保与stable-baselines3兼容
    """
    
    def __init__(self, base_env_config, random_start=True):
        super().__init__()
        self.base_env_config = base_env_config
        self.random_start = random_start
        
        # 创建基础环境
        self.env = MapEnv(base_env_config)
        
        # 复制观察空间和动作空间
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # 保存原始起点和目标
        self.original_start = self.env.start.copy()
        self.original_target = self.env.target.copy()
        
        # 环境边界
        self.xrange = base_env_config["envxrange"]
        self.yrange = base_env_config["envyrange"]
        self.zrange = base_env_config["envzrange"]
    
    def reset(self, **kwargs):
        """
        重置环境，如果启用随机起点和目标，则生成随机值
        """
        if self.random_start:
            # 生成随机起点
            start_pos = np.random.uniform(
                low=[self.xrange[0], self.yrange[0], self.zrange[0]],
                high=[self.xrange[1], self.yrange[1], self.zrange[1]],
                size=(3,)
            )
            start_ges = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
            start = np.concatenate((start_pos, start_ges))
            
            # 生成随机目标点，确保与起点有一定距离
            target_pos = np.random.uniform(
                low=[self.xrange[0], self.yrange[0], self.zrange[0]],
                high=[self.xrange[1], self.yrange[1], self.zrange[1]],
                size=(3,)
            )
            
            # 确保目标点和起点之间的距离至少为5
            while np.linalg.norm(target_pos - start_pos) < 5:
                target_pos = np.random.uniform(
                    low=[self.xrange[0], self.yrange[0], self.zrange[0]],
                    high=[self.xrange[1], self.yrange[1], self.zrange[1]],
                    size=(3,)
                )
            
            target_ges = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
            target = np.concatenate((target_pos, target_ges))
            
            # 使用随机起点和目标重置环境
            obs = self.env.reset(start=start, target=target, generate_obstacles=True)
        else:
            # 使用配置中的起点和目标
            obs = self.env.reset(generate_obstacles=True)
        
        return obs
    
    def step(self, action):
        """执行一步动作"""
        return self.env.step(action)
    
    def render(self, mode='human'):
        """渲染环境"""
        return self.env.render()
    
    def close(self):
        """关闭环境"""
        return self.env.close()
    
    def seed(self, seed=None):
        """设置随机种子"""
        return self.env.seed(seed)
    
    @property
    def unwrapped(self):
        """返回原始环境"""
        return self.env


def create_training_env(config):
    """
    创建训练环境
    
    Args:
        config: 环境配置
    
    Returns:
        创建好的环境
    """
    def _make_env():

            # 使用原始环境
        env = MapEnv(config, render_mode="human")
        
        # 添加Monitor包装器以记录训练信息
        #env = Monitor(env)
        return env
    
    # 创建向量化环境
    return DummyVecEnv([_make_env])


def train_sac(config_path="AxisPathPlanEnv/env_config.yaml", 
              total_timesteps=500000,
              random_start=True,
              tensorboard_log="./sac_tensorboard/",
              model_save_path="./models/"):
    """
    训练SAC模型
    
    Args:
        config_path: 配置文件路径
        total_timesteps: 总训练步数
        random_start: 是否使用随机起点和终点
        tensorboard_log: TensorBoard日志路径
        model_save_path: 模型保存路径
    """
    print("="*60)
    print("开始训练SAC模型")
    print("="*60)
    
    # 加载配置
    config = load_config(config_path)
    
    # 确保配置文件中有必需的参数
    required_params = ["reachpos_scale", "reachges_scale"]
    for param in required_params:
        if param not in config:
            print(f"警告: 配置文件中缺少参数 '{param}'，使用默认值10.0")
            config[param] = 10.0
    
    # 创建训练环境
    print("创建训练环境...")
    train_env = create_training_env(config)

    
    # 创建评估环境（不使用随机起点和终点以便公平比较）
    print("创建评估环境...")
    eval_env = create_training_env(config)
    
    # 创建模型保存目录
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(os.path.join(model_save_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(model_save_path, "best_model"), exist_ok=True)
    
    # 初始化SAC模型
    print("初始化SAC模型...")
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1,
        learning_starts=10000,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device='auto'  # 自动选择GPU或CPU
    )
    
    # 创建回调函数
    print("设置回调函数...")
    
    # 检查点回调
    checkpoint_cb = CheckpointCallback(
        save_freq=max(5000, total_timesteps // 100),
        save_path=os.path.join(model_save_path, "checkpoints"),
        name_prefix="sac_checkpoint",
        verbose=1
    )
    
    # 提前停止回调
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=10,
        verbose=1
    )
    
    # 评估回调
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_path, "best_model"),
        log_path=os.path.join(model_save_path, "logs"),
        eval_freq=max(5000, total_timesteps // 100),
        deterministic=True,
        render=False,
        verbose=1,
        callback_after_eval=stop_cb
    )
    
    # 开始训练
    print(f"\n开始训练，总步数: {total_timesteps}")
    print("-" * 40)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb],
            log_interval=10,
            tb_log_name="sac_training",
            reset_num_timesteps=True
        )
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存最终模型
    final_model_path = os.path.join(model_save_path, "sac_final")
    model.save(final_model_path)
    print(f"\n最终模型已保存到: {final_model_path}")
    
    # 关闭环境
    train_env.close()
    eval_env.close()
    
    return model, final_model_path


class KeyboardListener:
    """键盘监听器"""
    
    def __init__(self):
        self.stop_requested = False
        self.listener_thread = None
    
    def start(self):
        """开始监听"""
        def listen():
            print("键盘监听器已启动，按 'Ctrl+Q' 停止训练")
            keyboard.wait('ctrl+q')
            self.stop_requested = True
            print("\n收到停止信号，正在停止训练...")
        
        self.listener_thread = threading.Thread(target=listen)
        self.listener_thread.daemon = True
        self.listener_thread.start()
    
    def stop(self):
        """停止监听"""
        self.stop_requested = True
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1)
    
    def is_stop_requested(self):
        """检查是否请求停止"""
        return self.stop_requested


def evaluate_model(model_path, config_path="AxisPathPlanEnv/env_config.yaml", 
                   test_times=10, random_targets=True, render=False):
    """
    评估模型性能
    
    Args:
        model_path: 模型路径
        config_path: 配置文件路径
        test_times: 测试次数
        random_targets: 是否使用随机目标点
        render: 是否渲染环境
    
    Returns:
        评估结果字典
    """
    print("="*60)
    print("评估模型性能")
    print("="*60)
    
    # 加载配置
    config = load_config(config_path)
    
    # 确保配置文件中有必需的参数
    required_params = ["reachpos_scale", "reachges_scale"]
    for param in required_params:
        if param not in config:
            print(f"警告: 配置文件中缺少参数 '{param}'，使用默认值10.0")
            config[param] = 10.0
    
    # 创建环境
    env = MapEnv(config, render_mode="human" if render else None)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    try:
        model = SAC.load(model_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 创建记录目录
    train_result_path = "evaluation_logs"
    current_path = create_record_dir(train_result_path)
    log_file = os.path.join(train_result_path, current_path, "evaluation_log.txt")
    
    # 初始化统计变量
    success_count = 0
    collision_count = 0
    timeout_count = 0
    episode_rewards = []
    episode_lengths = []
    
    print(f"\n开始评估，测试次数: {test_times}")
    print("-" * 40)
    
    for episode in range(test_times):
        print(f"\n测试第 {episode+1}/{test_times} 次...")
        
        # 设置目标点
        if random_targets:
            # 随机生成目标点
            targetpos = np.random.uniform(low=-10, high=10, size=(3,))
            targetges = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
            target = np.concatenate((targetpos, targetges))
            
            obs = env.reset(target=target, generate_obstacles=True)
        else:
            # 使用配置中的固定目标点
            obs = env.reset(generate_obstacles=True)
        
        # 保存图片路径
        pic_path = os.path.join(train_result_path, current_path, f"episode_{episode+1}.png")
        
        # 记录日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"第 {episode+1} 次测试\n")
            f.write(f"{'='*50}\n")
            if random_targets:
                f.write(f"目标点: {target}\n")
        
        # 运行一个episode
        episode_reward = 0
        done = False
        steps = 0
        info = {}
        
        while not done and steps < 1000:
            # 预测动作
            action, _states = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            # 更新统计
            episode_reward += reward
            steps += 1
            
            # 记录日志
            if steps % 100 == 0 or done:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"步数: {steps}, 累计奖励: {episode_reward:.2f}, "
                           f"到目标位置距离: {obs[12]:.2f}, 到目标姿态距离: {obs[13]:.2f}\n")
        
        # 记录结果
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # 判断终止原因
        if done:
            terminal_reason = info.get("terminal", "unknown")
            if terminal_reason == "reached_target":
                success_count += 1
                print(f"  成功到达目标! 奖励: {episode_reward:.2f}, 步数: {steps}")
            elif terminal_reason == "collision":
                collision_count += 1
                print(f"  发生碰撞! 奖励: {episode_reward:.2f}, 步数: {steps}")
            elif terminal_reason == "timeout":
                timeout_count += 1
                print(f"  超时! 奖励: {episode_reward:.2f}, 步数: {steps}")
        else:
            timeout_count += 1
            print(f"  超时! 奖励: {episode_reward:.2f}, 步数: {steps}")
        
        # 渲染最后一帧
        if render:
            env.render(pic_path, info)
    
    # 计算统计数据
    valid_tests = success_count + collision_count + timeout_count
    success_rate = success_count / valid_tests if valid_tests > 0 else 0
    collision_rate = collision_count / valid_tests if valid_tests > 0 else 0
    timeout_rate = timeout_count / valid_tests if valid_tests > 0 else 0
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    
    # 保存评估结果
    results = {
        'success_count': success_count,
        'collision_count': collision_count,
        'timeout_count': timeout_count,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeout_rate,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'log_file': log_file
    }
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果汇总")
    print("="*60)
    print(f"成功次数: {success_count}/{test_times} ({success_rate*100:.1f}%)")
    print(f"碰撞次数: {collision_count}/{test_times} ({collision_rate*100:.1f}%)")
    print(f"超时次数: {timeout_count}/{test_times} ({timeout_rate*100:.1f}%)")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_length:.1f}")
    print(f"详细日志: {log_file}")
    
    # 写入总结日志
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write("评估结果汇总\n")
        f.write(f"{'='*50}\n")
        f.write(f"总测试次数: {test_times}\n")
        f.write(f"成功次数: {success_count} ({success_rate*100:.1f}%)\n")
        f.write(f"碰撞次数: {collision_count} ({collision_rate*100:.1f}%)\n")
        f.write(f"超时次数: {timeout_count} ({timeout_rate*100:.1f}%)\n")
        f.write(f"平均奖励: {avg_reward:.2f}\n")
        f.write(f"平均步数: {avg_length:.1f}\n")
    
    # 关闭环境
    env.close()
    
    return results


def visualize_training_progress(log_dir="./sac_tensorboard/", model_save_path="./models/"):
    """
    可视化训练进度
    
    Args:
        log_dir: TensorBoard日志目录
        model_save_path: 模型保存路径
    """
    try:
        # 尝试导入TensorBoard相关库
        from torch.utils.tensorboard import SummaryWriter
        import matplotlib.pyplot as plt
        
        print("\n" + "="*60)
        print("分析训练进度")
        print("="*60)
        
        # 这里可以添加训练进度分析代码
        # 例如：读取TensorBoard日志，绘制训练曲线等
        
        print("提示: 使用以下命令查看TensorBoard:")
        print(f"tensorboard --logdir={log_dir}")
        print("\n训练日志和模型保存在:")
        print(f"TensorBoard日志: {log_dir}")
        print(f"模型文件: {model_save_path}")
        
    except ImportError:
        print("无法导入TensorBoard，请确保已安装: pip install tensorboard torch")


def train_with_keyboard_control():
    """带键盘控制的训练"""
    print("="*60)
    print("带键盘控制的训练")
    print("="*60)
    
    # 创建键盘监听器
    keyboard_listener = KeyboardListener()
    keyboard_listener.start()
    
    # 训练配置
    config_path = "AxisPathPlanEnv/env_config.yaml"
    total_timesteps = 500000
    
    # 加载配置
    config = load_config(config_path)
    
    # 确保配置文件中有必需的参数
    required_params = ["reachpos_scale", "reachges_scale"]
    for param in required_params:
        if param not in config:
            print(f"警告: 配置文件中缺少参数 '{param}'，使用默认值10.0")
            config[param] = 10.0
    
    # 创建环境
    train_env = create_training_env(config, random_start=True)
    
    # 初始化模型
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./sac_tensorboard/"
    )
    
    # 训练循环（手动控制）
    print(f"\n开始训练，按 'Ctrl+Q' 停止")
    print("-" * 40)
    
    timesteps = 0
    checkpoint_interval = 10000
    
    while timesteps < total_timesteps and not keyboard_listener.is_stop_requested():
        # 训练一步
        model.learn(
            total_timesteps=checkpoint_interval,
            reset_num_timesteps=False,
            log_interval=10,
            tb_log_name="sac_manual"
        )
        
        timesteps += checkpoint_interval
        print(f"已训练步数: {timesteps}/{total_timesteps}")
        
        # 保存检查点
        if timesteps % 50000 == 0:
            checkpoint_path = f"./models/checkpoints/sac_manual_{timesteps}"
            model.save(checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")
    
    # 保存最终模型
    if timesteps > 0:
        final_path = "./models/sac_manual_final"
        model.save(final_path)
        print(f"\n最终模型已保存: {final_path}")
    
    # 停止监听器
    keyboard_listener.stop()
    
    # 关闭环境
    train_env.close()
    
    print("\n训练完成!")


if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='SAC路径规划训练和评估')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'train_manual', 'visualize'],
                       help='运行模式: train(训练), eval(评估), train_manual(手动训练), visualize(可视化)')
    parser.add_argument('--model_path', type=str, default='./models/sac_final',
                       help='模型路径 (用于评估模式)')
    parser.add_argument('--timesteps', type=int, default=2000000,
                       help='训练步数 (用于训练模式)')
    parser.add_argument('--test_times', type=int, default=30,
                       help='测试次数 (用于评估模式)')
    parser.add_argument('--random_targets', action='store_true', default=True,
                       help='是否使用随机目标点 (用于评估模式)')
    parser.add_argument('--render', action='store_true', default=False,
                       help='是否渲染环境 (用于评估模式)')
    parser.add_argument('--config', type=str, default='AxisPathPlanEnv/env_config.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 根据模式执行相应操作
    if args.mode == 'train':
        # 自动训练模式
        model, model_path = train_sac(
            config_path=args.config,
            total_timesteps=args.timesteps,
            random_start=False
        )
        
        # 评估训练好的模型
        if model_path:
            print("\n评估训练好的模型...")
            results = evaluate_model(
                model_path=model_path,
                config_path=args.config,
                test_times=10,
                random_targets=True,
                render=False
            )
        
    elif args.mode == 'eval':
        # 评估模式
        results = evaluate_model(
            model_path=args.model_path,
            config_path=args.config,
            test_times=args.test_times,
            random_targets=args.random_targets,
            render=args.render
        )
        
    elif args.mode == 'train_manual':
        # 手动训练模式（带键盘控制）
        train_with_keyboard_control()
        
    elif args.mode == 'visualize':
        # 可视化训练进度
        visualize_training_progress()
        
    else:
        print(f"未知模式: {args.mode}")
        parser.print_help()
