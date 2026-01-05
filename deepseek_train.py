import os
import yaml
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from AxisPathPlanEnv.MapEnv import MapEnv
from AxisPathPlanEnv.util import *
import keyboard
from AxisPathPlanEnv.Prime import *

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def train():
    # 加载配置
    config = load_config("AxisPathPlanEnv/env_config.yaml")
    
    # 创建环境
    env = MapEnv(config)
    
    # 初始化SAC模型
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./sac_tensorboard/"
    )
    
    # 回调函数（保存模型）
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="./models/",
        name_prefix="sac_model"
    )
    
    # 开始训练
    model.learn(
        total_timesteps=500000,
        callback=checkpoint_cb,
        log_interval=4,
        tb_log_name="sac_run"
    )
    
    # 保存最终模型
    model.save("./models/sac_final")

def key_listener():
    """键盘监听线程"""
    keyboard.wait('crtl+q')  # 阻塞直到按下ESC
    stop_requested = True
    print("\n停止指令已接收")
    return stop_requested

# 测试模型函数
def test_model(model_path,times=10):
    # 加载配置
    config = load_config("AxisPathPlanEnv/env_config.yaml")

    # 创建环境
    env = MapEnv(config, render_mode="human")

    obstacle=Cylinder(height=2,radius=1,centerPoint=np.array([0,0,0]))

    # 加载模型
    model = SAC.load(model_path)

    train_result_path="logs"
    current_path=create_record_dir(train_result_path)
    filepath="log.txt"
    filepath =train_result_path+"/"+ current_path+"/"+filepath
    success_count = 0
    collision_count=0
    timeout_count=0
    wrong_start_count=0
    stop_requested = False
    # 测试模型
    for time in range(times):
        pic_path = train_result_path+"/"+current_path+"/"+"pic_"+str(time)+".png"
        write_str_to_file(filepath,"the "+str(time)+" time test\n")
        #随机生成终点
        targetpos=np.random.uniform(low=-10, high=10, size=(3,))
        targetges=np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
        target=np.concatenate((targetpos,targetges))
        obs = env.reset(True) 
        #obs = env.reset() 
        #stop_requested = key_listener()
        if stop_requested:
            break

        for step in range(1000):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)     
            write_str_to_file(filepath, f"step:{step}\t reward: {reward}\tinfo: {info}\tdone: {done}\t target_relative:{obs[0:6]}\t distance_to_targetpos:{obs[12]}\t distance_to_targetges:{obs[13]}\n")
            if done :
                env.render(pic_path,info)
                if step==0:
                    wrong_start_count+=1
                    break
                if info["terminal"]=="reached_target":
                    success_count+=1
                if info["terminal"]=="collision":
                    collision_count+=1
                if info["terminal"]=="timeout":
                    timeout_count+=1
                break


    write_str_to_file(filepath,"success rate:"+str(success_count/(times-wrong_start_count))+"\n")
    write_str_to_file(filepath,"collision rate:"+str(collision_count/(times-wrong_start_count))+"\n")
    write_str_to_file(filepath,"timeout rate:"+str(timeout_count/(times-wrong_start_count))+"\n")
    print("success_count:",success_count)
    print("collision_count:",collision_count)
    print("wrong_start_count:",wrong_start_count)
       

if __name__ == "__main__":
    #train()
    
    # 注册热键（非阻塞方式）
    #keyboard.add_hotkey('esc', lambda: print("\n停止指令已接收") or exit(0))
    test_model("./models/sac_final",100)
