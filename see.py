import gym
import stable_baselines3
from AxisPathPlanEnv.util import *
import os
print(gym.__version__)
print(stable_baselines3.__version__)

import keyboard
import time

def main_task():
    while True:
        print("程序运行中... 按 ESC 键停止")
        time.sleep(1)  # 你的主要任务代码
        if(keyboard.is_modifier('esc')):
            print("停止指令已接收")
            break

def run_with_stop():
    print("程序启动，按 ESC 键可随时停止")
    
    # 注册热键（非阻塞方式）
    keyboard.add_hotkey('esc', lambda: print("\n停止指令已接收") or exit(0))
    
    # 运行主任务
    main_task()

if __name__ == "__main__":
    import numpy as np
    import os
    # run_with_stop()
    log_dir = "logs"
    run_num = 0
    str1=create_record_dir(log_dir)
    print(str1)
