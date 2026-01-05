# import gym
# import numpy as np
# from gym import spaces
# from stable_baselines3 import SAC

# # 改进版环境类
# class DynamicTargetContinuousEnv(gym.Env):
#     metadata = {'render.modes': ['human']}
    
#     def __init__(self, train_mode=True):
#         super(DynamicTargetContinuousEnv, self).__init__()
        
#         # 连续动作空间 [-1, 1]
#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
#         # 状态空间包含当前位置和目标位置
#         self.observation_space = spaces.Box(
#             low=np.array([-10.0, -10.0]),
#             high=np.array([10.0, 10.0]),
#             dtype=np.float32
#         )
        
#         # 环境参数
#         self.train_mode = train_mode      # 训练模式标志
#         self.max_steps = 200              # 最大步数
#         self.movement_penalty = 0.1       # 动作惩罚系数
#         self.position = np.array([0.0])   # 当前位置
#         self.target = 0.0                 # 目标位置
#         self.current_step = 0

#     def reset(self):
#         # 训练模式随机生成目标（-9到9之间）
#         if self.train_mode:
#             self.target = np.random.uniform(-9.0, 9.0)
#         else:
#             self.target = 0.0  # 测试时由外部设置
            
#         self.position = np.array([0.0])
#         self.current_step = 0
#         return np.array([self.position[0], self.target], dtype=np.float32)
    
#     def step(self, action):
#         self.current_step += 1
        
#         # 执行动作
#         move = np.clip(action[0], -1.0, 1.0) * 2.0
#         self.position += move
#         self.position = np.clip(self.position, -10.0, 10.0)
        
#         # 计算动态距离
#         distance = abs(self.position[0] - self.target)
#         done = False
        
#         # 终止条件
#         if distance <= 0.1:
#             reward = 100.0
#             done = True
#         elif self.current_step >= self.max_steps:
#             reward = -50.0
#             done = True
#         else:
#             reward = -distance - self.movement_penalty * abs(action[0])
            
#         return (
#             np.array([self.position[0], self.target], dtype=np.float32),
#             float(reward),
#             done,
#             {"distance": distance}
#         )
    
#     def render(self, mode='human'):
#         print(f"Pos: {self.position[0]:6.2f} | Target: {self.target:6.2f} | Step: {self.current_step}")

# # 训练代码调整
# def train_agent():
#     # 创建训练环境（自动生成随机目标）
#     env = DynamicTargetContinuousEnv(train_mode=True)
    
#     model = SAC(
#         "MlpPolicy",
#         env,
#         verbose=1,
#         buffer_size=200000,
#         learning_starts=5000,
#         batch_size=512,
#         gamma=0.95,
#         policy_kwargs=dict(net_arch=[256, 256])
#     )
    
#     # 训练30万步（约1小时）
#     model.learn(total_timesteps=300000)
#     model.save("dynamic_target_sac")
#     return model

# # 改进版测试函数
# def test_model(model, target=8.0, episodes=5):
#     env = DynamicTargetContinuousEnv(train_mode=False)
#     env.target = float(target)
    
#     for ep in range(episodes):
#         obs = env.reset()
#         obs[1] = target  # 显式设置目标观测值
#         done = False
#         total_reward = 0
        
#         print(f"\nTesting Target: {target}")
#         while not done:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, done, _ = env.step(action)
#             total_reward += reward
#             env.render()
            
#             # 保持目标信息更新（重要！）
#             obs[1] = target  
            
#         print(f"Final Position: {obs[0]:.2f} | Target: {target} | Total Reward: {total_reward:.1f}")

# # 执行训练和测试
# if __name__ == "__main__":
#     #trained_model = train_agent()
#     env=DynamicTargetContinuousEnv(train_mode=False)
#     model = SAC(
#         "MlpPolicy",
#         env,
#         verbose=1,
#         buffer_size=200000,
#         learning_starts=5000,
#         batch_size=512,
#         gamma=0.95,
#         policy_kwargs=dict(net_arch=[256, 256])
#     )
    
#     trained_model=model.load("dynamic_target_sac")
#     # 测试不同目标
#     test_model(trained_model, target=-9.8)
#     #test_model(trained_model, target=311.5)
#     #test_model(trained_model, target=92.8)

import json

def read_model_info(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    model_list = data['modelList']
    result = []
    
    for model_key, model_data in model_list.items():
        model_info = {
            'ModelType': model_data['ModelType'],
            'PropertyList': model_data['PropertyList'],
            'Posematrix4x3': model_data['Posematrix4x3']
        }
        result.append(model_info)
    
    return result

# 使用示例
if __name__ == "__main__":
    file_path = 'model.json'  # 替换为你的JSON文件路径
    models_info = read_model_info(file_path)
    
    # 打印结果
    for i, model in enumerate(models_info):
        print(f"Model {i}:")
        print(f"  ModelType: {model['ModelType']}")
        print(f"  PropertyList: {model['PropertyList']}")
        print(f"  Posematrix4x3: {model['Posematrix4x3']}")
        print()

import json
import numpy as np
from AxisPathPlanEnv.Prime import Cuboid, Cylinder, Sphere

def create_prime_objects_from_json(json_data):
    """
    从JSON数据创建基元对象
    
    参数:
    json_data: 包含模型信息的JSON数据
    
    返回:
    prime_objects: 包含所有创建的基元对象的列表
    """
    prime_objects = []
    
    # 解析JSON数据
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
        
    model_list = data['modelList']
    
    for model_key, model_data in model_list.items():
        model_type = model_data['ModelType']
        properties = model_data['PropertyList']
        pose_matrix = model_data['Posematrix4x3']
        
        # 从4x3位姿矩阵中提取旋转矩阵和平移向量
        # 前9个元素是3x3旋转矩阵（按行优先存储）
        rotation_matrix = np.array([
            pose_matrix[0:3],
            pose_matrix[3:6],
            pose_matrix[6:9]
        ])
        
        # 最后3个元素是平移向量
        translation = np.array(pose_matrix[9:12])
        
        # 根据模型类型创建相应的基元对象
        if model_type == 'cuboid':
            # 长方体
            length = properties['Length']
            width = properties['width']
            height = properties['height']
            size = [length, width, height]
            
            cuboid = Cuboid(size, translation, rotation_matrix)
            prime_objects.append(cuboid)
            
        elif model_type == 'cylinder':
            # 圆柱体
            height = properties['height']
            radius = properties['radius']
            
            cylinder = Cylinder(height, radius, translation, rotation_matrix)
            prime_objects.append(cylinder)
            
        elif model_type == 'sphere':
            # 球体
            radius = properties['radius']
            
            sphere = Sphere(radius, translation)
            prime_objects.append(sphere)
            
        else:
            print(f"未知的模型类型: {model_type}")
    
    return prime_objects

# 使用示例
if __name__ == "__main__":
    # 从文件读取JSON数据
    with open('model.json', 'r') as file:
        json_data = json.load(file)
    
    # 创建基元对象
    prime_objects = create_prime_objects_from_json(json_data)
    
    # 打印创建的基元对象信息
    for i, obj in enumerate(prime_objects):
        print(f"对象 {i}: 类型={obj.type}, 中心点={obj.centerPoint}, 尺寸={obj.size}")