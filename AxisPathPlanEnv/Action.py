import numpy as np
import random
class Action:
    def __init__(self,action:list, env_config): #将轴的速度作为动作空间，动作空间五个参数都是连续变量
        self.Vmax=env_config["Vmax"]
        self.Vmin=0.0
        #self.wmax=env_config["wmax"]
        #self.wmin=0.0
        self.action = action

    def random_action(self):
        V=[random.uniform(self.Vmin,self.Vmax) for _ in range(3)]
       # w=[random.uniform(self.wmin,self.wmax) for _ in range(2)]
        self.action = [V]
        return self.action
    
    def apply_action(self, env): #将动作作用于环境
        stepvec=np.array([i*env.period for i in self.action])
        env.nowpos=env.nowpos+stepvec
        return np.array(self.action)

    
