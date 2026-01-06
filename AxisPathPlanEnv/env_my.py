import numpy as np
from AxisPathPlanEnv.Prime import *
from AxisPathPlanEnv.util import *
import fcl
from gym import spaces
from AxisPathPlanEnv.State import State
from AxisPathPlanEnv.Action import Action
import gym
import random

class MapEnv(gym.Env):
    def __init__(self,env_config:dict[str,any],render_mode: str = None)->None:
        self.render_mode = render_mode
        self.xrange = env_config["envxrange"]
        self.yrange = env_config["envyrange"]
        self.zrange = env_config["envzrange"]

        self.obstacles=[] #障碍物列表，初始化为空列表
        # self.append_obstacle(Sphere(radius=0.5,centerPoint=np.array([2,2,3]))) #添加一个球体障碍物
        # self.append_obstacle(Sphere(radius=2,centerPoint=np.array([6,7,5]))) #添加一个球体障碍物
        # self.append_obstacle(Cuboid(size=np.array([2,2,2]),centerPoint=np.array([5,5,5]))) #添加一个长方体障碍物
        # self.append_obstacle(Cylinder(height=2, radius=1,centerPoint=np.array([-5,-5,-5]))) #添加一个圆柱体障碍物
        self.obstacles_num=env_config["obstacles_num"] #障碍物数量

        self.start = np.array(env_config["start"]) #起始点坐标
        self.target =np.array( env_config["target"]) #目标点坐标
        self.startpos=self.start[0:3] #起始点位置坐标
        self.startges=self.start[3:6] #起始点方向向量
        self.targetpos=self.target[0:3] #目标点位置坐标
        self.targetges=self.target[3:6] #目标点方向向量

        self.totalreward=0 #奖励
        self.done=False #是否完成
        self.nowpos=self.startpos #当前点坐标
        self.nowges=self.startges #当前点方向向量

        self.lastpos=self.startpos #上一步点坐标,用于计算距离和速度
        self.lastges=self.startges #上一步点方向向量

        self.last_dis_to_target=0 #上一步距离目标的距离
        self.last_dir_to_target=0
        self.goal_distance=0 #起点距离目标的距离
        self.last_dis_to_obstacle=[0 for i in range(self.obstacles_num)] #上一步距离障碍物的距离集合
        self.lastpath=np.array([0,0,0]) #上一步路径
        self.timestep=0
        self.trajectory=[] #轨迹
        self.tool_size=env_config["tool_size"] #刀具尺寸
        self.moving_tool = Cylinder(height=self.tool_size[0], radius=self.tool_size[1],centerPoint=self.nowpos) #刀具

        self.maxstep=env_config["maxstep"] #最大步数
        self.period=env_config["period"] #每一时间步的周期
        self.safe_distance=env_config["safe_distance"] #安全距离
        self.alpha_max=env_config["alpha_max"] #最大转角
        self.reach_distance=self.goal_distance/20.0 #到达距离
        self.Vmax=1.0

        state=State(self)
        state_dim=len(state.states)
        self.observation_space = spaces.Box(
            low=-10,high=10,shape=(state_dim,),dtype=np.float32
        )
        #self.observation_space=spaces.Box(low=np.array([self.xrange[0],self.yrange[0],self.zrange[0]]),high=np.array([self.xrange[1],self.yrange[1],self.zrange[1]]),dtype=np.float32)
        self.action_space=spaces.Box(low=-1,high=1,shape=(6,),dtype=np.float32)
        self.reset()


    def append_obstacle(self,obstacle): #添加障碍物
        self.obstacles.append(obstacle)

    def reset(self,target_random_flag=True,obstacles_random_flag=True): #重置环境
        self.nowpos=self.startpos
        self.nowges=self.startges

        self.totalreward=0
        self.done=False
        
        obstacles_right_flag=False
        self.moving_tool = Cylinder(height=self.tool_size[0], radius=self.tool_size[1],centerPoint=self.nowpos) #刀具
        #随机生成终点
        if target_random_flag:
            targetpos=np.random.uniform(low=-10, high=10, size=(3,))
            targetges=np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
            self.target=np.concatenate((targetpos,targetges))
            self.targetpos=self.target[0:3]
            self.targetges=self.target[3:6]
            while obstacles_right_flag!=True and obstacles_random_flag==True:
                self.generate_obstacles() #生成障碍物
                obstacles_right_flag=self.is_point_safe(self.startpos)
                if obstacles_right_flag==True:
                    obstacles_right_flag=self.is_point_safe(targetpos)
        self.timestep=0
        self.last_dis_to_target=0
        self.last_dir_to_target=0
        self.goal_distance=calculate_distance(self.startpos,self.targetpos)
        self.goal_dir_to_target=calculate_distance(self.startges,self.targetges)
        self.trajectory=[]
        self.reach_distance=self.goal_distance/50.0
        self.reach_ges=self.goal_dir_to_target/10.0
        if self.reach_distance<0.1:
            self.reach_distance=0.1
        if self.reach_ges<0.1:
            self.reach_ges=0.1
        
        self.state=State(self)
        return self.state.states

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行动作并返回标准Gym接口的四个返回值
        """
        self.lastpos = self.nowpos.copy()  # 记录上一步位置
        self.lastges = self.nowges.copy()  # 记录上一步姿态
        self.lastpath = self.nowpos - self.lastpos  # 记录上一段路径
        
        # 1. 应用动作（直接使用numpy运算）
        scaled_action = action[0:3] * self.Vmax  # 根据配置中的最大速度缩放
        self.nowpos =self.nowpos+ scaled_action * self.period
        self.nowges =self.nowges+ action[3:6] * self.period
        rotation_matrix=euler_to_rotation_matrix(self.nowges[0],self.nowges[1],self.nowges[2])
        self.moving_tool=Cylinder(height=self.tool_size[0], radius=self.tool_size[1],centerPoint=self.nowpos,orientation=rotation_matrix) #刀具位置变化
        self.trajectory.append(self.nowpos.copy())

        self.timestep += 1
        
        # 2. 更新状态（必须放在奖励计算前）
        self.state = State(self)  # 创建新状态对象
        
        # 3. 计算奖励（启用所有奖励组件）
        reward = 0
        reward += self.stepReward()
        reward += self.obstacleAwayReward()  
        reward += self.angleReward()         
        
        # 4. 终止判断
        done = False
        info = {}
        info["nowpos"]=self.nowpos
        info["nowges"]=self.nowges
        info["target"]=self.target
        
        if self.judgeTarget():
            self.trajectory.append(self.target.copy())
            reward += 100
            done = True
            info["terminal"] = "reached_target"
        elif self.judgeObstacle():
            reward += -50
            done = True
            info["terminal"] = "collision"
        elif self.judgeTime():
            reward += -5
            done = True
            info["terminal"] = "timeout"
        
        # 5. 返回标准接口（obs, reward, done, info）
        return self.state.states, reward, done, info  

    def stepReward(self): #时间步奖励
        step_reward=0
        #计算当前点与终点相对距离
        dis=calculate_distance(self.nowpos,self.targetpos)
        dir=calculate_distance(self.nowges,self.targetges)
        base_reward=-dis/self.goal_distance-dir/self.goal_dir_to_target

        direction=self.targetpos-self.nowpos
        movement=self.nowpos-self.lastpos
        if(np.linalg.norm(direction)>1e-5) and (np.linalg.norm(movement)>1e-5):
            cos_sim=np.dot(direction,movement)/(np.linalg.norm(direction)*np.linalg.norm(movement))
            direction_reward=2*cos_sim #方向一致奖励
        else:
            direction_reward=0
        if(dis<self.last_dis_to_target and dir<self.last_dir_to_target):
            step_reward+=1
        else:
            step_reward+=-1

        step_reward+=direction_reward
        step_reward+=base_reward

        self.last_dis_to_target=dis
        self.last_dir_to_target=dir
        return step_reward
    
    def obstacleAwayReward(self):
        obstacle_away_reward = 0
        for k, obstacle in enumerate(self.obstacles):
            dis_to_obstacle = calculate_distance(self.nowpos, obstacle.centerPoint)
            if dis_to_obstacle < self.safe_distance:
                # 奖励与安全距离的比值，距离越近惩罚越大
                obstacle_away_reward -= (self.safe_distance - dis_to_obstacle) / self.safe_distance
        return obstacle_away_reward

    def angleReward(self): #转角奖励
        angle_reward=0
        now_path=np.array(self.nowpos)-np.array(self.lastpos)
        #计算当前路径段与上一段路径的夹角
        if(np.linalg.norm(self.lastpath)>1e-5) and (np.linalg.norm(now_path)>1e-5):
            angle=calculate_angle(now_path,self.lastpath)
        else:
            angle=0
        if(angle>self.alpha_max):
            angle_reward+=-5
        return angle_reward

    def judgeTarget(self): #判断是否到达目标点
        result=False
        if(calculate_distance(self.nowpos,self.targetpos)<self.reach_distance and calculate_distance(self.nowges,self.targetges)<self.reach_ges):
            result=True
        return result

    def judgeObstacle(self):
    # 检查是否超出环境边界
        x, y, z = self.nowpos
        if not (self.xrange[0] <= x <= self.xrange[1] and
                self.yrange[0] <= y <= self.yrange[1] and
                self.zrange[0] <= z <= self.zrange[1]):
            return True
        # 检查障碍物碰撞
        
        for obstacle in self.obstacles:
            request = fcl.CollisionRequest()
            result = fcl.CollisionResult()
            fcl.collide(self.moving_tool.modelforfcl, obstacle.modelforfcl, request, result)
            if result.is_collision:
                return True
        return False

    def judgeTime(self): #判断是否超时
        return self.timestep>=self.maxstep 

    def render(self,pic_path,info):   #绘制环境
        #self.print_info()
        save_plot_3d_path(self.trajectory, np.concatenate((self.xrange, self.yrange, self.zrange), axis=0),self.obstacles,pic_path,info)
        pass

    def generate_obstacles(self):
        """生成多种类型的障碍物"""
        self.obstacles = []
        #num_obstacles = random.randint(self.obstacles_num[0], self.obstacles_num[1])  # 随机生成5到10个障碍物
        
        
        # 静态障碍物
        for _ in range(self.obstacles_num):
            pos = np.array([
                np.random.uniform(self.xrange[0], self.xrange[1]),
                np.random.uniform(self.yrange[0], self.yrange[1]),
                np.random.uniform(self.zrange[0], self.zrange[1])
            ])
            
            # 障碍物类型：球体、圆柱体、长方体
            obs_type = self.np_random.choice(['sphere', 'cylinder', 'box'])
            
            if obs_type == 'sphere':
                radius = np.random.uniform(2, 5)
                obstacle = Sphere(radius, pos)
            elif obs_type == 'cylinder':
                radius = np.random.uniform(1, 3)
                height = np.random.uniform(2, 6)
                obstacle = Cylinder(height, radius, pos)
            else:  # box
                size = np.random.uniform(3, 8, size=3)
                obstacle = Cuboid(size, pos)
            
            self.obstacles.append(obstacle)
    def is_point_safe(self,pointpos):
        #计算点到障碍物体心的距离加上障碍物包围盒的最大半径，判断是否在安全距离内
        for obstacle in self.obstacles:
            if calculate_distance(pointpos, obstacle.centerPoint) < self.safe_distance + obstacle.equivalentRadius:
                return False
        return True