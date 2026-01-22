import numpy as np
import gymnasium as gym
from gymnasium import spaces
import fcl
from typing import Tuple, Dict, Any, Union, Optional

# 引用你的自定义模块
from AxisPathPlanEnv.Prime import Cylinder, Sphere, Cuboid
from AxisPathPlanEnv.util import calculate_distance, calculate_angle, euler_to_rotation_matrix, save_plot_3d_path

class MapEnv(gym.Env):
    """
    基于 Gymnasium 标准重构的 3D 路径规划环境
    支持 HER (Hindsight Experience Replay) 和 FCL 碰撞检测
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, env_config: Dict[str, Any], render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        MAX_OBS_DETECT = 10
        # --- 1. 配置加载 ---
        self.x_range = np.array(env_config["envxrange"])
        self.y_range = np.array(env_config["envyrange"])
        self.z_range = np.array(env_config["envzrange"])
        self.obstacles_num = env_config["obstacles_num"]
        
        # 物理/工具参数
        self.tool_size = env_config["tool_size"] 
        self.max_step = env_config["maxstep"] 
        self.dt = env_config["period"] 
        self.safe_distance = env_config["safe_distance"] 
        self.alpha_max = env_config["alpha_max"] 
        self.v_max = env_config.get("Vmax", 1.0)
        
        # 缩放因子
        self.reach_pos_scale = env_config.get("reachpos_scale", 10.0)
        self.reach_ges_scale = env_config.get("reachges_scale", 10.0)

        # HER 配置
        self.goal_conditioned = env_config.get("goal_conditioned", False)
        self.reward_type = env_config.get("reward_type", "dense") # 'sparse' or 'dense'

        # --- 2. 状态变量初始化 ---
        self.start_pos = np.array(env_config["start"][0:3], dtype=np.float32)
        self.start_ges = np.array(env_config["start"][3:6], dtype=np.float32)
        self.target_pos = np.array(env_config["target"][0:3], dtype=np.float32)
        self.target_ges = np.array(env_config["target"][3:6], dtype=np.float32)

        self.now_pos = self.start_pos.copy()
        self.now_ges = self.start_ges.copy()
        self.last_pos = self.start_pos.copy()
        self.last_ges = self.start_ges.copy()
        
        self.obstacles = []
        self.moving_tool = None # 在 reset 中初始化
        self.timestep = 0

        # --- 3. 定义空间 (Action & Observation) ---
        # 动作: [vx, vy, vz, d_roll, d_pitch, d_yaw]
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # 预计算观测维度 (Pos(3) + Ges(3) + Vel(3) + AngVel(3) + ObstacleInfo(Num*4))
        # 注意：这里假设每个障碍物提供距离(1) + 方向(3) = 4维信息
        

        if self.goal_conditioned:
            obs_dim = 6 + 6 + (MAX_OBS_DETECT * 4) 
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            })
        else:
            # 非 HER 模式通常需要把 Target 相对位置放入 Observation
            # 这里为了简化，保持与 feature_dim 一致，但在 _get_obs 中拼接 target 差值
            flat_dim = self.obs_feature_dim + 6 + 2 # + TargetRel(6) + Dists(2)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(flat_dim,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        # 1. Gymnasium 标准随机数初始化
        super().reset(seed=seed) 
        
        # 解析 options
        options = options or {}
        target_random = options.get('target_random', True)
        obstacles_random = options.get('obstacles_random', True)

        # 2. 重置状态
        self.timestep = 0
        self.now_pos = self.start_pos.copy()
        self.now_ges = self.start_ges.copy()
        self.last_pos = self.start_pos.copy()
        self.last_ges = self.start_ges.copy()
        self.trajectory = [self.now_pos.copy()]
        
        # 3. 初始化工具模型
        self._update_tool_model()

        # 4. 生成场景
        if target_random:
            self._generate_random_target()
        
        # 总是重新生成障碍物，确保其位置有效性
        if obstacles_random or not self.obstacles:
            self._generate_valid_obstacles()
        
        # 5. 更新距离阈值 (根据任务难度动态调整)
        self.goal_dist_init = calculate_distance(self.start_pos, self.target_pos)
        self.goal_ges_init = calculate_distance(self.start_ges, self.target_ges)
        
        # 这里的逻辑保留你原有的：阈值随距离动态变化，但不能小于 0.1
        self.reach_dist_threshold = max(self.goal_dist_init / self.reach_pos_scale, 0.1)
        self.reach_ges_threshold = max(self.goal_ges_init / self.reach_ges_scale, 0.1)

        # 6. 获取初始观测
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        # 1. 记录上一步
        self.last_pos = self.now_pos.copy()
        self.last_ges = self.now_ges.copy()
        
        # 2. 物理更新
        # 动作裁剪与缩放
        action = np.clip(action, self.action_space.low, self.action_space.high)
        scaled_pos_act = action[0:3] * self.v_max
        scaled_ges_act = action[3:6] # 假设姿态速度不需要额外缩放，或者也在 v_max 内
        
        self.now_pos += scaled_pos_act * self.dt
        self.now_ges += scaled_ges_act * self.dt
        
        #self._update_tool_model()
        self.trajectory.append(self.now_pos.copy())
        self.timestep += 1

        # 3. 获取观测
        obs = self._get_obs()
        
        # 4. 判定状态
        is_success = self._is_success(self.now_pos, self.now_ges, self.target_pos, self.target_ges)
        is_collision = self._check_collision()
        is_out_of_bounds = self._check_out_of_bounds()
        
        # 5. 终止条件 (Gymnasium 分为 terminated 和 truncated)
        terminated = is_success or is_collision or is_out_of_bounds
        truncated = self.timestep >= self.max_step
        
        info = self._get_info()
        info['is_success'] = is_success
        info['is_collision'] = is_collision

        # 6. 计算奖励
        if self.goal_conditioned:
            reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        else:
            reward = self._compute_dense_reward(is_success, is_collision or is_out_of_bounds, truncated)

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        HER 必须实现的接口，支持 Batch 计算
        """
        # 确保输入是 numpy 数组
        ag = np.array(achieved_goal)
        dg = np.array(desired_goal)
        
        # 如果是单个样本，扩展维度以便统一处理
        if ag.ndim == 1:
            ag = ag.reshape(1, -1)
            dg = dg.reshape(1, -1)
            
        # 计算距离 (位置 + 姿态)
        # 假设前3维是位置，后3维是姿态
        d_pos = np.linalg.norm(ag[:, :3] - dg[:, :3], axis=1)
        d_ges = np.linalg.norm(ag[:, 3:] - dg[:, 3:], axis=1)
        
        # 判定是否成功 (Batch wise)
        # 注意：这里使用 self.reach_dist_threshold 可能有问题，因为 HER 回放时阈值应该是当初那个 episode 的
        # 但为了简化，通常取一个固定的小值，或者认为当前环境参数是全局通用的
        success = (d_pos < self.reach_dist_threshold) & (d_ges < self.reach_ges_threshold)
        
        if self.reward_type == 'sparse':
            # 成功 0，失败 -1
            return (success.astype(np.float32) - 1.0).squeeze()
        else:
            # 密集奖励: 负距离
            return -(d_pos + d_ges).squeeze()

    # ================= 内部辅助方法 =================

    def _get_obs(self) -> Union[Dict, np.ndarray]:
        """生成观测向量，替代原本的 State.py"""
        # 1. 基础物理量
        pos_vel = self.now_pos - self.last_pos
        ges_vel = self.now_ges - self.last_ges
        
        # --- 核心修改：处理障碍物 ---
        all_obs_info = []
        MAX_OBS_DETECT = 10
        req = fcl.DistanceRequest(enable_nearest_points=True)
        
        # 1. 遍历所有障碍物，计算距离并存储
        temp_obstacles = []
        for obs in self.obstacles:
            res = fcl.DistanceResult()
            dist = fcl.distance(obs.modelforfcl, self.moving_tool.modelforfcl, req, res)
            
            vec = obs.centerPoint - self.now_pos
            # 简单的防除零
            norm_vec = vec / (np.linalg.norm(vec) + 1e-6)
            
            temp_obstacles.append({
                'dist': dist,
                'vec': norm_vec
            })
            
        # 2. 按距离从小到大排序 (关注最近的危险)
        temp_obstacles.sort(key=lambda x: x['dist'])
        
        # 3. 截断与填充
        obs_features = []
        for i in range(MAX_OBS_DETECT):
            if i < len(temp_obstacles):
                # 有真实障碍物
                o = temp_obstacles[i]
                obs_features.extend([o['dist'], *o['vec']])
            else:
                # 填充“虚拟障碍物” (放在无穷远处，或者 safe_distance 之外很远)
                # 建议：dist 设为一个大数 (如 100.0)，方向设为 0
                obs_features.extend([20.0, 0.0, 0.0, 0.0])
                
        # 转换 convert list to array...
        # 拼接到最终的 self.states 中

        # 3. 组装 HER Observation (纯净的状态，不包含 Target)
        # 包含：速度, 绝对坐标, 绝对姿态, 障碍物信息
        obs_vec = np.concatenate([
            pos_vel,
            ges_vel,
            self.now_pos, 
            self.now_ges,
            np.array(obs_features, dtype=np.float32)
        ]).astype(np.float32)

        if self.goal_conditioned:
            return {
                'observation': obs_vec,
                'achieved_goal': np.concatenate([self.now_pos, self.now_ges]).astype(np.float32),
                'desired_goal': np.concatenate([self.target_pos, self.target_ges]).astype(np.float32)
            }
        else:
            # 非 HER 模式：需要把目标相对位置加进去
            rel_pos = self.target_pos - self.now_pos
            rel_ges = self.target_ges - self.now_ges
            dist_p = calculate_distance(self.now_pos, self.target_pos)
            dist_g = calculate_distance(self.now_ges, self.target_ges)
            
            flat_obs = np.concatenate([
                rel_pos, rel_ges, obs_vec, [dist_p, dist_g]
            ]).astype(np.float32)
            return flat_obs

    def _update_tool_model(self):
        rotation_matrix = euler_to_rotation_matrix(self.now_ges[0], self.now_ges[1], self.now_ges[2])
        self.moving_tool = Cylinder(
            height=self.tool_size[0], 
            radius=self.tool_size[1], 
            centerPoint=self.now_pos, 
            orientation=rotation_matrix
        )

    def _generate_random_target(self):
        # 使用 gymnasium 提供的随机数生成器
        pos = self.np_random.uniform(low=-10, high=10, size=(3,))
        ges = self.np_random.uniform(low=-np.pi, high=np.pi, size=(3,))
        self.target_pos = pos
        self.target_ges = ges

    def _generate_valid_obstacles(self):
        max_attempts = 100
        self.obstacles = []
        
        for _ in range(self.obstacles_num):
            for _ in range(max_attempts):
                # 随机生成位置
                pos = np.array([
                    self.np_random.uniform(self.x_range[0], self.x_range[1]),
                    self.np_random.uniform(self.y_range[0], self.y_range[1]),
                    self.np_random.uniform(self.z_range[0], self.z_range[1])
                ])
                
                # 随机类型
                obs_type = self.np_random.choice(['sphere', 'cylinder', 'box'])
                if obs_type == 'sphere':
                    obs = Sphere(self.np_random.uniform(2, 5), pos)
                elif obs_type == 'cylinder':
                    obs = Cylinder(self.np_random.uniform(2, 6), self.np_random.uniform(1, 3), pos)
                else:
                    obs = Cuboid(self.np_random.uniform(3, 8, size=3), pos)
                
                # 简单检查：起点和终点必须是安全的
                # 注意：这里仅检查质心距离，严格来说应该做 Collision Check
                d_start = calculate_distance(self.start_pos, pos)
                d_target = calculate_distance(self.target_pos, pos)
                safe_margin = self.safe_distance + obs.equivalentRadius
                
                if d_start > safe_margin and d_target > safe_margin:
                    self.obstacles.append(obs)
                    break

    def _check_collision(self) -> bool:
        req = fcl.CollisionRequest()
        
        for obs in self.obstacles:
            res = fcl.CollisionResult()
            fcl.collide(self.moving_tool.modelforfcl, obs.modelforfcl, req, res)
            if res.is_collision:
                return True
        return False

    def _check_out_of_bounds(self) -> bool:
        x, y, z = self.now_pos
        return not (self.x_range[0] <= x <= self.x_range[1] and
                    self.y_range[0] <= y <= self.y_range[1] and
                    self.z_range[0] <= z <= self.z_range[1])

    def _is_success(self, pos, ges, target_pos, target_ges) -> bool:
        d_pos = calculate_distance(pos, target_pos)
        d_ges = calculate_distance(ges, target_ges)
        return (d_pos < self.reach_dist_threshold) and (d_ges < self.reach_ges_threshold)

    def _compute_dense_reward(self, success, collision, timeout):
        # 1. 基础距离奖励 (归一化)
        d_pos = calculate_distance(self.now_pos, self.target_pos)
        d_ges = calculate_distance(self.now_ges, self.target_ges)
        r_dist = -(d_pos / self.goal_dist_init) - (d_ges / self.goal_ges_init)
        
        # 2. 引导奖励 (Direction Reward)
        r_dir = 0
        vec_to_target = self.target_pos - self.now_pos
        vec_move = self.now_pos - self.last_pos
        if np.linalg.norm(vec_to_target) > 1e-5 and np.linalg.norm(vec_move) > 1e-5:
            cos_sim = np.dot(vec_to_target, vec_move) / (np.linalg.norm(vec_to_target) * np.linalg.norm(vec_move))
            r_dir = 2.0 * cos_sim
            
        # 3. 避障奖励
        r_obs = 0
        for obs in self.obstacles:
            d = calculate_distance(self.now_pos, obs.centerPoint)
            if d < self.safe_distance:
                r_obs -= (self.safe_distance - d) / self.safe_distance

        # 4. 事件奖励
        r_event = 0
        if success: r_event += 100
        if collision: r_event -= 50
        if timeout: r_event -= 5
        
        return r_dist + r_dir + r_obs + r_event

    def _get_info(self):
        return {
            "pos": self.now_pos.copy(),
            "target": self.target_pos.copy()
        }
    
    def render(self):
        if self.render_mode == "human":
            # 调用 util 中的绘图，或者这里实现简单的 print
            pass
        elif self.render_mode == "rgb_array":
            # 返回图像数组
            pass