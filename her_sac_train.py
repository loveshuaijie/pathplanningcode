"""
HER + SAC (self-contained, 不依赖 sb3-contrib)

用法:
  1) 确保 AxisPathPlanEnv/MapEnv.py 是已添加 goal_conditioned 支持的版本。
  2) 安装依赖:
       pip install torch numpy pyyaml gym
  3) 运行训练:
       python her_sac_no_contrib.py
  4) 训练结束后可调用 evaluate(...) 进行评估。

说明:
 - 这是一个研究/原型实现，便于你观察 HER 与 SAC 的交互逻辑并可按需修改。
 - 若你希望把网络/训练细节替换为 stable-baselines3 的实现，我也可以把 HER buffer 输出为 sb3 可用格式。
"""
import os
import yaml
import time
import math
import random
import pickle
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

from AxisPathPlanEnv.MapEnv import MapEnv
from sb3_contrib import HerReplayBuffer

# ----------------------
# GoalObsNormalizer (same idea as earlier)
# ----------------------
class GoalObsNormalizer(gym.ObservationWrapper):
    """
    仅对 observation 字段做 running mean/std 归一化。
    achieved_goal / desired_goal 保持原始坐标以便 HER 精确判断。
    """
    def __init__(self, env: gym.Env, eps: float = 1e-8, clip_range: float = 10.0):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        obs_space = env.observation_space['observation']
        assert isinstance(obs_space, gym.spaces.Box)
        self.obs_dim = obs_space.shape[0]
        self.eps = eps
        self.clip_range = clip_range
        self.count = 0
        self.mean = np.zeros(self.obs_dim, dtype=np.float64)
        self.sq_mean = np.zeros(self.obs_dim, dtype=np.float64)

    def observation(self, obs: Dict[str, Any]):
        o = np.array(obs['observation'], dtype=np.float64)
        if self.count > 1:
            std = np.sqrt(self.sq_mean / (self.count - 1)) + self.eps
            o_norm = (o - self.mean) / std
            o_norm = np.clip(o_norm, -self.clip_range, self.clip_range)
        else:
            o_norm = o.astype(np.float32)
        return {
            'observation': o_norm.astype(np.float32),
            'achieved_goal': np.array(obs['achieved_goal'], dtype=np.float32),
            'desired_goal': np.array(obs['desired_goal'], dtype=np.float32)
        }

    def update(self, obs: Dict[str, Any]):
        o = np.array(obs['observation'], dtype=np.float64)
        self.count += 1
        delta = o - self.mean
        self.mean += delta / self.count
        delta2 = o - self.mean
        self.sq_mean += delta * delta2

    def reset(self, **kwargs):
        raw = self.env.reset(**kwargs)  # raw is dict
        try:
            self.update(raw)
        except Exception:
            pass
        return self.observation(raw)

    def step(self, action):
        raw_next, reward, done, info = self.env.step(action)
        try:
            self.update(raw_next)
        except Exception:
            pass
        return self.observation(raw_next), reward, done, info

    def save_stats(self, path: str):
        data = {'count': self.count, 'mean': self.mean, 'sq_mean': self.sq_mean}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_stats(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.count = data['count']
        self.mean = data['mean']
        self.sq_mean = data['sq_mean']

# ----------------------
# Replay buffer with HER (future strategy)
# ----------------------
class HERReplayBuffer:
    def __init__(self, capacity: int = 1_000_000, k_future: int = 4):
        self.capacity = int(capacity)
        self.k_future = k_future
        # We store transitions as lists to support variable-length episodes
        # Under the hood we'll append flat arrays for sampling
        self.storage = []
        self.ptr = 0

    def add_transitions(self, transitions: List[Dict[str, Any]], env: MapEnv):
        """
        transitions: list of transition dicts from a single episode, in order:
          each dict has keys: 'obs' (dict), 'action' (np.array), 'next_obs'(dict), 'done' (bool)
        We'll:
          - store original transitions (with reward computed via env.compute_reward_from_goal using next_obs)
          - for each transition i, sample k_future future indices and create relabeled transitions where desired_goal becomes achieved_goal[t_future]
        """
        T = len(transitions)
        for i in range(T):
            t = transitions[i]
            # compute reward based on env.compute_reward_from_goal
            achieved_next = t['next_obs']['achieved_goal']
            desired = t['obs']['desired_goal']  # original desired
            r = env.compute_reward_from_goal(achieved_next, desired, sparse=True)
            tr = {
                'obs': t['obs'],
                'action': t['action'],
                'reward': float(r),
                'next_obs': t['next_obs'],
                'done': bool(t['done'])
            }
            self._append(tr)
            # future relabeling
            for _ in range(self.k_future):
                if T <= i + 1:
                    break
                future_idx = np.random.randint(i + 1, T)
                new_goal = transitions[future_idx]['next_obs']['achieved_goal'].copy()
                # create relabeled copy
                rel_obs = {
                    'observation': t['obs']['observation'],
                    'achieved_goal': t['obs']['achieved_goal'],
                    'desired_goal': new_goal
                }
                rel_next_obs = {
                    'observation': t['next_obs']['observation'],
                    'achieved_goal': t['next_obs']['achieved_goal'],
                    'desired_goal': new_goal
                }
                new_r = env.compute_reward_from_goal(rel_next_obs['achieved_goal'], new_goal, sparse=True)
                new_done = bool(env.is_success(rel_next_obs['achieved_goal'], new_goal))
                rel_tr = {
                    'obs': rel_obs,
                    'action': t['action'],
                    'reward': float(new_r),
                    'next_obs': rel_next_obs,
                    'done': new_done
                }
                self._append(rel_tr)

    def _append(self, transition: Dict[str, Any]):
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            # overwrite older
            self.storage[self.ptr] = transition
            self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in idxs]
        return batch

    def __len__(self):
        return len(self.storage)

# ----------------------
# Networks (MLP)
# ----------------------
def mlp(input_dim, output_dim, hidden_sizes=[256,256], activation=nn.ReLU, output_activation=None):
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if output_activation:
        layers.append(output_activation())
    return nn.Sequential(*layers)

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256,256], log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        super().__init__()
        self.net = mlp(obs_dim, hidden_sizes[-1], hidden_sizes=hidden_sizes[:-1], activation=nn.ReLU)
        # use separate heads for mean and log_std
        last_hidden = hidden_sizes[-1]
        self.mean = nn.Linear(last_hidden, action_dim)
        self.log_std = nn.Linear(last_hidden, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x):
        h = self.net(x)
        mu = self.mean(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, x):
        mu, log_std = self.forward(x)
        std = log_std.exp()
        # reparameterization trick
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        tanh_z = torch.tanh(z)
        action = tanh_z
        # log prob correction for tanh
        log_prob = normal.log_prob(z) - torch.log(1 - tanh_z.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mu_action = torch.tanh(mu)
        return action, log_prob, mu_action

    def deterministic(self, x):
        mu, _ = self.forward(x)
        return torch.tanh(mu)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256,256]):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, 1, hidden_sizes=hidden_sizes, activation=nn.ReLU)
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)  # shape (batch,)

# ----------------------
# SAC Agent
# ----------------------
class SACAgent:
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 device='cpu',
                 hidden_sizes=[256,256],
                 lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 alpha='auto',
                 target_entropy=None):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.policy = GaussianPolicy(obs_dim, action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.q1 = QNetwork(obs_dim, action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.q1_target = QNetwork(obs_dim, action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.q2_target = QNetwork(obs_dim, action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)

        # entropy alpha (automatic)
        if alpha == 'auto':
            if target_entropy is None:
                target_entropy = -action_dim
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.target_entropy = target_entropy
        else:
            self.log_alpha = None
            self.alpha = float(alpha)

    def select_action(self, obs_tensor: torch.Tensor, deterministic=False):
        # obs_tensor: shape (batch, obs_dim) or (obs_dim,) (single)
        with torch.no_grad():
            if obs_tensor.dim() == 1:
                obs_t = obs_tensor.unsqueeze(0).to(self.device)
            else:
                obs_t = obs_tensor.to(self.device)
            if deterministic:
                a = self.policy.deterministic(obs_t)
                return a.cpu().numpy()[0]
            else:
                a, _, _ = self.policy.sample(obs_t)
                return a.cpu().numpy()[0]

    def update(self, batch: List[Dict[str, Any]]):
        # convert batch to tensors
        obs = np.stack([b['obs']['observation'] for b in batch], axis=0).astype(np.float32)
        desired = np.stack([b['obs']['desired_goal'] for b in batch], axis=0).astype(np.float32)
        act = np.stack([b['action'] for b in batch], axis=0).astype(np.float32)
        rew = np.stack([b['reward'] for b in batch], axis=0).astype(np.float32)
        next_obs = np.stack([b['next_obs']['observation'] for b in batch], axis=0).astype(np.float32)
        next_des = np.stack([b['next_obs']['desired_goal'] for b in batch], axis=0).astype(np.float32)
        done = np.stack([b['done'] for b in batch], axis=0).astype(np.float32)

        # build network inputs: concat observation + desired_goal
        s = torch.tensor(np.concatenate([obs, desired], axis=-1), dtype=torch.float32, device=self.device)
        s_next = torch.tensor(np.concatenate([next_obs, next_des], axis=-1), dtype=torch.float32, device=self.device)
        a = torch.tensor(act, dtype=torch.float32, device=self.device)
        r = torch.tensor(rew, dtype=torch.float32, device=self.device).unsqueeze(-1)
        d = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # ------------------ Critic update ------------------
        with torch.no_grad():
            # sample next action and logp
            next_a, next_logp, _ = self.policy.sample(s_next)
            q1_next = self.q1_target(s_next, next_a).unsqueeze(-1)
            q2_next = self.q2_target(s_next, next_a).unsqueeze(-1)
            q_next = torch.min(q1_next, q2_next)
            if self.log_alpha is not None:
                alpha = torch.exp(self.log_alpha)
            else:
                alpha = torch.tensor(self.alpha, device=self.device)
            target_q = r + (1 - d) * self.gamma * (q_next - alpha * next_logp)

        # current Q estimates
        q1_val = self.q1(s, a).unsqueeze(-1)
        q2_val = self.q2(s, a).unsqueeze(-1)
        q1_loss = torch.nn.functional.mse_loss(q1_val, target_q)
        q2_loss = torch.nn.functional.mse_loss(q2_val, target_q)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # ------------------ Policy update ------------------
        new_a, logp, _ = self.policy.sample(s)
        q1_pi = self.q1(s, new_a).unsqueeze(-1)
        q2_pi = self.q2(s, new_a).unsqueeze(-1)
        q_pi = torch.min(q1_pi, q2_pi)
        if self.log_alpha is not None:
            alpha = torch.exp(self.log_alpha)
        else:
            alpha = torch.tensor(self.alpha, device=self.device)
        policy_loss = (alpha * logp - q_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # ------------------ Alpha update ------------------
        if self.log_alpha is not None:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha = torch.exp(self.log_alpha).item()
        else:
            alpha_loss = torch.tensor(0.)
            alpha = float(self.alpha)

        # ------------------ Soft updates ------------------
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # return diagnostics
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': alpha
        }

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(dirname, "policy.pt"))
        torch.save(self.q1.state_dict(), os.path.join(dirname, "q1.pt"))
        torch.save(self.q2.state_dict(), os.path.join(dirname, "q2.pt"))
        torch.save(self.q1_target.state_dict(), os.path.join(dirname, "q1_target.pt"))
        torch.save(self.q2_target.state_dict(), os.path.join(dirname, "q2_target.pt"))

    def load(self, dirname):
        self.policy.load_state_dict(torch.load(os.path.join(dirname, "policy.pt"), map_location=self.device))
        self.q1.load_state_dict(torch.load(os.path.join(dirname, "q1.pt"), map_location=self.device))
        self.q2.load_state_dict(torch.load(os.path.join(dirname, "q2.pt"), map_location=self.device))
        self.q1_target.load_state_dict(torch.load(os.path.join(dirname, "q1_target.pt"), map_location=self.device))
        self.q2_target.load_state_dict(torch.load(os.path.join(dirname, "q2_target.pt"), map_location=self.device))

# ----------------------
# Utilities
# ----------------------
def make_env(config: Dict[str, Any]):
    conf = config.copy()
    conf['goal_conditioned'] = True
    env = MapEnv(conf)
    env = GoalObsNormalizer(env)
    return env

# ----------------------
# Training loop
# ----------------------
def train(config_path="AxisPathPlanEnv/env_config.yaml",
          total_timesteps=500_000,
          max_episode_steps=None,
          start_training_after=5000,
          updates_per_step=1,
          batch_size=256,
          buffer_capacity=500_000,
          k_future=4,
          device='cpu',
          seed=0):
    # load config
    with open(config_path, 'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if max_episode_steps is None:
        max_episode_steps = config.get('maxstep', 1000)

    # create env
    env = make_env(config)
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_sample = env.reset()
    obs_dim = obs_sample['observation'].shape[0]
    goal_dim = obs_sample['desired_goal'].shape[0]
    # network input dim = observation_dim + desired_goal_dim
    net_input_dim = obs_dim + goal_dim
    action_dim = env.action_space.shape[0]

    agent = SACAgent(net_input_dim, action_dim, device=device, hidden_sizes=[256,256], lr=3e-4,
                     gamma=0.99, tau=0.005, alpha='auto', target_entropy=-action_dim)
    buffer = HERReplayBuffer(capacity=buffer_capacity, k_future=k_future)

    total_steps = 0
    episode = 0
    start_time = time.time()
    logs = []

    model_dir = "./her_sac_no_contrib_models"
    os.makedirs(model_dir, exist_ok=True)

    while total_steps < total_timesteps:
        obs = env.reset()
        done = False
        ep_transitions = []
        ep_steps = 0
        while not done and ep_steps < max_episode_steps and total_steps < total_timesteps:
            # build network input: concat normalized observation + desired_goal (desired_goal kept raw)
            inp = np.concatenate([obs['observation'], obs['desired_goal']], axis=-1).astype(np.float32)
            # select action (stochastic)
            action = agent.select_action(torch.tensor(inp), deterministic=False)
            # action is in [-1,1] due to tanh
            next_obs, rew_env, done, info = env.step(action)
            # store transition (we store env-normalized observation as returned by GoalObsNormalizer)
            t = {
                'obs': obs,
                'action': action,
                'next_obs': next_obs,
                'done': done
            }
            ep_transitions.append(t)
            obs = next_obs
            total_steps += 1
            ep_steps += 1

            # perform updates if buffer large enough
            if len(buffer) >= batch_size and total_steps > start_training_after:
                for _ in range(updates_per_step):
                    batch = buffer.sample(batch_size)
                    stats = agent.update(batch)
                    logs.append(stats)

        # episode finished, add to buffer with HER relabeling
        buffer.add_transitions(ep_transitions, env)
        episode += 1

        # periodic logging & saving
        if episode % 5 == 0:
            elapsed = time.time() - start_time
            avg_q1 = np.mean([l['q1_loss'] for l in logs[-100:]]) if logs else 0
            avg_policy = np.mean([l['policy_loss'] for l in logs[-100:]]) if logs else 0
            print(f"Episode {episode:04d} total_steps {total_steps} elapsed {elapsed:.1f}s buffer {len(buffer)} avg_q1 {avg_q1:.4f} avg_policy {avg_policy:.4f}")
            # save intermediate model
            agent.save(model_dir)

    # training finished: save final artifacts
    agent.save(model_dir)
    try:
        env.save_stats(os.path.join(model_dir, "goal_obs_normalizer.pkl"))
    except Exception:
        pass
    print("Training finished. Models saved to", model_dir)

# ----------------------
# Evaluation
# ----------------------
def evaluate(model_dir="./her_sac_no_contrib_models", config_path="AxisPathPlanEnv/env_config.yaml", episodes=50, render=False):
    with open(config_path, 'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)
    env = make_env(config)
    # load normalizer stats if exist
    try:
        env.load_stats(os.path.join(model_dir, "goal_obs_normalizer.pkl"))
    except Exception:
        pass

    obs = env.reset()
    obs_dim = obs['observation'].shape[0]
    goal_dim = obs['desired_goal'].shape[0]
    net_input_dim = obs_dim + goal_dim
    action_dim = env.action_space.shape[0]

    agent = SACAgent(net_input_dim, action_dim, device='cpu')
    agent.load(model_dir)

    success = 0
    collisions = 0
    timeouts = 0
    steps_total = 0
    for ep in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < config.get('maxstep', 1000):
            inp = np.concatenate([obs['observation'], obs['desired_goal']], axis=-1).astype(np.float32)
            action = agent.select_action(torch.tensor(inp), deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
            info_obj = info if isinstance(info, dict) else (info[0] if isinstance(info, (list,tuple)) else info)
            term = info_obj.get('terminal', None) if isinstance(info_obj, dict) else None
            if term == "reached_target":
                success += 1
                break
            if term == "collision":
                collisions += 1
                break
            if term == "timeout":
                timeouts += 1
                break
        steps_total += steps
    print(f"Eval {episodes} eps -> success: {success}, collisions: {collisions}, timeouts: {timeouts}, avg_steps: {steps_total/episodes:.1f}")
    return {"success": success, "collisions": collisions, "timeouts": timeouts, "avg_steps": steps_total/episodes}

# ----------------------
# If run as script
# ----------------------
if __name__ == "__main__":
    # train(total_timesteps=int(5e5),
    #       start_training_after=2000,
    #       updates_per_step=1,
    #       batch_size=256,
    #       buffer_capacity=200000,
    #       k_future=4,
    #       device='cpu',
    #       seed=42)
    
    evaluate()