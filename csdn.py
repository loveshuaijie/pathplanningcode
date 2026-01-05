import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
 
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
 
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu_head = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 使用nn.Parameter定义log_std
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mu_head, gain=0.01)
 
 
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = 2.0*torch.tanh(self.mu_head(x))  # 输出均值在[-2, 2]之间
        std = torch.exp(self.log_std)
        return mu, std
 
    def get_dist(self, s):
        mean,std = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = torch.distributions.Normal(mean, std)  # Get the Gaussian distribution
        return dist
 
# 定义Critic网络
import torch.nn.functional as F
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value
 
class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
 
    def update(self, x):#动态更新平均值和标准差可以用到在线算法（online algorithm），其中最常见的方法是Welford的算法
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)
 
 
class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)
 
    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
 
        return x
 
# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, update_steps=10):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
 
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-3)
        self.lr= lr
 
        self.gamma = gamma
        self.K_epochs = 10
        self.clip_epsilon = clip_epsilon
        self.epsilon = clip_epsilon
        self.entropy_coef = 0.01
        self.update_steps = update_steps
 
        # 同步actor和old_actor的参数
 
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.actor(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action = torch.clamp(action, -2, 2)
        return action.cpu().numpy().flatten(), dist.log_prob(action).sum(axis=-1).item()
 
 
    def lr_decay(self, ep):
        lr_a_now = self.lr * (1 - ep / 3000)
        lr_c_now = self.lr * (1 - ep / 3000)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_c_now
 
    def update1(self, replay_buffer,ep):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
     
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * 0.95 * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1) #tensor [2048,1]
            v_target = adv + vs #tensor[2048,1]
 
 
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
 
 
            dist_now = self.actor.get_dist(s)
            dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
            a_logprob_now = dist_now.log_prob(a)
 
            ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob.sum(1, keepdim=True))  # shape(mini_batch_size X 1)
            # print(a_logprob_now- a_logprob[index],ratios)
 
 
            surr1 = ratios * adv  # Only calculate the gradient of 'a_logprob_now' in ratios
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv
            actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
 
            v_s = self.critic(s)
            critic_loss = F.mse_loss(v_target, v_s)
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
 
        self.lr_decay(ep)
 
# 环境和训练设置
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
 
agent = PPOAgent(state_dim, action_dim)
 
max_episodes = 3000
max_steps = 200
batch_size = 64
 
state_norm = Normalization(shape=state_dim)  # Trick 2:state normalization
class ReplayBuffer:
    def __init__(self, batch_size,state_dim,action_dim):
        self.s = np.zeros((batch_size, state_dim))
        self.a = np.zeros((batch_size, action_dim))
        self.a_logprob = np.zeros((batch_size, action_dim))
        self.r = np.zeros((batch_size, 1))
        self.s_ = np.zeros((batch_size, state_dim))
        self.dw = np.zeros((batch_size, 1))
        self.done = np.zeros((batch_size, 1))
        self.count = 0
 
    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1
 
    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
 
        done = torch.tensor(self.done, dtype=torch.float)
 
        return s, a, a_logprob, r, s_, dw, done
 
class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)
 
    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x
 
    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)
replay_buffer = ReplayBuffer(128,state_dim,action_dim)
reward_scaling = RewardScaling(shape=1, gamma=0.99)
env_evaluate = gym.make("Pendulum-v1",render_mode="human")
# env_evaluate = gym.make("Pendulum-v1")
for episode in range(max_episodes):
    state,_ = env.reset()
    episode_reward = 0
    memory = []
    done = False
    ter = False
    step = 0
    while not done:
        step +=1
        action, log_prob = agent.select_action(state)
        next_state, reward, done,ter, _ = env.step(action)
        reward = reward_scaling(reward)
        if(abs(reward)<0.01):
            done=True
        epend = 0
 
        episode_reward += reward
 
        dw = False
        if done:
            dw = True
 
        end = False
        if done or ter:
            end = True
            done = True
 
        replay_buffer.store(state, action, log_prob, reward, next_state, dw, end)
        if replay_buffer.count == 128:
            agent.update1(replay_buffer,episode)
            replay_buffer.count = 0
 
        state = next_state
 
    if episode % 1 == 0:
        print(f'Episode {episode}, {step},Reward: {episode_reward}')
    # if (episode+1) % 200 == 0:
    #     for i in range (3):
    #         s, _ = env_evaluate.reset()
    #         er = 0
    #         done1 = False
    #         ss = 0
    #         while not done1:
    #             ss +=1
    #             s = torch.FloatTensor(s).unsqueeze(0)
    #             a,_ = agent.actor(s)  # We use the deterministic policy during the evaluating
    #             s_, r, done1,ter, _ = env_evaluate.step(a.detach().numpy().flatten())
    #             if ter:
    #                 done1 = True
    #             er += r
    #             s = s_
    #         print(f'evaluate step {ss},episode_reward {er}')