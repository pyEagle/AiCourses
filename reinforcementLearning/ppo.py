# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GridWorld:
    def __init__(self, size=6):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.pos = [0, 0]
        return self._state()

    def _state(self):
        s = np.zeros(self.size * self.size, dtype=np.float32)
        idx = self.pos[0] * self.size + self.pos[1]
        s[idx] = 1.0
        return s

    def step(self, action):
        x, y = self.pos
        if action == 0: x -= 1
        if action == 1: x += 1
        if action == 2: y -= 1
        if action == 3: y += 1

        reward = -0.01
        done = False

        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            reward = -1.0
        else:
            self.pos = [x, y]

        if tuple(self.pos) == self.goal:
            reward = 10.0
            done = True

        return self._state(), reward, done


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.pi = nn.Linear(128, action_dim)
        self.v = nn.Linear(128, 1)

    def forward(self, s):
        x = self.common(s)
        return self.pi(x), self.v(x)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.entropy_coef = 0.01
        self.epochs = 4
        self.batch_size = 64

        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=3e-4)

    def select_action(self, s):
        s = torch.from_numpy(s).float().to(device).unsqueeze(0)
        with torch.no_grad():
            logits, v = self.model(s)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)

        return action.item(), logp.item(), v.item()

    def compute_gae(self, rewards, values, dones):
        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
        return adv

    def train(self, states, actions, old_logps, returns, advs):
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        old_logps = torch.tensor(old_logps, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advs = torch.tensor(advs, dtype=torch.float32).to(device)

        N = states.size(0)
        idxs = np.arange(N)

        for _ in range(self.epochs):
            np.random.shuffle(idxs)

            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logps = old_logps[batch_idx]
                b_returns = returns[batch_idx]
                b_advs = advs[batch_idx]

                logits, values = self.model(b_states)
                values = values.squeeze()

                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - b_old_logps)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_advs
                loss_pi = -torch.min(surr1, surr2).mean()

                value_pred_clipped = b_returns + (values - b_returns).clamp(-0.2, 0.2)
                value_losses = (values - b_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - b_returns).pow(2)
                loss_v = torch.max(value_losses, value_losses_clipped).mean()

                loss = loss_pi + 0.5 * loss_v - self.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.opt.step()


env = GridWorld(size=6)
agent = PPO(state_dim=36, action_dim=4)

EPISODES = 800

for ep in range(EPISODES):
    s = env.reset()
    states, actions, rewards, logps, values, dones = [], [], [], [], [], []
    done = False
    ep_reward = 0.0

    while not done:
        a, logp, v = agent.select_action(s)
        s2, r, done = env.step(a)

        states.append(s)
        actions.append(a)
        rewards.append(r)
        logps.append(logp)
        values.append(v)
        dones.append(done)

        s = s2
        ep_reward += r

    values.append(0.0)

    advs = agent.compute_gae(rewards, values, dones)
    returns = advs + np.array(values[:-1], dtype=np.float32)

    agent.train(
        np.array(states),
        np.array(actions),
        np.array(logps),
        returns,
        advs
    )

    if ep % 50 == 0:
        print(f"Episode {ep:4d} | Reward: {ep_reward:6.2f}")


arrow = {0: '↑', 1: '↓', 2: '←', 3: '→'}
policy = np.empty((6, 6), dtype=object)

agent.model.eval()
with torch.no_grad():
    for i in range(6):
        for j in range(6):
            s = np.zeros(36, dtype=np.float32)
            s[i * 6 + j] = 1.0
            s_ts = torch.from_numpy(s).to(device).unsqueeze(0)
            logits, _ = agent.model(s_ts)
            a = torch.argmax(logits, dim=1).item()
            policy[i, j] = arrow[a]

policy[5, 5] = 'G'

plt.figure(figsize=(6, 6))
for i in range(6):
    for j in range(6):
        plt.text(j, 5 - i, policy[i, j], ha='center', va='center', fontsize=20)

plt.xticks(range(6))
plt.yticks(range(6))
plt.grid(True)
plt.title("PPO policy Map (Standard PPO)")
plt.show()

