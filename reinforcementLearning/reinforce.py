# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.agent = (0, 0)
        return self._get_state()

    def _get_state(self):
        state = np.zeros((self.size, self.size), dtype=np.float32)
        state[self.agent] = 1.0
        return state.flatten()

    def step(self, action):
        x, y = self.agent
        if action == 0:   # up
            x -= 1
        elif action == 1: # down
            x += 1
        elif action == 2: # left
            y -= 1
        elif action == 3: # right
            y += 1

        x = np.clip(x, 0, self.size - 1)
        y = np.clip(y, 0, self.size - 1)
        self.agent = (x, y)

        if self.agent == self.goal:
            reward = 10.0
            done = True
        else:
            reward = -0.1
            done = False
        return self._get_state(), reward, done

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNet(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        probs = self.policy(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update(self, rewards, log_probs):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 计算 Loss: -log_prob * Gt
        loss = []
        for log_prob, Gt in zip(log_probs, returns):
            loss.append(-log_prob * Gt)
        
        loss = torch.stack(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train():
    env = GridWorld(size=5)
    state_dim = 25
    action_dim = 4

    agent = REINFORCEAgent(state_dim, action_dim)
    rewards_history = []

    for episode in range(800):
        state = env.reset()
        log_probs, rewards = [], []

        for step in range(50):
            action, log_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            if done:
                break

        agent.update(rewards, log_probs)
        total_reward = sum(rewards)
        rewards_history.append(total_reward)

        if episode % 50 == 0:
            print(f"Episode {episode}, total reward = {total_reward:.2f}")

    return agent, env, rewards_history

def visualize_policy(agent, size=5):
    arrow_map = {0: (0, 0.4), 1: (0, -0.4), 2: (-0.4, 0), 3: (0.4, 0)}

    plt.figure(figsize=(6, 6))
    for x in range(size):
        for y in range(size):
            state = np.zeros((size, size), dtype=np.float32)
            state[x, y] = 1.0
            state_tensor = torch.from_numpy(state.flatten()).to(device)
            
            with torch.no_grad():
                probs = agent.policy(state_tensor)
                action = torch.argmax(probs).item()
            
            dx, dy = arrow_map[action]
            plt.arrow(y, size - 1 - x, dx, dy, 
                      head_width=0.15, length_includes_head=True, color='blue')

    plt.scatter(size - 1, 0, c="red", s=200, marker="*", label="Goal")
    plt.grid(True)
    plt.xticks(range(size))
    plt.yticks(range(size))
    plt.title("Learned Policy (PyTorch REINFORCE)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    trained_agent, env, history = train()
    visualize_policy(trained_agent)

