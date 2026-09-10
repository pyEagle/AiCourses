# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

env_rng = np.random.RandomState(SEED)
agent_rng = np.random.RandomState(SEED + 1)
viz_rng = np.random.RandomState(SEED + 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {device}")


# ==========================================
# 1. 环境定义 (GridWorld MDP)
# ==========================================
class GridWorld:
    def __init__(self, size=6, obstacle_ratio=0.2, rng=None):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.rng = rng if rng is not None else np.random.RandomState()
        self._build_valid_map(obstacle_ratio)
        self.reset()

    def _is_path_exists(self, grid):
        queue = deque([self.start])
        visited = {self.start}
        
        while queue:
            x, y = queue.popleft()
            if (x, y) == self.goal:
                return True
                
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if grid[nx, ny] != -1 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    def _build_valid_map(self, obstacle_ratio):
        num_obstacles = int(self.size * self.size * obstacle_ratio)
        max_attempts = 10000
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            self.grid = np.zeros((self.size, self.size), dtype=np.int32)
            obstacles = set()
            while len(obstacles) < num_obstacles:
                x = self.rng.randint(0, self.size)
                y = self.rng.randint(0, self.size)
                if (x, y) not in [self.start, self.goal]:
                    obstacles.add((x, y))

            for (x, y) in obstacles:
                self.grid[x, y] = -1
                
            if self._is_path_exists(self.grid):
                return
                
        raise RuntimeError(f"Exceeded max attempts ({max_attempts}) to generate a valid map.")

    def reset(self):
        self.agent_pos = self.start
        return self._state_index(self.agent_pos)

    def _state_index(self, pos):
        return pos[0] * self.size + pos[1]

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:
            nx, ny = x - 1, y
        elif action == 1:
            nx, ny = x + 1, y
        elif action == 2:
            nx, ny = x, y - 1
        else:
            nx, ny = x, y + 1

        if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
            return self._state_index(self.agent_pos), -1.0, False
        if self.grid[nx, ny] == -1:
            return self._state_index(self.agent_pos), -1.0, False

        self.agent_pos = (nx, ny)

        # 终止状态判断
        if self.agent_pos == self.goal:
            return self._state_index(self.agent_pos), 10.0, True

        return self._state_index(self.agent_pos), -0.1, False


class QLearningAgent:
    def __init__(self, n_states, n_actions,
                 lr=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, rng=None):

        self.n_states = n_states
        self.n_actions = n_actions

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng if rng is not None else np.random.RandomState()

        self.Q = torch.zeros((n_states, n_actions), dtype=torch.float32, device=device)

    def select_action(self, state):
        if self.rng.rand() < self.epsilon:
            return self.rng.randint(self.n_actions)

        q_values = self.Q[state]
        
        max_q = torch.max(q_values)
        best_actions = torch.where(q_values == max_q)[0]
        return int(best_actions[self.rng.randint(len(best_actions))].item())

    def update(self, s, a, r, s_next, done):
        with torch.no_grad():
            q_sa = self.Q[s, a]
            q_next_max = torch.max(self.Q[s_next])
            target = r if done else r + self.gamma * q_next_max
            self.Q[s, a] += self.lr * (target - q_sa)

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


def train():
    env = GridWorld(size=6, obstacle_ratio=0.2, rng=env_rng)
    n_states = env.size * env.size
    n_actions = 4

    agent = QLearningAgent(n_states, n_actions, rng=agent_rng)

    episodes = 800
    rewards_history = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(200):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    return env, agent, rewards_history


def plot_rewards(rewards):
    plt.figure(figsize=(6, 4))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward per Episode")
    plt.title("Q-Learning Convergence Curve")
    plt.grid(True)
    plt.show()


def visualize_policy(env, agent):
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    policy_map = np.full((env.size, env.size), " ", dtype=object)

    for x in range(env.size):
        for y in range(env.size):
            if env.grid[x, y] == -1:
                policy_map[x, y] = "■"
                continue
            if (x, y) == env.goal:
                policy_map[x, y] = "G"
                continue

            s = x * env.size + y
            q_values = agent.Q[s]
            
            max_q = torch.max(q_values)
            best_actions = torch.where(q_values == max_q)[0]
            a = int(best_actions[viz_rng.randint(len(best_actions))].item())
            
            policy_map[x, y] = arrows[a]

    plt.figure(figsize=(5, 5))
    plt.table(cellText=policy_map, loc="center", cellLoc="center")
    plt.axis("off")
    plt.title("Learned Optimal Policy")
    plt.show()


if __name__ == "__main__":
    env, agent, rewards = train()
    plot_rewards(rewards)
    visualize_policy(env, agent)
