# -*- coding: utf-8 -*-

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# =====================
# 1 GridWorld
# =====================
class GridWorld:
    def __init__(self, size=6):
        self.size = size
        self.action_space = 4
        self.state_dim = size * size

        self.goal = (size - 1, size - 1)
        self.obstacles = {(2, 2), (3, 2), (1, 4)}

        self.reset()

    def reset(self):
        while True:
            self.agent = (
                np.random.randint(self.size),
                np.random.randint(self.size)
            )
            if self.agent != self.goal and self.agent not in self.obstacles:
                break
        return self._encode_state()

    def step(self, action):
        row, col = self.agent
        nrow, ncol = row, col

        if action == 0:      # up
            nrow -= 1
        elif action == 1:    # down
            nrow += 1
        elif action == 2:    # left
            ncol -= 1
        elif action == 3:    # right
            ncol += 1

        # 边界检查
        if not (0 <= nrow < self.size and 0 <= ncol < self.size):
            nrow, ncol = row, col

        # 障碍物检查
        if (nrow, ncol) in self.obstacles:
            nrow, ncol = row, col

        self.agent = (nrow, ncol)

        reward = -0.05
        done = False

        if self.agent == self.goal:
            reward = 1.0
            done = True

        return self._encode_state(), reward, done

    def _encode_state(self):
        s = np.zeros(self.size * self.size, dtype=np.float32)
        idx = self.agent[0] * self.size + self.agent[1]
        s[idx] = 1.0
        return s


# =====================
# 2 Actor-Critic
# =====================
class ActorCritic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.fc2 = tf.keras.layers.Dense(128, activation="relu")
        self.policy_head = tf.keras.layers.Dense(action_dim)
        self.value_head = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.policy_head(x), self.value_head(x)


# =====================
# 3 A2C
# =====================
class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99):
        self.gamma = gamma
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def select_action(self, state):
        logits, value = self.model(state[None, :])
        probs = tf.nn.softmax(logits)
        action = np.random.choice(len(probs[0]), p=probs.numpy()[0])
        return action, value.numpy()[0, 0]

    def update(self, states, actions, rewards, values, next_value, done):
        returns = []
        R = 0.0 if done else next_value

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        values = tf.convert_to_tensor(values, dtype=tf.float32)

        advantages = returns - values

        with tf.GradientTape() as tape:
            logits, value_preds = self.model(states)
            value_preds = tf.squeeze(value_preds)

            policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=actions,
                logits=logits
            )
            policy_loss = tf.reduce_mean(policy_loss * tf.stop_gradient(advantages))

            value_loss = tf.reduce_mean(tf.square(returns - value_preds))

            entropy = -tf.reduce_mean(
                tf.nn.softmax(logits) * tf.nn.log_softmax(logits)
            )

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


# =====================
# 4 训练
# =====================
env = GridWorld(size=6)
agent = A2CAgent(env.state_dim, env.action_space)

EPISODES = 600
MAX_STEPS = 80

for ep in range(EPISODES):
    state = env.reset()
    states, actions, rewards, values = [], [], [], []

    for _ in range(MAX_STEPS):
        action, value = agent.select_action(state)
        next_state, reward, done = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)

        state = next_state
        if done:
            break

    _, next_value = agent.model(state[None, :])
    agent.update(states, actions, rewards, values, next_value.numpy()[0, 0], done)

    if ep % 50 == 0:
        print(f"Episode {ep:3d} | steps: {len(rewards)}")


# =====================
# 5 可视化
# =====================
def visualize_policy(env, agent):
    arrow = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    grid = [["" for _ in range(env.size)] for _ in range(env.size)]

    for r in range(env.size):
        for c in range(env.size):
            if (r, c) == env.goal:
                grid[r][c] = "G"
            elif (r, c) in env.obstacles:
                grid[r][c] = "X"
            else:
                s = np.zeros(env.size * env.size, dtype=np.float32)
                s[r * env.size + c] = 1.0
                logits, _ = agent.model(s[None, :])
                a = tf.argmax(logits, axis=1).numpy()[0]
                grid[r][c] = arrow[a]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_xticks(np.arange(env.size + 1))
    ax.set_yticks(np.arange(env.size + 1))
    ax.grid(True)

    for r in range(env.size):
        for c in range(env.size):
            ax.text(
                c + 0.5,
                env.size - r - 0.5,
                grid[r][c],
                ha="center",
                va="center",
                fontsize=16
            )

    ax.set_title("A2C 学到的策略（箭头方向）")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()


visualize_policy(env, agent)
