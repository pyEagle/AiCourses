# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ======================
# GridWorld
# ======================
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
        if action == 0: x -= 1      # up
        if action == 1: x += 1      # down
        if action == 2: y -= 1      # left
        if action == 3: y += 1      # right

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


# ======================
# Actor-Critic
# ======================
class ActorCritic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.pi = tf.keras.layers.Dense(action_dim)
        self.v = tf.keras.layers.Dense(1)

    def call(self, s):
        x = self.fc1(s)
        x = self.fc2(x)
        return self.pi(x), self.v(x)


# ======================
# PPO Agent
# ======================
class PPO:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.entropy_coef = 0.01

        self.model = ActorCritic(state_dim, action_dim)
        self.opt = tf.keras.optimizers.Adam(3e-4)

    def select_action(self, s):
        s = tf.convert_to_tensor(s.reshape(1, -1), dtype=tf.float32)
        logits, v = self.model(s)
        prob = tf.nn.softmax(logits)

        action = np.random.choice(prob.shape[1], p=prob.numpy()[0])
        logp = tf.math.log(prob[0, action] + 1e-8)

        return action, logp.numpy().astype(np.float32), v.numpy()[0, 0].astype(np.float32)

    def compute_gae(self, rewards, values, dones):
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)

        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae

        return adv

    def train(self, states, actions, old_logps, returns, advs):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_logps = tf.convert_to_tensor(old_logps, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advs = tf.convert_to_tensor(advs, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits, values = self.model(states)
            values = tf.squeeze(values, axis=1)

            probs = tf.nn.softmax(logits)
            action_prob = tf.reduce_sum(
                probs * tf.one_hot(actions, probs.shape[1]), axis=1
            )
            logp = tf.math.log(action_prob + 1e-8)

            ratio = tf.exp(logp - old_logps)
            clipped = tf.clip_by_value(ratio, 1 - self.clip, 1 + self.clip) * advs

            loss_pi = -tf.reduce_mean(tf.minimum(ratio * advs, clipped))
            loss_v = tf.reduce_mean(tf.square(returns - values))
            entropy = -tf.reduce_mean(probs * tf.math.log(probs + 1e-8))

            loss = loss_pi + 0.5 * loss_v - self.entropy_coef * entropy

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))


# ======================
# Training Loop
# ======================
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

    values.append(np.float32(0.0))

    advs = agent.compute_gae(rewards, values, dones)
    returns = advs + np.asarray(values[:-1], dtype=np.float32)

    agent.train(
        np.asarray(states, dtype=np.float32),
        actions,
        logps,
        returns,
        advs
    )

    if ep % 50 == 0:
        print(f"Episode {ep:4d} | Reward: {ep_reward:6.2f}")


# ======================
# Policy Visualization
# ======================
arrow = {0: '↑', 1: '↓', 2: '←', 3: '→'}
policy = np.empty((6, 6), dtype=str)

for i in range(6):
    for j in range(6):
        s = np.zeros(36, dtype=np.float32)
        s[i * 6 + j] = 1.0
        logits, _ = agent.model(s.reshape(1, -1))
        a = tf.argmax(logits, axis=1).numpy()[0]
        policy[i, j] = arrow[a]

policy[5, 5] = 'G'

plt.figure(figsize=(5, 5))
for i in range(6):
    for j in range(6):
        plt.text(j, 5 - i, policy[i, j], ha='center', va='center', fontsize=16)

plt.xticks(range(6))
plt.yticks(range(6))
plt.grid()
plt.title("PPO(Arrow Map)")
