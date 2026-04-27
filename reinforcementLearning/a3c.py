# -*- coding: utf-8 -*-

import threading
import numpy as np
import tensorflow as tf
import time

# =========================
# 超参数
# =========================
GRID_SIZE = 7
STATE_DIM = GRID_SIZE * GRID_SIZE
ACTION_DIM = 4

GAMMA = 0.99
LR = 1e-3
ENTROPY_BETA = 0.01
T_MAX = 10
MAX_EPISODES = 800
NUM_WORKERS = 4

# =========================
# 环境
# =========================
class GridWorld:
    def __init__(self, size):
        self.size = size
        self.goals = [(size - 1, size - 1), (size - 1, 0)]
        self.walls = [(2, 2), (2, 3), (3, 2)]
        self.traps = [(4, 3), (1, 4)]
        self.reset()

    def reset(self):
        self.pos = [0, 0]
        return self._state()

    def _state(self):
        s = np.zeros(self.size * self.size, dtype=np.float32)
        s[self.pos[0] * self.size + self.pos[1]] = 1.0
        return s

    def step(self, action):
        x, y = self.pos
        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1

        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return self._state(), -1.0, False
        if (x, y) in self.walls:
            return self._state(), -1.0, False

        self.pos = [x, y]

        if (x, y) in self.traps:
            return self._state(), -5.0, True
        if (x, y) in self.goals:
            return self._state(), 10.0, True

        return self._state(), -0.02, False

# =========================
# Actor-Critic
# =========================
class ActorCritic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc = tf.keras.layers.Dense(128, activation='relu')
        self.pi = tf.keras.layers.Dense(ACTION_DIM)
        self.v = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.fc(x)
        return self.pi(x), self.v(x)

# =========================
# Worker
# =========================
class Worker(threading.Thread):
    def __init__(self, wid, global_net, optimizer, global_ep):
        super().__init__()
        self.wid = wid
        self.global_net = global_net
        self.optimizer = optimizer
        self.global_ep = global_ep

        self.local_net = ActorCritic()
        dummy = tf.zeros((1, STATE_DIM))
        self.local_net(dummy)
        self.local_net.set_weights(self.global_net.get_weights())

        self.env = GridWorld(GRID_SIZE)

    def run(self):
        while self.global_ep[0] < MAX_EPISODES:
            s = self.env.reset()
            states, actions, rewards = [], [], []
            ep_r = 0

            for _ in range(T_MAX):
                logits, _ = self.local_net(tf.expand_dims(s, 0))
                probs = tf.nn.softmax(logits)[0].numpy()
                a = np.random.choice(ACTION_DIM, p=probs)

                s_, r, done = self.env.step(a)
                states.append(s)
                actions.append(a)
                rewards.append(r)

                s = s_
                ep_r += r
                if done:
                    break

            if done:
                R = 0.0
            else:
                _, v = self.local_net(tf.expand_dims(s, 0))
                R = float(v.numpy()[0, 0])

            returns = []
            for r in reversed(rewards):
                R = r + GAMMA * R
                returns.insert(0, R)

            self.update_global(states, actions, returns)
            self.global_ep[0] += 1

            if self.wid == 0 and self.global_ep[0] % 10 == 0:
                print(f"[Episode {self.global_ep[0]}] reward={ep_r:.2f}")

    def update_global(self, states, actions, returns):
        states = tf.convert_to_tensor(states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.int32)
        returns = tf.convert_to_tensor(returns, tf.float32)

        with tf.GradientTape() as tape:
            logits, values = self.global_net(states)
            values = tf.squeeze(values, 1)
            td = returns - values

            logp = tf.reduce_sum(
                tf.one_hot(actions, ACTION_DIM) *
                tf.nn.log_softmax(logits), axis=1
            )

            entropy = -tf.reduce_sum(
                tf.nn.softmax(logits) *
                tf.nn.log_softmax(logits), axis=1
            )

            loss = (
                -tf.reduce_mean(logp * td)
                + 0.5 * tf.reduce_mean(td ** 2)
                - ENTROPY_BETA * tf.reduce_mean(entropy)
            )

        grads = tape.gradient(loss, self.global_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.global_net.trainable_variables)
        )
        self.local_net.set_weights(self.global_net.get_weights())

# =========================
# 训练
# =========================
def train():
    global_net = ActorCritic()
    dummy = tf.zeros((1, STATE_DIM))
    global_net(dummy)

    optimizer = tf.keras.optimizers.Adam(LR)
    optimizer.build(global_net.trainable_variables)

    global_ep = [0]
    workers = [Worker(i, global_net, optimizer, global_ep)
               for i in range(NUM_WORKERS)]

    start = time.time()
    for w in workers: w.start()
    for w in workers: w.join()

    print("Training finished. Time:", time.time() - start)
    return global_net

# =========================
# 策略可视化
# =========================
def visualize_policy(net):
    env = GridWorld(GRID_SIZE)
    arrows = ['↑', '↓', '←', '→']

    print("\nLearned Policy:")
    for x in range(GRID_SIZE):
        row = []
        for y in range(GRID_SIZE):
            if (x, y) in env.walls:
                row.append('#')
            elif (x, y) in env.traps:
                row.append('T')
            elif (x, y) in env.goals:
                row.append('G')
            else:
                s = np.zeros(STATE_DIM, np.float32)
                s[x * GRID_SIZE + y] = 1.0
                logits, _ = net(tf.expand_dims(s, 0))
                a = tf.argmax(logits, axis=1).numpy().item()
                row.append(arrows[a])
        print(' '.join(row))

# =========================
# main
# =========================
if __name__ == "__main__":
    net = train()
    visualize_policy(net)
