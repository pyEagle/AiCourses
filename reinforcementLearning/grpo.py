# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ======================
# GridWorld
# ======================
class GridWorld5x5:
    def __init__(self):
        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)

    def reset(self):
        self.pos = [0, 0]
        return self._state()

    def _state(self):
        s = np.zeros(25, dtype=np.float32)
        s[self.pos[0] * 5 + self.pos[1]] = 1.0
        return s

    def step(self, a):
        x, y = self.pos
        if a == 0: x -= 1
        elif a == 1: x += 1
        elif a == 2: y -= 1
        elif a == 3: y += 1

        r = -0.05
        done = False

        if 0 <= x < 5 and 0 <= y < 5:
            self.pos = [x, y]
        else:
            r = -0.2

        if self.pos == list(self.goal):
            r = 1.0
            done = True

        return self._state(), r, done


# ======================
# Networks
# ======================
class Policy(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(64, activation='tanh')
        self.d2 = tf.keras.layers.Dense(64, activation='tanh')
        self.out = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.out(self.d2(self.d1(x)))

class Value(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(64, activation='tanh')
        self.d2 = tf.keras.layers.Dense(64, activation='tanh')
        self.out = tf.keras.layers.Dense(1)

    def call(self, x):
        return tf.squeeze(self.out(self.d2(self.d1(x))), axis=1)

policy = Policy()
value_fn = Value()
value_opt = tf.keras.optimizers.Adam(1e-3)

# ======================
# Utils
# ======================
def flat_grads(grads, vars):
    out = []
    for g, v in zip(grads, vars):
        if g is None:
            out.append(tf.zeros_like(tf.reshape(v, [-1])))
        else:
            out.append(tf.reshape(g, [-1]))
    return tf.concat(out, axis=0)

def flat_vars(vars):
    return tf.concat([tf.reshape(v, [-1]) for v in vars], axis=0)

def assign_flat(vars, flat):
    idx = 0
    for v in vars:
        size = tf.size(v)
        v.assign(tf.reshape(flat[idx:idx + size], v.shape))
        idx += size

def returns(rews, gamma=0.99):
    out, g = [], 0
    for r in reversed(rews):
        g = r + gamma * g
        out.insert(0, g)
    return np.array(out, np.float32)

def kl(old, new):
    p = tf.nn.softmax(old)
    logp = tf.nn.log_softmax(old)
    logq = tf.nn.log_softmax(new)
    return tf.reduce_mean(tf.reduce_sum(p * (logp - logq), axis=1))

# ======================
# Fisher Vector Product
# ======================
def fisher_vector_product(states, vec):
    with tf.GradientTape(persistent=True) as tape:
        logits = policy(states)
        old = tf.stop_gradient(logits)
        kl_val = kl(old, logits)

    grads = tape.gradient(kl_val, policy.trainable_variables)
    g_flat = flat_grads(grads, policy.trainable_variables)

    gv = tf.reduce_sum(g_flat * vec)
    grads2 = tape.gradient(gv, policy.trainable_variables)
    del tape

    return flat_grads(grads2, policy.trainable_variables) + 0.1 * vec

# ======================
# Conjugate Gradient
# ======================
def conjugate_gradient(fvp, b, iters=10):
    x = tf.zeros_like(b)
    r = b
    p = b
    rr = tf.reduce_sum(r * r)

    for _ in range(iters):
        Ap = fvp(p)
        alpha = rr / (tf.reduce_sum(p * Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        rr_new = tf.reduce_sum(r * r)
        p = r + (rr_new / (rr + 1e-8)) * p
        rr = rr_new

    return x

# ======================
# Training
# ======================
env = GridWorld5x5()
max_kl = 0.01

for ep in range(100):
    s = env.reset()
    states, actions, rewards = [], [], []
    done = False

    count = 0
    while not done and count < 100:
        logits = policy(tf.expand_dims(s, 0))
        probs = tf.nn.softmax(logits).numpy()[0]
        a = np.random.choice(4, p=probs)
        s, r, done = env.step(a)

        states.append(s)
        actions.append(a)
        rewards.append(r)
        count += 1

    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    rets = returns(rewards)
    vals = value_fn(states).numpy()

    adv = rets - vals
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    old_logits = tf.stop_gradient(policy(states))

    with tf.GradientTape() as tape:
        logits = policy(states)
        logp = tf.nn.log_softmax(logits)
        logp_old = tf.nn.log_softmax(old_logits)
        ratio = tf.exp(
            tf.reduce_sum(tf.one_hot(actions, 4) * logp, axis=1) -
            tf.reduce_sum(tf.one_hot(actions, 4) * logp_old, axis=1)
        )
        loss = tf.reduce_mean(ratio * adv)

    grads = tape.gradient(loss, policy.trainable_variables)
    g = flat_grads(grads, policy.trainable_variables)

    step = conjugate_gradient(lambda x: fisher_vector_product(states, x), g)
    shs = 0.5 * tf.reduce_sum(step * fisher_vector_product(states, step))
    step_size = tf.sqrt(max_kl / (shs + 1e-8))

    old_params = flat_vars(policy.trainable_variables)
    assign_flat(policy.trainable_variables, old_params + step_size * step)

    with tf.GradientTape() as tape:
        v_loss = tf.reduce_mean((value_fn(states) - rets) ** 2)
    v_grads = tape.gradient(v_loss, value_fn.trainable_variables)
    value_opt.apply_gradients(zip(v_grads, value_fn.trainable_variables))

    if ep % 50 == 0:
        print(f"Episode {ep}, return = {sum(rewards):.3f}")

# ======================
# Visualization (FIXED G)
# ======================
arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
goal_x, goal_y = env.goal
grid = np.empty((5, 5), dtype=str)

plt.figure(figsize=(5, 5))

for i in range(5):
    for j in range(5):
        s = np.zeros(25, np.float32)
        s[i * 5 + j] = 1
        a = tf.argmax(policy(tf.expand_dims(s, 0)), axis=1).numpy()[0]
        grid[i, j] = arrows[a]
        plt.text(j, 4 - i, grid[i, j], ha='center', va='center', fontsize=20)

# draw goal G
plt.gca().add_patch(
    plt.Rectangle((goal_y - 0.5, 4 - goal_x - 0.5), 1, 1,
                  fill=True, color='lightgreen', alpha=0.6)
)
plt.text(goal_y, 4 - goal_x, 'G', ha='center', va='center',
         fontsize=22, fontweight='bold', color='green')

plt.xlim(-0.5, 4.5)
plt.ylim(-0.5, 4.5)
plt.grid(True)
plt.title("TRPO Policy on 5x5 GridWorld (Goal G)")
plt.show()
