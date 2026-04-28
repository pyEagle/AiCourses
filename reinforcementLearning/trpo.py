# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        if self.pos == [4, 4]:
            r = 1.0
            done = True

        return self._state(), r, done


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


policy = Policy().to(device)
value_fn = Value().to(device)
value_opt = optim.Adam(value_fn.parameters(), lr=1e-3)

def get_returns(rews, gamma=0.99):
    out, g = [], 0
    for r in reversed(rews):
        g = r + gamma * g
        out.insert(0, g)
    return torch.tensor(out, dtype=torch.float32, device=device)


def kl_divergence(old_logits, new_logits):
    p = torch.softmax(old_logits, dim=-1)
    log_p = torch.log_softmax(old_logits, dim=-1)
    log_q = torch.log_softmax(new_logits, dim=-1)
    return torch.mean(torch.sum(p * (log_p - log_q), dim=-1))


def fisher_vector_product(states, vec, damping=0.1):
    logits = policy(states)
    old_logits = logits.detach()

    kl = kl_divergence(old_logits, logits)
    grads = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    flat_grad_kl = parameters_to_vector(grads)

    g_v = (flat_grad_kl * vec).sum()

    grads2 = torch.autograd.grad(g_v, policy.parameters())
    flat_grads2 = parameters_to_vector(grads2)

    return flat_grads2 + damping * vec


def conjugate_gradient(fvp_fn, b, iters=10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(iters):
        Ap = fvp_fn(p)
        alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = torch.dot(r, r)
        p = r + (new_rdotr / (rdotr + 1e-8)) * p
        rdotr = new_rdotr
    return x


env = GridWorld5x5()
max_kl = 0.01

for ep in range(100):
    s = env.reset()
    states, actions, rewards = [], [], []
    done = False

    # rollout
    for _ in range(100):
        s_ts = torch.from_numpy(s).float().to(device)

        with torch.no_grad():
            logits = policy(s_ts.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        a = np.random.choice(4, p=probs)
        s2, r, done = env.step(a)

        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = s2
        if done:
            break

    states_ts = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions_ts = torch.tensor(actions, dtype=torch.long, device=device)
    returns_ts = get_returns(rewards)

    # advantage
    with torch.no_grad():
        values_ts = value_fn(states_ts)

    adv = returns_ts - values_ts
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # policy forward
    logits = policy(states_ts)
    old_logits = logits.detach()

    log_probs = torch.log_softmax(logits, dim=-1)
    old_log_probs = torch.log_softmax(old_logits, dim=-1)

    log_p_a = log_probs.gather(1, actions_ts.unsqueeze(1)).squeeze()
    old_log_p_a = old_log_probs.gather(1, actions_ts.unsqueeze(1)).squeeze()

    ratio = torch.exp(log_p_a - old_log_p_a)
    loss = torch.mean(ratio * adv)

    grads = torch.autograd.grad(loss, policy.parameters())
    g = parameters_to_vector(grads)

    # CG
    fvp_fn = lambda v: fisher_vector_product(states_ts, v)
    step_dir = conjugate_gradient(fvp_fn, g)

    # step size
    shs = 0.5 * torch.dot(step_dir, fvp_fn(step_dir))
    step_size = torch.sqrt(max_kl / (shs + 1e-8))

    full_step = step_size * step_dir

    old_params = parameters_to_vector(policy.parameters())

    def surrogate_loss():
        logits_new = policy(states_ts)
        log_probs_new = torch.log_softmax(logits_new, dim=-1)
        log_p_a_new = log_probs_new.gather(1, actions_ts.unsqueeze(1)).squeeze()
        return torch.mean(torch.exp(log_p_a_new - old_log_p_a) * adv)

    old_loss = surrogate_loss().item()

    success = False
    for frac in [1.0, 0.5, 0.25, 0.125, 0.0625]:
        new_params = old_params + frac * full_step
        vector_to_parameters(new_params, policy.parameters())

        new_loss = surrogate_loss().item()
        kl = kl_divergence(old_logits, policy(states_ts)).item()

        if kl < max_kl and new_loss > old_loss:
            success = True
            break

    if not success:
        vector_to_parameters(old_params, policy.parameters())

    # value update
    value_loss = nn.MSELoss()(value_fn(states_ts), returns_ts)
    value_opt.zero_grad()
    value_loss.backward()
    value_opt.step()

    if ep % 10 == 0:
        print(f"Episode {ep}, return = {sum(rewards):.3f}")


arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}

plt.figure(figsize=(5, 5))
policy.eval()

for i in range(5):
    for j in range(5):
        s = np.zeros(25, np.float32)
        s[i * 5 + j] = 1
        s_ts = torch.from_numpy(s).float().to(device).unsqueeze(0)

        with torch.no_grad():
            a = torch.argmax(policy(s_ts), dim=1).item()

        plt.text(j, 4 - i, arrows[a], ha='center', va='center', fontsize=20)

plt.gca().add_patch(
    plt.Rectangle((4 - 0.5, 4 - 4 - 0.5), 1, 1,
                  fill=True, color='lightgreen', alpha=0.6)
)
plt.text(4, 0, 'G', ha='center', va='center',
         fontsize=22, fontweight='bold', color='green')

plt.xlim(-0.5, 4.5)
plt.ylim(-0.5, 4.5)
plt.grid(True)
plt.title("TRPO Policy (Fixed & Full Version)")
plt.show()
