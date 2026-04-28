# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import time

GRID_SIZE = 7
STATE_DIM = GRID_SIZE * GRID_SIZE
ACTION_DIM = 4

GAMMA = 0.99
LR = 1e-3
ENTROPY_BETA = 0.01
T_MAX = 10
MAX_EPISODES = 800
NUM_WORKERS = 4

DEVICE = torch.device("cpu")

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


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(STATE_DIM, 128)
        self.pi = nn.Linear(128, ACTION_DIM)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.pi(x), self.v(x)


def worker(wid, global_net, optimizer, global_ep, lock):
    local_net = ActorCritic()
    env = GridWorld(GRID_SIZE)

    while True:
        with lock:
            if global_ep.value >= MAX_EPISODES:
                break
            global_ep.value += 1
            ep_id = global_ep.value

        local_net.load_state_dict(global_net.state_dict())

        s = env.reset()
        states, actions, rewards = [], [], []
        ep_r = 0

        for _ in range(T_MAX):
            s_tensor = torch.FloatTensor(s).unsqueeze(0)
            logits, _ = local_net(s_tensor)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample().item()

            s_, r, done = env.step(a)

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
            s_tensor = torch.FloatTensor(s).unsqueeze(0)
            _, v = local_net(s_tensor)
            R = v.item()

        returns = []
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        s_batch = torch.FloatTensor(np.array(states))
        a_batch = torch.LongTensor(actions)
        r_batch = torch.FloatTensor(returns)

        logits, values = local_net(s_batch)
        values = values.squeeze(1)

        td = r_batch - values

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        log_p_a = log_probs.gather(1, a_batch.unsqueeze(1)).squeeze(1)

        policy_loss = -(log_p_a * td.detach()).mean()
        value_loss = 0.5 * td.pow(2).mean()
        entropy = -(probs * log_probs).sum(dim=1).mean()

        loss = policy_loss + value_loss - ENTROPY_BETA * entropy

        optimizer.zero_grad()
        loss.backward()

        for lp, gp in zip(local_net.parameters(), global_net.parameters()):
            gp._grad = lp.grad

        optimizer.step()

        if wid == 0 and ep_id % 10 == 0:
            print(f"[Episode {ep_id}] reward={ep_r:.2f}")


def train():
    mp.set_start_method('spawn', force=True)

    global_net = ActorCritic()
    global_net.share_memory()

    optimizer = torch.optim.Adam(global_net.parameters(), lr=LR)

    global_ep = mp.Value('i', 0)
    lock = mp.Lock()

    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker,
                       args=(i, global_net, optimizer, global_ep, lock))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    return global_net


def visualize_policy(net):
    env = GridWorld(GRID_SIZE)
    arrows = ['↑', '↓', '←', '→']

    print("\nLearned Policy:")
    for x in range(GRID_SIZE):
        row = []
        for y in range(GRID_SIZE):
            if (x, y) in env.walls:
                row.append(' # ')
            elif (x, y) in env.traps:
                row.append(' T ')
            elif (x, y) in env.goals:
                row.append(' G ')
            else:
                s = np.zeros(STATE_DIM, np.float32)
                s[x * GRID_SIZE + y] = 1.0
                s_tensor = torch.FloatTensor(s).unsqueeze(0)

                with torch.no_grad():
                    logits, _ = net(s_tensor)
                    a = torch.argmax(logits, dim=1).item()

                row.append(f" {arrows[a]} ")
        print(''.join(row))


if __name__ == "__main__":
    start = time.time()
    net = train()
    print("Training finished:", time.time() - start)

    visualize_policy(net)

