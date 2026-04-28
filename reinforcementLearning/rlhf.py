# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = ['a', 'b', 'c', 'd', 'e', ' ']
vocab_size = len(vocab)
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for i, c in enumerate(vocab)}

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, dff=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.logits = nn.Linear(d_model, vocab_size)
        self.value = nn.Linear(d_model, 1)

    def forward(self, x):
        emb = self.embedding(x)
        attn_out, _ = self.attn(emb, emb, emb)

        x = self.norm1(emb + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        logits = self.logits(x)
        values = self.value(x).squeeze(-1)
        return logits, values

class RewardModel(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        emb = self.embedding(x)       # [B, L, D]
        pooled = emb.mean(dim=1)      # 简化：平均池化
        return self.fc(pooled).squeeze(-1)

def true_reward(seq):
    # 模拟人类偏好：喜欢多样性 + 少重复 + 少空格
    unique = len(set(seq))
    penalty = seq.count(' ') * 0.5
    repeat_penalty = max(seq.count(c) for c in set(seq))
    return unique - penalty - 0.3 * repeat_penalty

def generate_preference_data(n=2000, seq_len=8):
    data = []
    for _ in range(n):
        s1 = [random.randint(0, vocab_size-1) for _ in range(seq_len)]
        s2 = [random.randint(0, vocab_size-1) for _ in range(seq_len)]

        r1 = true_reward(s1)
        r2 = true_reward(s2)

        if r1 > r2:
            data.append((s1, s2))
        else:
            data.append((s2, s1))
    return data

def train_reward_model():
    rm = RewardModel(vocab_size).to(device)
    opt = optim.Adam(rm.parameters(), lr=1e-3)

    data = generate_preference_data()

    for epoch in range(5):
        total_loss = 0
        for s1, s2 in data:
            s1 = torch.tensor([s1], dtype=torch.long).to(device)
            s2 = torch.tensor([s2], dtype=torch.long).to(device)

            r1 = rm(s1)
            r2 = rm(s2)

            loss = -torch.log(torch.sigmoid(r1 - r2)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"[RM] epoch {epoch} loss {total_loss:.4f}")

    return rm

def sample_sequence(model, seq_len=8):
    model.eval()
    x = torch.zeros((1,1), dtype=torch.long).to(device)

    seq, logps, values = [], [], []

    with torch.no_grad():
        for _ in range(seq_len):
            logits, val = model(x)
            probs = torch.softmax(logits[:, -1], dim=-1)

            dist = torch.distributions.Categorical(probs)
            a = dist.sample()

            seq.append(a.item())
            logps.append(dist.log_prob(a).item())
            values.append(val[0, -1].item())

            x = torch.cat([x, a.view(1,1)], dim=1)

    return seq, logps, values

def compute_adv(rewards, values, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rewards)
    last = 0
    for t in reversed(range(len(rewards))):
        next_v = values[t+1] if t+1 < len(values) else 0
        delta = rewards[t] + gamma * next_v - values[t]
        adv[t] = last = delta + gamma * lam * last
    return adv

policy = TransformerLM(vocab_size).to(device)
ref_policy = copy.deepcopy(policy).eval()  # 冻结参考模型

optimizer = optim.Adam(policy.parameters(), lr=1e-3)

print("训练 Reward Model...")
reward_model = train_reward_model()

clip_ratio = 0.2
beta = 0.1   # KL 权重
epochs = 200
batch_size = 16
seq_len = 8

for epoch in range(epochs):

    batch_seq, batch_logp, batch_val = [], [], []
    batch_rewards = []

    for _ in range(batch_size):
        seq, logp, val = sample_sequence(policy, seq_len)

        seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
        with torch.no_grad():
            r = reward_model(seq_tensor).item()

        batch_seq.append(seq)
        batch_logp.append(logp)
        batch_val.append(val)
        batch_rewards.append([r]*seq_len)

    seqs = torch.tensor(batch_seq, dtype=torch.long).to(device)
    old_logp = torch.tensor(batch_logp, dtype=torch.float32).to(device)
    values = torch.tensor(batch_val, dtype=torch.float32).to(device)
    rewards = np.array(batch_rewards)

    advs = np.array([compute_adv(rewards[i], values[i].cpu().numpy()) for i in range(batch_size)])
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    advs = torch.tensor(advs, dtype=torch.float32).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

    logits, vals = policy(seqs)
    logp_all = torch.log_softmax(logits, dim=-1)
    curr_logp = torch.gather(logp_all, 2, seqs.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        ref_logits, _ = ref_policy(seqs)
        ref_logp_all = torch.log_softmax(ref_logits, dim=-1)
        ref_logp = torch.gather(ref_logp_all, 2, seqs.unsqueeze(-1)).squeeze(-1)

    kl = (curr_logp - ref_logp)

    ratio = torch.exp(curr_logp - old_logp)

    surr1 = ratio * advs
    surr2 = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advs

    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = nn.MSELoss()(vals, rewards)

    kl_loss = kl.mean()

    total_loss = policy_loss + 0.5*value_loss + beta * kl_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        seq, _, _ = sample_sequence(policy, seq_len)
        text = ''.join([idx2char[i] for i in seq])
        print(f"Epoch {epoch} | Sample: {text} | Loss: {total_loss.item():.4f}")

print("训练完成")

