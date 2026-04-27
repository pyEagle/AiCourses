# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf

# -----------------------------
# 1. Token表和数据
# -----------------------------
vocab = ['a','b','c','d','e',' ']
vocab_size = len(vocab)
char2idx = {c:i for i,c in enumerate(vocab)}
idx2char = {i:c for i,c in enumerate(vocab)}

def generate_sequences(num_sequences=1000, seq_len=8):
    data = []
    for _ in range(num_sequences):
        seq = np.random.choice(vocab, size=seq_len)
        data.append(''.join(seq))
    return data

train_texts = generate_sequences()

# -----------------------------
# 2. Transformer LM + Value
# -----------------------------
class TransformerLM(tf.keras.Model):
    def __init__(self, vocab_size, d_model=64, num_heads=4, dff=128):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.logits_layer = tf.keras.layers.Dense(vocab_size)
        self.value_layer = tf.keras.layers.Dense(1)

    def call(self, x):
        x_emb = self.embedding(x)
        attn_out = self.attention(x_emb, x_emb)
        x1 = self.norm1(x_emb + attn_out)
        ffn_out = self.ffn(x1)
        x2 = self.norm2(x1 + ffn_out)
        logits = self.logits_layer(x2)
        values = tf.squeeze(self.value_layer(x2), axis=-1)
        return logits, values

model = TransformerLM(vocab_size)
optimizer = tf.keras.optimizers.Adam(1e-3)

# -----------------------------
# 3. 奖励模型
# -----------------------------
def reward_model(seq_text):
    return seq_text.count('a')  # 奖励 'a' 越多

# -----------------------------
# 4. PPO 工具
# -----------------------------
def sample_sequence(model, seq_len=8):
    seq_idx = []
    seq_char = []
    x = tf.zeros((1,1), dtype=tf.int32)
    old_log_probs = []
    values_list = []

    for _ in range(seq_len):
        logits, values = model(x)
        probs = tf.nn.softmax(logits[:,-1,:])
        idx = np.random.choice(vocab_size, p=probs.numpy()[0])
        seq_idx.append(idx)
        seq_char.append(idx2char[idx])
        old_log_probs.append(tf.math.log(probs[0, idx] + 1e-8))
        values_list.append(values[0, -1].numpy())
        x = tf.concat([x, tf.constant([[idx]])], axis=1)

    seq_text = ''.join(seq_char)
    reward = reward_model(seq_text)
    return seq_idx, seq_char, old_log_probs, values_list, reward

def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advs = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t+1] if t+1 < len(values) else 0
        delta = rewards[t] + gamma * next_value - values[t]
        advs[t] = lastgaelam = delta + gamma * lam * lastgaelam
    return advs

def log_prob_from_logits(logits, actions):
    one_hot = tf.one_hot(actions, vocab_size)
    logp_all = tf.nn.log_softmax(logits)
    logp = tf.reduce_sum(logp_all * one_hot, axis=-1)
    return logp

# -----------------------------
# 5. PPO 更新循环
# -----------------------------
epochs = 200
batch_size = 16
seq_len = 8
clip_ratio = 0.2
gamma = 0.99
lam = 0.95

for epoch in range(epochs):
    batch_seqs_idx = []
    batch_seqs_char = []
    batch_old_log_probs = []
    batch_values = []
    batch_rewards = []

    # 收集一批序列
    for _ in range(batch_size):
        seq_idx, seq_char, old_log_probs, values, reward = sample_sequence(model, seq_len)
        batch_seqs_idx.append(seq_idx)
        batch_seqs_char.append(seq_char)
        batch_old_log_probs.append(old_log_probs)
        batch_values.append(values)
        batch_rewards.append([reward]*seq_len)

    batch_seqs_idx = np.array(batch_seqs_idx, dtype=np.int32)
    batch_old_log_probs = np.array(batch_old_log_probs, dtype=np.float32)
    batch_values = np.array(batch_values, dtype=np.float32)
    batch_rewards = np.array(batch_rewards, dtype=np.float32)

    # 计算优势
    batch_advs = np.zeros_like(batch_rewards)
    for i in range(batch_size):
        batch_advs[i] = compute_advantages(batch_rewards[i], batch_values[i], gamma, lam)
    batch_advs = (batch_advs - batch_advs.mean()) / (batch_advs.std() + 1e-8)

    # PPO 更新
    with tf.GradientTape() as tape:
        policy_losses = []
        value_losses = []
        for i in range(batch_size):
            x = tf.constant([batch_seqs_idx[i][:-1]], dtype=np.int32)
            actions = batch_seqs_idx[i][1:]
            logits, values = model(x)
            new_log_probs = log_prob_from_logits(logits, actions)
            old_log_probs_seq = tf.constant(batch_old_log_probs[i][1:], dtype=tf.float32)
            ratio = tf.exp(new_log_probs - old_log_probs_seq)
            clip_adv = tf.clip_by_value(ratio, 1-clip_ratio, 1+clip_ratio) * batch_advs[i][1:]
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * batch_advs[i][1:], clip_adv))
            value_loss = tf.reduce_mean((values[0] - batch_rewards[i][1:])**2)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
        total_loss = tf.reduce_mean(policy_losses) + 0.5*tf.reduce_mean(value_losses)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # -----------------------------
    # 输出测试结果
    # -----------------------------
    test_idx, test_char, _, _, reward_test = sample_sequence(model, seq_len)
    test_text = ''.join(test_char)
    if epoch%50 ==0:
        print(f"Epoch {epoch} | Sample Seq Text: {test_text} | Reward: {reward_test} | Action Chars: {test_char}")

print("训练完成")
