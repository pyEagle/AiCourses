# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

# ==========================================
# 1. 自动设备选择
# ==========================================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"[*] 当前设备: {DEVICE}")


# ==========================================
# 2. DeepFM 模型（稳定版）
# ==========================================
class TorchLiteDeepFM(nn.Module):
    def __init__(self, feature_sizes, embed_dim=8, hidden_dims=[32, 16]):
        super().__init__()

        self.num_fields = len(feature_sizes)

        # ========= FM 一阶 =========
        self.linear_emb = nn.ModuleList([
            nn.Embedding(feat_size, 1) for feat_size in feature_sizes
        ])

        # ========= FM 二阶 =========
        self.embed = nn.ModuleList([
            nn.Embedding(feat_size, embed_dim) for feat_size in feature_sizes
        ])

        # ========= Deep =========
        input_dim = self.num_fields * embed_dim
        layers = []

        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers)

        # ========= 初始化（关键）=========
        self._init_weights()

    def _init_weights(self):
        # linear 部分初始化为 0（常见做法）
        for emb in self.linear_emb:
            nn.init.zeros_(emb.weight)

        # embedding 用 Xavier
        for emb in self.embed:
            nn.init.xavier_uniform_(emb.weight)

        # deep 网络
        for m in self.deep:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # ========= FM 一阶 =========
        linear_terms = [
            emb(x[:, i]) for i, emb in enumerate(self.linear_emb)
        ]
        linear_part = torch.sum(torch.cat(linear_terms, dim=1), dim=1, keepdim=True)

        # ========= FM 二阶 =========
        embeddings = [
            emb(x[:, i]) for i, emb in enumerate(self.embed)
        ]

        stack_emb = torch.stack(embeddings, dim=1)  # [B, F, K]

        sum_square = torch.sum(stack_emb, dim=1) ** 2
        square_sum = torch.sum(stack_emb ** 2, dim=1)

        fm_second = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)

        # ========= Deep =========
        deep_input = torch.flatten(stack_emb, start_dim=1)
        deep_out = self.deep(deep_input)

        # ========= 输出（logits，不做 sigmoid）=========
        out = linear_part + fm_second + 0.1 * deep_out
        return out


# ==========================================
# 3. Toy 数据（可学习非线性）
# ==========================================
def generate_toy_data(num_samples=2000, num_fields=6, vocab_size=20):
    X = []
    y = []

    for _ in range(num_samples):
        sample = [random.randint(0, vocab_size - 1) for _ in range(num_fields)]

        # 非线性标签（DeepFM 能学到）
        score = sample[0] * sample[1] + sample[2] * sample[3]

        label = 1 if score % 2 == 0 else 0

        X.append(sample)
        y.append(label)

    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float32)


# ==========================================
# 4. 训练函数（mini-batch）
# ==========================================
def train():
    num_fields = 6
    vocab_size = 20
    batch_size = 64
    epochs = 15

    feature_sizes = [vocab_size] * num_fields

    model = TorchLiteDeepFM(feature_sizes).to(DEVICE)

    X, y = generate_toy_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()

    print("\n[*] 开始训练...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        total_samples = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = model(batch_x).squeeze()
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                prob = torch.sigmoid(logits)
                acc = ((prob > 0.5) == batch_y.bool()).float().sum()

            total_loss += loss.item() * batch_x.size(0)
            total_acc += acc.item()
            total_samples += batch_x.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

    print("\n[*] 训练完成")


# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    train()
