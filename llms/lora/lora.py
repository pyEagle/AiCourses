# -*- coding:utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from weather import WeatherDataset
from model import WeatherLSTM
from globalSetting import device


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
 
        # 冻结原始权重
        for param in self.base_layer.parameters():
            param.requires_grad = False
 
        # 初始化 LoRA 矩阵 A (in_features × rank), B (rank × out_features)
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))  # 输入→低秩
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features)) # 低秩→输出
 
        self.reset_parameters()
 
    def reset_parameters(self):
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)
 
    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = x@self.lora_A @ self.lora_B * self.scaling
        return base_out + lora_out
 
print("1 加载预训练模型并冻结所有参数...")
lora_model = WeatherLSTM(input_dim=3, hidden_dim=64, num_layers=2, output_dim=1).to(device)
lora_model.load_state_dict(torch.load('weather_lstm_model.pt', map_location=device))
 
# 冻结整个模型
for param in lora_model.parameters():
    param.requires_grad = False
 
# 替换 fc 层为 LoRALinear
print("2 替换 fc 层为 LoRALinear（rank=4, alpha=1.0）...")
original_fc = lora_model.fc
lora_model.fc = LoRALinear(original_fc, rank=4, alpha=1.0, dropout=0.05).to(device)
print(f"  ✓ 原始 fc 权重形状: {original_fc.weight.shape}")
print(f"  ✓ LoRA A 形状: {lora_model.fc.lora_A.shape}, B 形状: {lora_model.fc.lora_B.shape}")
print(f"  ✓ 可训练参数量: {sum(p.numel() for p in lora_model.fc.parameters() if p.requires_grad):,} (仅 LoRA 部分)")
 
# 配置仅训练 LoRA 参数的优化器
print("3 配置优化器（仅优化 LoRA 参数）...")
lora_params = [p for p in lora_model.fc.parameters() if p.requires_grad]
optimizer_lora = optim.Adam(lora_params, lr=1e-3)
criterion_lora = nn.MSELoss()
 
# LoRA 微调训练
# 构建数据集和数据加载器
seq_len = 6  
batch_size = 32
train_dataset = WeatherDataset(seq_len=seq_len, num_samples=800)
test_dataset = WeatherDataset(seq_len=seq_len, num_samples=200)
 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("4 开始 LoRA 微调训练（仅更新 LoRA 适配器）...")
num_lora_epochs = 15
lora_train_losses = []
lora_test_losses = []

for epoch in range(num_lora_epochs):
    lora_model.train()
    epoch_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device).squeeze()
 
        optimizer_lora.zero_grad()
        outputs = lora_model(x).squeeze()
        loss = criterion_lora(outputs, y)
        loss.backward()
        optimizer_lora.step()
 
        epoch_loss += loss.item() * x.size(0)
 
    # 测试集评估
    lora_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device).squeeze()
            outputs = lora_model(x).squeeze()
            test_loss += criterion_lora(outputs, y).item() * x.size(0)
 
    avg_train = epoch_loss / len(train_loader.dataset)
    avg_test = test_loss / len(test_loader.dataset)
    lora_train_losses.append(avg_train)
    lora_test_losses.append(avg_test)
 
    if (epoch + 1) % 3 == 0:
        print(f"LoRA Epoch [{epoch+1}/{num_lora_epochs}] | Train Loss: {avg_train:.5f} | Test Loss: {avg_test:.5f}")
 
# 保存 LoRA 微调后模型
torch.save(lora_model.state_dict(), 'weather_lstm_lora_finetuned.pt')
print("5 LoRA 微调模型已保存: weather_lstm_lora_finetuned.pt")
 
# 对比原始模型与 LoRA 模型在测试数据上的预测
print("\n===== LoRA 模型预测效果对比 =====")
np.random.seed(123)
test_time = np.linspace(100, 110, seq_len)  # 新的时间区间
test_temp = 20 + 5 * np.sin(test_time) + np.random.randn(len(test_time)) * 0.5
test_humidity = 60 - 8 * np.sin(test_time) + np.random.randn(len(test_time)) * 0.8
test_pressure = 1013 + np.random.randn(len(test_time)) * 0.3
test_input = np.stack([test_temp, test_humidity, test_pressure], axis=1).astype(np.float32)
true_temp = 20 + 5 * np.sin(110.1) + np.random.randn() * 0.5
test_tensor = torch.from_numpy(test_input).unsqueeze(0).to(device)

lora_model.eval()
with torch.no_grad():
    lora_pred = lora_model(test_tensor).item()
 
print(f"LoRA 微调后预测: {lora_pred:.2f}℃")
print(f"真实温度: {true_temp:.2f}℃")
print(f"原始误差: LoRA 误差: {abs(lora_pred - true_temp):.4f}℃")
 
