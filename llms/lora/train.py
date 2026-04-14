# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from weather import WeatherDataset
from model import WeatherLSTM
from globalSetting import device

from torch.utils.data import Dataset, DataLoader

# 1. 构建数据集和数据加载器
seq_len = 6  
batch_size = 32
train_dataset = WeatherDataset(seq_len=seq_len, num_samples=800)
test_dataset = WeatherDataset(seq_len=seq_len, num_samples=200)
 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. 初始化模型并训练
model = WeatherLSTM(input_dim=3, hidden_dim=64, num_layers=2, output_dim=1).to(device)
criterion = nn.MSELoss()  # 回归任务用均方误差
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
train_loss_list = []
test_loss_list = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device).squeeze()

        # 前向传播
        outputs = model(x).squeeze()
        loss = criterion(outputs, y)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)

    # 验证阶段
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device).squeeze()
            outputs = model(x).squeeze()
            loss = criterion(outputs, y)
            test_loss += loss.item() * x.size(0)

    # 计算平均损失
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_test_loss = test_loss / len(test_loader.dataset)
    train_loss_list.append(avg_train_loss)
    test_loss_list.append(avg_test_loss)

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

# 3. 保存模型参数
torch.save(model.state_dict(), 'weather_lstm_model.pt')

# 4. 测试效果
np.random.seed(123)  # 固定测试数据种子
test_time = np.linspace(100, 110, seq_len)  # 新的时间区间
test_temp = 20 + 5 * np.sin(test_time) + np.random.randn(len(test_time)) * 0.5
test_humidity = 60 - 8 * np.sin(test_time) + np.random.randn(len(test_time)) * 0.8
test_pressure = 1013 + np.random.randn(len(test_time)) * 0.3
test_input = np.stack([test_temp, test_humidity, test_pressure], axis=1).astype(np.float32)
true_temp = 20 + 5 * np.sin(110.1) + np.random.randn() * 0.5

loaded_model = WeatherLSTM().to(device)
loaded_model.load_state_dict(torch.load('weather_lstm_model.pt', map_location=device))
loaded_model.eval()  

test_tensor = torch.from_numpy(test_input).unsqueeze(0).to(device)
with torch.no_grad():
    pred_temp = loaded_model(test_tensor).item()  # 预测温度

print("\n===== 测试数据与预测结果 =====")
print(f"测试输入序列（温度/湿度/气压）：")
for i in range(seq_len):
    print(f"  时刻{i+1}: 温度={test_temp[i]:.2f}℃, 湿度={test_humidity[i]:.2f}%, 气压={test_pressure[i]:.2f}hPa")
print(f"\n真实下一刻温度：{true_temp:.2f}℃")
print(f"模型预测温度：{pred_temp:.2f}℃")
print(f"预测误差：{abs(pred_temp - true_temp):.4f}℃")
