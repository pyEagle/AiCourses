# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # 归一化到 [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Tanh() 
        )

    def forward(self, x):
        # x 的形状: [batch_size, 1, 28, 28] -> 展平为 [batch_size, 784]
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(x.size(0), 1, 28, 28)

model = Autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
print("开始训练...")

for epoch in range(num_epochs):
    total_loss = 0
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        
        output = model(img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

print("训练完成！")

model.eval()
with torch.no_grad():
    example_data, _ = next(iter(train_loader))
    example_data = example_data.to(device)
    reconstructed = model(example_data)

    plt.figure(figsize=(10, 4))
    for i in range(5):
        # 原始图像
        plt.subplot(2, 5, i + 1)
        plt.imshow(example_data[i].cpu().squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # 重构图像
        plt.subplot(2, 5, i + 6)
        plt.imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()
