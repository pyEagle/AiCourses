import torch
import torch.nn as nn
import torch.optim as optim


class UnifiedConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(UnifiedConvNet, self).__init__()
        # 卷积层提取特征
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: (16, 14, 14)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出: (32, 7, 7)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def generate_fake_data(batch_size, device):
    images = torch.randn(batch_size, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    return images, labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UnifiedConvNet(num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n--- 开始训练演示 ---")
    for epoch in range(1, 6):
        model.train()
        
        images, labels = generate_fake_data(batch_size=32, device=device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/5] | Loss: {loss.item():.4f} | 设备: {images.device}")

    print("\n--- 任务完成 ---")

if __name__ == "__main__":
    main()

