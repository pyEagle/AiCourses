import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 5
T = 1000
LR = 1e-3

print(f"[Device] {DEVICE}")

class LightDiffuser:
    def __init__(self, T, device):
        self.T = T
        self.device = device

        betas = torch.linspace(1e-4, 0.02, T)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)

        sqrt_acp = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_1_acp = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)

        xt = sqrt_acp * x0 + sqrt_1_acp * noise
        return xt, noise

class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, t):
        # t embedding
        t = t.float().view(-1, 1) / 1000.0
        t_emb = self.time_mlp(t).view(-1, 64, 1, 1)

        h1 = F.relu(self.conv1(x) + t_emb)
        h2 = self.pool(h1)

        h3 = F.relu(self.conv2(h2))

        h4 = self.up(h3)
        h5 = F.relu(self.conv3(h4))

        return self.conv4(h5)

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = TinyUNet().to(DEVICE)
    diffuser = LightDiffuser(T, DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    print("Training...")

    for epoch in range(EPOCHS):
        total_loss = 0

        for x, _ in loader:
            x = x.to(DEVICE)
            t = torch.randint(0, T, (x.size(0),), device=DEVICE)
            xt, noise = diffuser.add_noise(x, t)

            pred = model(xt, t)

            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

    print("finish...")

    model.eval()

    with torch.no_grad():
        x = torch.randn(16, 1, 28, 28).to(DEVICE)
        for t in reversed(range(T)):
            t_tensor = torch.full((16,), t, device=DEVICE, dtype=torch.long)
            pred_noise = model(x, t_tensor)
            beta = diffuser.betas[t]
            alpha = diffuser.alphas[t]
            acp = diffuser.alphas_cumprod[t]
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            mean = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - acp)) * pred_noise
            )
            x = mean + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(x[i].squeeze().cpu(), cmap="gray")
            ax.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

