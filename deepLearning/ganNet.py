import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 0.001
EPOCHS = 3000
BATCH_SIZE = 128
NOISE_DIM = 5
DATA_DIM = 2

def get_real_data(batch_size):
    x = torch.rand(batch_size, 1) * 2 - 1 
    y = x ** 2 + 0.1 * torch.randn(batch_size, 1)
    return torch.cat((x, y), dim=1).to(device)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, DATA_DIM)
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(DATA_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

opt_G = optim.Adam(generator.parameters(), lr=LR)
opt_D = optim.Adam(discriminator.parameters(), lr=LR)

criterion = nn.BCELoss()

print("开始对抗训练...")
for epoch in range(EPOCHS):
    real_data = get_real_data(BATCH_SIZE)
    real_labels = torch.ones((BATCH_SIZE, 1), device=device)
    
    noise = torch.randn((BATCH_SIZE, NOISE_DIM), device=device)
    fake_data = generator(noise)
    fake_labels = torch.zeros((BATCH_SIZE, 1), device=device)
    
    opt_D.zero_grad()
    loss_D_real = criterion(discriminator(real_data), real_labels) 
    loss_D_fake = criterion(discriminator(fake_data.detach()), fake_labels) 
    
    loss_D = loss_D_real + loss_D_fake
    loss_D.backward()
    opt_D.step()

    opt_G.zero_grad()
    predictions_on_fake = discriminator(fake_data)
    loss_G = criterion(predictions_on_fake, real_labels) 
    
    loss_G.backward()
    opt_G.step()

    if epoch % 500 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch [{epoch}/{EPOCHS}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

print("训练完成，正在生成可视化结果...")
generator.eval()

with torch.no_grad():
    test_noise = torch.randn((500, NOISE_DIM), device=device)
    generated_points = generator(test_noise).cpu().numpy()
    
real_points = get_real_data(500).cpu().numpy()

plt.figure(figsize=(8, 6))
plt.scatter(real_points[:, 0], real_points[:, 1], color='blue', alpha=0.5, label='Real Data (y = x^2)')
plt.scatter(generated_points[:, 0], generated_points[:, 1], color='red', alpha=0.5, label='Generated Data')
plt.title('EchoGAN-Primer: Real vs Generated Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

