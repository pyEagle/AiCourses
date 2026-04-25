# -*- coding:utf-8 -*-

import torch


def train_logistic_regression(X, y, lr=0.01, epochs=1000, device=device):
    X = X.to(device)
    y = y.to(device)

    n_features = X.shape[1]
    w = torch.randn(n_features, 1, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.randn(1, device=device, dtype=torch.float32, requires_grad=True)

    for epoch in range(epochs):
        z = X @ w + b
        y_pred = 1 / (1 + torch.exp(-z))

        loss = -(y * torch.log(y_pred + 1e-8) + (1 - y) * torch.log(1 - y_pred + 1e-8)).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return w, b

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    X = torch.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).float().reshape(-1, 1)

    w, b = train_logistic_regression(X, y, lr=0.1, epochs=1000)
    print("训练完成！")
    print("权重:", w)
    print("偏置:", b)
