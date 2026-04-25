# -*- coding:utf-8 -*-

import torch

def train_linear_regression(X, y, lr=0.01, epochs=1000, device='cpu'):
    X = X.to(device)
    y = y.to(device)

    w = torch.randn(1, requires_grad=True, device=device)
    b = torch.randn(1, requires_grad=True, device=device)
    n = X.shape[0]

    for epoch in range(epochs):
        y_pred = X * w + b

        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}')

    return w, b


if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.linspace(0, 10, 100).unsqueeze(1)
    y = 2 * X + 3 + torch.randn_like(X) * 1.5
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    w_trained, b_trained = train_linear_regression(X, y, lr=0.01, epochs=1000, device=device)

    print(f'\n训练完成: w = {w_trained.item():.4f}, b = {b_trained.item():.4f}')
    print('真实值：w = 2, b =3')
