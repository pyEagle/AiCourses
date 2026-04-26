# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- [运行设备] 当前使用: {device} ---\n")

def build_toy_graph():
    """
    构造一个简单的社交网络图：
    7个节点，前3个节点属一类，后4个节点属另一类。
    """
    # 节点特征: 7个节点，每个节点有3个特征 (例如: [活跃度, 兴趣A, 兴趣B])
    X = torch.tensor([
        [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], # 类 0
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0] # 类 1
    ], dtype=torch.float32)

    # 邻接矩阵 A (描述谁和谁相连)
    A = torch.tensor([
        [0, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 0],
    ], dtype=torch.float32)

    # 节点标签: 前三个是 0, 后四个是 1
    y = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    
    return X, A, y

def normalize_adjacency(A):
    """
    实现经典公式: A_hat = D^-0.5 * (A + I) * D^-0.5
    """
    I = torch.eye(A.size(0)).to(device)
    A_hat = A + I

    degree = torch.sum(A_hat, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))

    return D_inv_sqrt @ A_hat @ D_inv_sqrt

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        # 权重初始化
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X, A_hat):
        support = torch.mm(X, self.weight)
        output = torch.mm(A_hat, support)
        return output

class TorchLiteGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(TorchLiteGCN, self).__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)

    def forward(self, X, A_hat):
        h = F.relu(self.gcn1(X, A_hat))
        h = self.gcn2(h, A_hat)
        return h

def main():
    X, A, y = build_toy_graph()
    
    X, A, y = X.to(device), A.to(device), y.to(device)

    A_norm = normalize_adjacency(A)

    model = TorchLiteGCN(in_dim=3, hidden_dim=8, out_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("--- [开始训练] ---")
    for epoch in range(101):
        model.train()
        logits = model(X, A_norm)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

    model.eval()
    with torch.no_grad():
        final_logits = model(X, A_norm)
        final_pred = final_logits.argmax(dim=1)
        
    print("\n--- [预测结果] ---")
    print(f"预测标签: {final_pred.cpu().numpy()}")
    print(f"真实标签: {y.cpu().numpy()}")

if __name__ == "__main__":
    main()

