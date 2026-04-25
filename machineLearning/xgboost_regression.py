# -*- coding:utf-8 -*-

import torch
import random

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps") 
else:
    DEVICE = torch.device("cpu")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在运行设备: {DEVICE}")

class XGBNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class TorchRegressionTree:
    def __init__(self, max_depth=3, min_samples_split=2, reg_lambda=1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.root = None

    def fit(self, X, grad, hess):
        self.root = self._build_tree(X, grad, hess, depth=0)

    def _gain(self, G_l, H_l, G_r, H_r):
        def score(G, H):
            return (G ** 2) / (H + self.reg_lambda)
        
        return 0.5 * (score(G_l, H_l) + score(G_r, H_r) - score(G_l + G_r, H_l + H_r))

    def _leaf_value(self, G, H):
        return -G / (H + self.reg_lambda)

    def _build_tree(self, X, grad, hess, depth):
        num_samples, num_features = X.shape
        G_total = grad.sum()
        H_total = hess.sum()

        if depth >= self.max_depth or num_samples < self.min_samples_split:
            return XGBNode(value=self._leaf_value(G_total, H_total))

        best_gain = 0.0
        best_criteria = None
        best_sets = None

        for f_idx in range(num_features):
            thresholds = torch.unique(X[:, f_idx])
            for thresh in thresholds:
                mask_l = X[:, f_idx] <= thresh
                mask_r = ~mask_l

                if mask_l.sum() == 0 or mask_r.sum() == 0:
                    continue

                G_l, H_l = grad[mask_l].sum(), hess[mask_l].sum()
                G_r, H_r = grad[mask_r].sum(), hess[mask_r].sum()

                gain = self._gain(G_l, H_l, G_r, H_r)

                if gain > best_gain:
                    best_gain = gain
                    best_criteria = (f_idx, thresh)
                    best_sets = (mask_l, mask_r)

        if best_gain > 0:
            f_idx, thresh = best_criteria
            mask_l, mask_r = best_sets
            left = self._build_tree(X[mask_l], grad[mask_l], hess[mask_l], depth + 1)
            right = self._build_tree(X[mask_r], grad[mask_r], hess[mask_r], depth + 1)
            return XGBNode(feature_idx=f_idx, threshold=thresh, left=left, right=right)
        
        return XGBNode(value=self._leaf_value(G_total, H_total))

    def predict(self, X):
        results = torch.zeros(X.shape[0], device=DEVICE)
        for i, x in enumerate(X):
            node = self.root
            while not node.is_leaf():
                if x[node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            results[i] = node.value
        return results

class TorchLiteXGBRegressor:
    def __init__(self, n_estimators=5, lr=0.1, max_depth=3, reg_lambda=1.0):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.trees = []
        self.base_score = None

    def fit(self, X, y):
        self.base_score = y.mean()
        y_pred = torch.full_like(y, self.base_score)

        for i in range(self.n_estimators):
            grad = y_pred - y
            hess = torch.ones_like(y)

            tree = TorchRegressionTree(
                max_depth=self.max_depth, 
                reg_lambda=self.reg_lambda
            )
            tree.fit(X, grad, hess)
            
            update = tree.predict(X)
            y_pred += self.lr * update
            
            self.trees.append(tree)
            
            mse = torch.mean((y - y_pred)**2)
            print(f"迭代轮次 [{i+1}/{self.n_estimators}] - 训练 MSE: {mse.item():.4f}")

    def predict(self, X):
        y_pred = torch.full((X.shape[0],), self.base_score, device=DEVICE)
        for tree in self.trees:
            y_pred += self.lr * tree.predict(X)
        return y_pred

if __name__ == "__main__":
    torch.manual_seed(42)
    X = torch.randn(150, 4, device=DEVICE)
    y = X[:, 0]**2 + X[:, 1] * 2 + 0.5 * torch.randn(150, device=DEVICE)

    model = TorchLiteXGBRegressor(n_estimators=10, lr=0.3, max_depth=3)
    model.fit(X, y)

    with torch.no_grad():
        preds = model.predict(X)
        final_mse = torch.mean((preds - y)**2)
        print("-" * 30)
        print(f"✅ 训练完成! 最终 MSE: {final_mse.item():.4f}")

