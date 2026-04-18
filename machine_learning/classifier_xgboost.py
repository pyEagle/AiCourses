# -*- coding:utf-8 -*-

import torch

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_iris()
X = torch.tensor(data.data, dtype=torch.float32, device=device)
y = torch.tensor((data.target == 0).astype(float), dtype=torch.float32, device=device)  # 二分类
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# ID3 决策树节点
# ------------------------
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

# ------------------------
# ID3 决策树
# ------------------------
class ID3DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2, feature_subsample=None, reg_lambda=1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subsample = feature_subsample
        self.reg_lambda = reg_lambda
        self.root = None

    def fit(self, X, grad, hess):
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, grad, hess)

    def _build_tree(self, X, grad, hess, depth=0):
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples < self.min_samples_split:
            return Node(value=self._leaf_value(grad, hess))

        if self.feature_subsample is None:
            features = list(range(num_features))
        else:
            import random
            features = random.sample(range(num_features), self.feature_subsample)

        best_gain = -float('inf')
        split_idx, split_thresh = None, None
        left_idx_best, right_idx_best = None, None

        for feat in features:
            thresholds = torch.unique(X[:, feat])
            for thresh in thresholds:
                left_idx = X[:, feat] <= thresh
                right_idx = X[:, feat] > thresh
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                gain = self._calc_gain(grad[left_idx], hess[left_idx], grad[right_idx], hess[right_idx])
                if gain > best_gain:
                    best_gain = gain
                    split_idx, split_thresh = feat, thresh
                    left_idx_best, right_idx_best = left_idx, right_idx

        if best_gain == -float('inf'):
            return Node(value=self._leaf_value(grad, hess))

        left_child = self._build_tree(X[left_idx_best], grad[left_idx_best], hess[left_idx_best], depth + 1)
        right_child = self._build_tree(X[right_idx_best], grad[right_idx_best], hess[right_idx_best], depth + 1)
        return Node(feature_index=split_idx, threshold=split_thresh, left=left_child, right=right_child)

    def _calc_gain(self, grad_left, hess_left, grad_right, hess_right):
        # XGBoost 分裂增益公式
        G_left, H_left = grad_left.sum(), hess_left.sum()
        G_right, H_right = grad_right.sum(), hess_right.sum()
        G_total, H_total = G_left + G_right, H_left + H_right
        gain = 0.5 * (G_left**2 / (H_left + self.reg_lambda) + G_right**2 / (H_right + self.reg_lambda) - G_total**2 / (H_total + self.reg_lambda))
        return gain

    def _leaf_value(self, grad, hess):
        return -(grad.sum() / (hess.sum() + self.reg_lambda)).item()

    def predict_sample(self, x):
        node = self.root
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        return torch.tensor([self.predict_sample(x) for x in X], device=device)

# ------------------------
# XGBoost模型
# ------------------------
class XGBoost:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=3, feature_subsample=None, reg_lambda=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample = feature_subsample
        self.reg_lambda = reg_lambda
        self.trees = []
        self.base_score = 0.5  # 二分类默认概率

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def fit(self, X, y):
        y_pred = torch.full((X.shape[0],), self.base_score, device=device)
        self.trees = []

        for _ in range(self.n_estimators):
            # 梯度和二阶导
            pred_prob = self._sigmoid(y_pred)
            grad = pred_prob - y  # gradient of logloss
            hess = pred_prob * (1 - pred_prob)  # hessian

            tree = ID3DecisionTree(max_depth=self.max_depth, feature_subsample=self.feature_subsample, reg_lambda=self.reg_lambda)
            tree.fit(X, grad, hess)

            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)

    def predict_raw(self, X):
        y_pred = torch.full((X.shape[0],), self.base_score, device=device)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

    def predict(self, X):
        y_raw = self.predict_raw(X)
        return (self._sigmoid(y_raw) > 0.5).float()

# ------------------------
# 测试 XGBoost
# ------------------------
xgb = XGBoost(n_estimators=5, learning_rate=0.1, max_depth=3, feature_subsample=2)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
accuracy = (y_pred == y_test).float().mean()
print(f"Accuracy: {accuracy:.4f}")
print("预测: ", y_pred.tolist())
print("真值: ", y_test.tolist())
