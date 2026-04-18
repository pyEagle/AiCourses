# -*- coding:utf-8 -*-

import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_iris()
X = torch.tensor(data.data, dtype=torch.float32, device=device)
y = torch.tensor((data.target == 0).astype(int), dtype=torch.float32, device=device)  # 二分类回归形式
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
    def __init__(self, max_depth=3, min_samples_split=2, feature_subsample=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subsample = feature_subsample
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples < self.min_samples_split or len(torch.unique(y)) <= 1:
            return Node(value=y.mean().item())  # 回归：叶子存均值

        # 随机特征子集
        if self.feature_subsample is None:
            features = list(range(num_features))
        else:
            import random
            features = random.sample(range(num_features), self.feature_subsample)

        best_feat, best_thresh = self._best_split(X, y, features)
        if best_feat is None:
            return Node(value=y.mean().item())

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(feature_index=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _best_split(self, X, y, features):
        best_loss = float('inf')
        split_idx, split_thresh = None, None
        for feat in features:
            thresholds = torch.unique(X[:, feat])
            for thresh in thresholds:
                left_idx = X[:, feat] <= thresh
                right_idx = X[:, feat] > thresh
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue
                loss = ((y[left_idx] - y[left_idx].mean())**2).sum() + ((y[right_idx] - y[right_idx].mean())**2).sum()
                if loss < best_loss:
                    best_loss = loss
                    split_idx = feat
                    split_thresh = thresh
        return split_idx, split_thresh

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
# GBDT
# ------------------------
class GBDT:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=3, feature_subsample=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample = feature_subsample
        self.trees = []
        self.init_value = None

    def fit(self, X, y):
        # 初始化预测为均值
        self.init_value = y.mean().item()
        y_pred = torch.full_like(y, self.init_value)

        self.trees = []
        for _ in range(self.n_estimators):
            residual = y - y_pred  # 负梯度
            tree = ID3DecisionTree(max_depth=self.max_depth, feature_subsample=self.feature_subsample)
            tree.fit(X, residual)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        y_pred = torch.full((X.shape[0],), self.init_value, device=device)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

if __name__ == "__main__":
    gbdt = GBDT(n_estimators=5, learning_rate=0.1, max_depth=3, feature_subsample=2)
    gbdt.fit(X_train, y_train)
    
    y_pred = gbdt.predict(X_test)
    
    # 二分类问题用 0.5 阈值
    y_pred_class = (y_pred > 0.5).float()
    accuracy = (y_pred_class == y_test).float().mean()
    print(f"Accuracy: {accuracy:.4f}")
    print("预测: ", y_pred_class.tolist())
    print("真值: ", y_test.tolist())
