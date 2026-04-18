# -*- coding:utf-8 -*-

import random
import torch

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_iris()
X = torch.tensor(data.data, dtype=torch.float32, device=device)
y = torch.tensor(data.target, dtype=torch.long, device=device)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
    def __init__(self, max_depth=5, min_samples_split=2, feature_subsample=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subsample = feature_subsample
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(torch.unique(y))

        if depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))

        # 随机特征子集
        if self.feature_subsample is None:
            features = list(range(num_features))
        else:
            features = random.sample(range(num_features), self.feature_subsample)

        best_feat, best_thresh = self._best_split(X, y, features)
        if best_feat is None:
            return Node(value=self._most_common_label(y))

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(feature_index=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _best_split(self, X, y, features):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat in features:
            thresholds = torch.unique(X[:, feat])
            for thresh in thresholds:
                gain = self._information_gain(y, X[:, feat], thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = thresh
        return split_idx, split_thresh

    def _information_gain(self, y, feature_column, threshold):
        parent_entropy = self._entropy(y)
        left_idx = feature_column <= threshold
        right_idx = feature_column > threshold
        if left_idx.sum() == 0 or right_idx.sum() == 0:
            return 0
        n = len(y)
        n_l, n_r = left_idx.sum(), right_idx.sum()
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        return parent_entropy - (n_l / n) * e_l - (n_r / n) * e_r

    def _entropy(self, y):
        counts = torch.bincount(y)
        probs = counts.float() / len(y)
        probs = probs[probs > 0]
        return -torch.sum(probs * torch.log2(probs))

    def _most_common_label(self, y):
        counts = torch.bincount(y)
        return torch.argmax(counts).item()

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
# 随机森林
# ------------------------
class RandomForest:
    def __init__(self, n_trees=5, max_depth=5, min_samples_split=2, feature_subsample=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subsample = feature_subsample
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for i in range(self.n_trees):
            # bootstrap 样本
            idxs = torch.randint(0, n_samples, (n_samples,), device=device)
            X_sample, y_sample = X[idxs], y[idxs]

            tree = ID3DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_subsample=self.feature_subsample
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = torch.stack([tree.predict(X) for tree in self.trees], dim=1)
        y_pred = []
        for preds in tree_preds:
            vals, counts = torch.unique(preds, return_counts=True)
            y_pred.append(vals[torch.argmax(counts)].item())
        return torch.tensor(y_pred, device=device)

if __name__ == "__main__":
    rf = RandomForest(n_trees=5, max_depth=4, feature_subsample=2)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    accuracy = (y_pred == y_test).float().mean()
    print(f"Accuracy: {accuracy:.4f}")
    
    # 输出预测值和真实值
    print("预测: ", y_pred.tolist())
    print("真值: ", y_test.tolist())
