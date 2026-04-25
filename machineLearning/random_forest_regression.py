# -*- coding:utf-8 -*-

import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 当前设备: {device}")

def make_regression(n_samples=500):
    X = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    y = torch.sin(X) + 0.1 * torch.randn_like(X)

    return X.to(device), y.squeeze().to(device)

X, y = make_regression()

perm = torch.randperm(len(X))
train_size = int(0.8 * len(X))
train_idx = perm[:train_size]
test_idx = perm[train_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

class Node:
    def __init__(self, feature_index=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class RegressionTree:
    def __init__(self, max_depth=5, min_samples_split=5, feature_subsample=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subsample = feature_subsample
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        if (depth >= self.max_depth or
            n_samples < self.min_samples_split):
            return Node(value=self._leaf_value(y))

        if self.feature_subsample is None:
            features = list(range(n_features))
        else:
            features = random.sample(range(n_features), self.feature_subsample)

        best_feat, best_thresh = self._best_split(X, y, features)

        if best_feat is None:
            return Node(value=self._leaf_value(y))

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = X[:, best_feat] > best_thresh

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(best_feat, best_thresh, left_child, right_child)

    def _best_split(self, X, y, features):
        best_mse = float("inf")
        split_idx, split_thresh = None, None

        for feat in features:
            thresholds = torch.unique(X[:, feat])

            for thresh in thresholds:
                left = y[X[:, feat] <= thresh]
                right = y[X[:, feat] > thresh]

                if len(left) == 0 or len(right) == 0:
                    continue

                mse = self._mse(left, right)

                if mse < best_mse:
                    best_mse = mse
                    split_idx = feat
                    split_thresh = thresh

        return split_idx, split_thresh

    def _mse(self, left, right):
        def var(y):
            return torch.var(y, unbiased=False) if len(y) > 0 else 0.0

        n = len(left) + len(right)
        return (len(left)/n) * var(left) + (len(right)/n) * var(right)

    def _leaf_value(self, y):
        return torch.mean(y).item()

    def predict_sample(self, x):
        node = self.root
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        return torch.tensor(
            [self.predict_sample(x) for x in X],
            device=device
        )


class TorchForestRegressorEdu:
    def __init__(self, n_trees=10, max_depth=5,
                 min_samples_split=5, feature_subsample=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subsample = feature_subsample
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]

        for i in range(self.n_trees):
            # bootstrap
            idxs = torch.randint(0, n_samples, (n_samples,), device=device)
            X_sample = X[idxs]
            y_sample = y[idxs]

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_subsample=self.feature_subsample
            )

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        preds = torch.stack(
            [tree.predict(X) for tree in self.trees],
            dim=1
        )
        return torch.mean(preds, dim=1)


if __name__ == "__main__":
    model = TorchForestRegressorEdu(
        n_trees=10,
        max_depth=6,
        feature_subsample=1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = torch.mean((y_pred - y_test) ** 2)

    print(f"\n[RESULT]")
    print(f"MSE: {mse:.6f}")

    print("\n预测前10个:")
    print(y_pred[:10])

    print("\n真实前10个:")
    print(y_test[:10])

