# -*- coding:utf-8 -*-

import torch


class TreeNode:
    def __init__(self, feature_index=None, feature_value=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index 
        self.feature_value = feature_value
        self.left = left
        self.right = right
        self.value = value

class RegressionTree:
    def __init__(self, max_depth=5, min_samples_split=2, device='cpu'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.device = torch.device(device)
        self.root = None

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        # 停止条件
        if depth >= self.max_depth or n_samples < self.min_samples_split or torch.unique(y).shape[0] == 1:
            leaf_value = y.mean().item()
            return TreeNode(value=leaf_value)

        # 寻找最佳分割
        best_feature, best_value, best_loss, best_sets = None, None, float('inf'), None
        for feature_index in range(n_features):
            values = torch.unique(X[:, feature_index])
            for val in values:
                left_mask = X[:, feature_index] == val
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                loss = (y_left.var() * y_left.shape[0] + y_right.var() * y_right.shape[0]).item()

                if loss < best_loss:
                    best_feature = feature_index
                    best_value = val.item()
                    best_loss = loss
                    best_sets = (X[left_mask], y[left_mask], X[right_mask], y[right_mask])

        if best_sets is None:
            leaf_value = y.mean().item()
            return TreeNode(value=leaf_value)

        left = self._build_tree(best_sets[0], best_sets[1], depth + 1)
        right = self._build_tree(best_sets[2], best_sets[3], depth + 1)
        return TreeNode(feature_index=best_feature, feature_value=best_value, left=left, right=right)

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        return torch.tensor([self._predict_one(x, self.root) for x in X], device=self.device)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] == node.feature_value:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

if __name__ == "__main__":
    X = [
        [0, 1],
        [0, 0],
        [1, 1],
        [1, 0],
        [2, 1],
        [2, 0]
    ]
    y = [1.0, 1.2, 2.3, 2.1, 3.0, 3.1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tree = RegressionTree(max_depth=3, device=device)
    tree.fit(X, y)
    preds = tree.predict(X)
    print("预测:", preds.cpu().numpy())
