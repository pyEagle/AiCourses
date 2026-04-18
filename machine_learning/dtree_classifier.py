# -*- coding:utf-8 -*-

import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from graphviz import Digraph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_iris()
X = torch.tensor(data.data, dtype=torch.float32, device=device)
y = torch.tensor(data.target, dtype=torch.long, device=device)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# 决策树节点
# ------------------------
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # 叶子节点

    def is_leaf(self):
        return self.value is not None

# ------------------------
# ID3 决策树
# ------------------------
class ID3DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # 训练
    def fit(self, X, y):
        self.n_classes = len(torch.unique(y))
        self.root = self._build_tree(X, y)

    # 构建树
    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(torch.unique(y))

        if depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(feature_index=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    # 选择最佳分割特征
    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        num_features = X.shape[1]
        for feat in range(num_features):
            thresholds = torch.unique(X[:, feat])
            for thresh in thresholds:
                gain = self._information_gain(y, X[:, feat], thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = thresh
        return split_idx, split_thresh

    # 信息增益
    def _information_gain(self, y, feature_column, threshold):
        parent_entropy = self._entropy(y)
        left_idx = feature_column <= threshold
        right_idx = feature_column > threshold
        if left_idx.sum() == 0 or right_idx.sum() == 0:
            return 0
        n = len(y)
        n_l, n_r = left_idx.sum(), right_idx.sum()
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    # 熵计算
    def _entropy(self, y):
        counts = torch.bincount(y)
        probs = counts.float() / len(y)
        probs = probs[probs > 0]
        return -torch.sum(probs * torch.log2(probs))

    # 多数投票
    def _most_common_label(self, y):
        counts = torch.bincount(y)
        return torch.argmax(counts).item()

    # 单样本预测
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    # 批量预测
    def predict(self, X):
        return torch.tensor([self._traverse_tree(x, self.root) for x in X], device=device)

    # 单样本预测路径
    def predict_with_path(self, x):
        path = []
        node = self.root
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                path.append(f"特征 {node.feature_index} <= {node.threshold:.3f} -> Left")
                node = node.left
            else:
                path.append(f"特征 {node.feature_index} > {node.threshold:.3f} -> Right")
                node = node.right
        path.append(f"叶子: {node.value}")
        return node.value, path

    # ------------------------
    # 树图绘制
    # ------------------------
    def plot_tree(self, node=None, dot=None, node_id=None):
        if node is None:
            node = self.root
            dot = Digraph(comment='ID3 Decision Tree')
            node_id = '0'
            dot.node(node_id, self._node_label(node))

        # 左子树
        if node.left:
            left_id = node_id + 'L'
            dot.node(left_id, self._node_label(node.left))
            dot.edge(node_id, left_id, label='<=')  # 标记左分支
            self.plot_tree(node.left, dot, left_id)

        # 右子树
        if node.right:
            right_id = node_id + 'R'
            dot.node(right_id, self._node_label(node.right))
            dot.edge(node_id, right_id, label='>')
            self.plot_tree(node.right, dot, right_id)

        return dot

    def _node_label(self, node):
        if node.is_leaf():
            return f"Leaf\n{node.value}"
        else:
            return f"F{node.feature_index} <= {node.threshold:.3f}"

    def plot_path(self, x):
        dot = self.plot_tree()
        node_id = '0'
        path_nodes = [node_id]
        node = self.root
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node_id += 'L'
                node = node.left
            else:
                node_id += 'R'
                node = node.right
            path_nodes.append(node_id)

        # 高亮路径
        for i in range(len(path_nodes)-1):
            dot.edge(path_nodes[i], path_nodes[i+1], color='red', penwidth='3')
        return dot


if __name__ == "__main__":
    tree = ID3DecisionTree(max_depth=3)
    tree.fit(X_train, y_train)
    
    tree_dot = tree.plot_tree()
    tree_dot.render("id3_tree", format="png", view=True)  # 保存并打开
    
    sample_idx = 0
    test_sample = X_test[sample_idx]
    pred, path = tree.predict_with_path(test_sample)
    
    for step in path:
        print("  " + step)
    
    # 绘制高亮路径的树
    path_dot = tree.plot_path(test_sample)
    path_dot.render("id3_tree_path", format="png", view=True)
