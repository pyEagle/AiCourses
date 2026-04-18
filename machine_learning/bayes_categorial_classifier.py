# -*- coding:utf-8 -*-

import torch


class DiscreteNaiveBayes:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = None
        self.class_counts = None
        self.feature_counts = None
        self.feature_log_probs = None
        self.class_log_prior = None
        self.n_features = None
        self.n_classes = None

    def fit(self, X, y):
        X = X.to(self.device)
        y = y.to(self.device)

        self.classes = torch.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]

        self.class_counts = torch.zeros(self.n_classes, device=self.device)
        self.feature_counts = []

        # 初始化每个特征的类别计数列表
        for j in range(self.n_features):
            n_values = int(torch.max(X[:, j]).item() + 1)  # 每个特征的取值数
            self.feature_counts.append(torch.zeros((self.n_classes, n_values), device=self.device))

        # 计算先验和条件概率
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_counts[idx] = X_c.shape[0]
            for j in range(self.n_features):
                for val in range(self.feature_counts[j].shape[1]):
                    self.feature_counts[j][idx, val] = (X_c[:, j] == val).sum()

        # 加1平滑
        self.feature_log_probs = []
        for j in range(self.n_features):
            smoothed = self.feature_counts[j] + 1
            smoothed /= smoothed.sum(dim=1, keepdim=True)
            self.feature_log_probs.append(torch.log(smoothed))

        # 先验概率
        self.class_log_prior = torch.log(self.class_counts / self.class_counts.sum())

    def predict(self, X):
        X = X.to(self.device)
        n_samples = X.shape[0]
        log_probs = torch.zeros((n_samples, self.n_classes), device=self.device)

        for idx, c in enumerate(self.classes):
            log_prob = self.class_log_prior[idx].repeat(n_samples)
            for j in range(self.n_features):
                log_prob += self.feature_log_probs[j][idx, X[:, j]]
            log_probs[:, idx] = log_prob

        return self.classes[torch.argmax(log_probs, dim=1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y.to(self.device)).float().mean().item()


if __name__ == "__main__":
    torch.manual_seed(0)

    n_samples = 100
    n_features = 3
    X = torch.randint(0, 3, (n_samples, n_features))
    y = torch.randint(0, 2, (n_samples,))


    nb = DiscreteNaiveBayes()
    nb.fit(X, y)
    y_pred = nb.predict(X)
    acc = nb.score(X, y)

    print("预测:", y_pred[:10])
    print(f"Accuracy: {acc:.4f}")
