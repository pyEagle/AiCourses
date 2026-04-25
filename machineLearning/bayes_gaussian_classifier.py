# -*- coding:utf-8 -*-

import torch


class GaussianNaiveBayes:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = None
        self.means = None
        self.vars = None
        self.priors = None

    def fit(self, X, y):
        X = X.to(self.device)
        y = y.to(self.device)
        self.classes = torch.unique(y)
        n_features = X.shape[1]

        means = []
        vars_ = []
        priors = []

        for c in self.classes:
            X_c = X[y == c]
            means.append(X_c.mean(dim=0))
            vars_.append(X_c.var(dim=0, unbiased=False))
            priors.append(X_c.shape[0] / X.shape[0])

        self.means = torch.stack(means)
        self.vars = torch.stack(vars_)
        self.priors = torch.tensor(priors, device=self.device)

    def _gaussian_log_prob(self, x, mean, var):
        return -0.5 * torch.log(2 * torch.pi * var) - (x - mean) ** 2 / (2 * var)

    def predict(self, X):
        X = X.to(self.device)
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_probs = []

        for idx, c in enumerate(self.classes):
            # 对每个类计算log(P(X|C)) + log(P(C))
            mean = self.means[idx]
            var = self.vars[idx]
            prior = torch.log(self.priors[idx])
            log_likelihood = self._gaussian_log_prob(X, mean, var).sum(dim=1) + prior
            log_probs.append(log_likelihood)

        log_probs = torch.stack(log_probs, dim=1)  # shape: (n_samples, n_classes)
        predictions = torch.argmax(log_probs, dim=1)
        return self.classes[predictions]

    def score(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y.to(self.device)).float().mean().item()


if __name__ == "__main__":
    torch.manual_seed(0)

    n_samples = 200
    n_features = 3
    X_class0 = torch.randn(n_samples // 2, n_features) + 1.0  # 类0
    X_class1 = torch.randn(n_samples // 2, n_features) + 5.0  # 类1
    X = torch.vstack([X_class0, X_class1])
    y = torch.hstack([torch.zeros(n_samples // 2), torch.ones(n_samples // 2)]).long()

    perm = torch.randperm(n_samples)
    X, y = X[perm], y[perm]

    gnb = GaussianNaiveBayes()
    gnb.fit(X, y)
    y_pred = gnb.predict(X)
    acc = gnb.score(X, y)
    print(f"预测: {y_pred}")
    print(f"Accuracy: {acc:.4f}")
