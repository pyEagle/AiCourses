# -*- coding:utf-8 -*-

import torch

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, device=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.centroids = None

    def fit(self, X):
        """
        X: 输入数据 tensor，形状 [num_samples, num_features]
        """
        X = X.to(self.device)
        num_samples, num_features = X.shape

        # 随机初始化质心
        indices = torch.randperm(num_samples)[:self.n_clusters]
        self.centroids = X[indices]

        for i in range(self.max_iters):
            distances = torch.cdist(X, self.centroids, p=2)  # shape [num_samples, n_clusters]

            # 分配
            labels = torch.argmin(distances, dim=1)

            # 更新
            new_centroids = torch.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = cluster_points.mean(dim=0)
                else:
                    # 如果某个簇没有点，随机重新初始化
                    new_centroids[k] = X[torch.randint(0, num_samples, (1,))]

            # 收敛判断
            shift = torch.norm(self.centroids - new_centroids, dim=1).sum()
            self.centroids = new_centroids
            if shift < self.tol:
                print(f"Converged at iteration {i+1}")
                break

        self.labels_ = labels

    def predict(self, X):
        X = X.to(self.device)
        distances = torch.cdist(X, self.centroids, p=2)
        labels = torch.argmin(distances, dim=1)
        return labels

if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)

    kmeans = KMeans(n_clusters=3, max_iters=100)
    kmeans.fit(X)

    print("中心坐标:\n", kmeans.centroids)
    print("预测标签:\n", kmeans.labels_)

