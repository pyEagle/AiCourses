# -*- coding:utf-8 -*-

import torch


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, device='cpu'):
        self.eps = eps
        self.min_samples = min_samples
        self.device = device
        self.labels_ = None

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        n_points = X.shape[0]
        self.labels_ = torch.full((n_points,), -1, dtype=torch.int32, device=self.device)
        cluster_id = 0

        # 计算距离
        dist_matrix = torch.cdist(X, X, p=2)
        visited = torch.zeros(n_points, dtype=torch.bool, device=self.device)

        for i in range(n_points):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._region_query(dist_matrix, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1 # 噪声
            else:
                self._expand_cluster(X, dist_matrix, i, neighbors, cluster_id, visited)
                cluster_id += 1

        return self

    def _region_query(self, dist_matrix, point_idx):
        neighbors = torch.nonzero(dist_matrix[point_idx] <= self.eps).flatten()
        return neighbors.tolist()

    def _expand_cluster(self, X, dist_matrix, point_idx, neighbors, cluster_id, visited):
        self.labels_[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            n_idx = neighbors[i]
            if not visited[n_idx]:
                visited[n_idx] = True
                n_neighbors = self._region_query(dist_matrix, n_idx)
                if len(n_neighbors) >= self.min_samples:
                    neighbors += [x for x in n_neighbors if x not in neighbors]
            if self.labels_[n_idx] == -1:
                self.labels_[n_idx] = cluster_id
            i += 1

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_.cpu().numpy()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import make_moons
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dbscan = DBSCAN(eps=0.2, min_samples=5, device=device)
    labels = dbscan.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10')
    plt.title("DBSCAN Clustering")
    plt.show()

