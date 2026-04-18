# optics_torch.py

import torch
import matplotlib.pyplot as plt  # 新增，用于绘图

class OPTICS:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, X):
        X = X.to(self.device)
        n_points = X.shape[0]
        reachability = torch.full((n_points,), float('inf'), device=self.device)
        core_distance = torch.full((n_points,), float('inf'), device=self.device)
        processed = torch.zeros(n_points, dtype=torch.bool, device=self.device)
        ordering = []

        # 计算核心距离
        for i in range(n_points):
            distances = torch.norm(X - X[i], dim=1)
            sorted_distances, _ = torch.sort(distances)
            if len(sorted_distances) > self.min_samples:
                core_distance[i] = sorted_distances[self.min_samples]

        # 主循环
        for point_idx in range(n_points):
            if not processed[point_idx]:
                self._expand_cluster_order(point_idx, X, processed, reachability, core_distance, ordering)

        self.reachability_ = reachability.cpu()
        self.core_distances_ = core_distance.cpu()
        self.ordering_ = ordering
        return self

    def _expand_cluster_order(self, point_idx, X, processed, reachability, core_distance, ordering):
        seeds = []
        processed[point_idx] = True
        ordering.append(point_idx)

        neighbors = self._get_neighbors(point_idx, X)
        if core_distance[point_idx] != float('inf'):
            self._update_seeds(point_idx, neighbors, X, seeds, reachability, processed, core_distance)

            while seeds:
                seeds.sort(key=lambda x: reachability[x])
                current = seeds.pop(0)
                if not processed[current]:
                    processed[current] = True
                    ordering.append(current)
                    current_neighbors = self._get_neighbors(current, X)
                    if core_distance[current] != float('inf'):
                        self._update_seeds(current, current_neighbors, X, seeds, reachability, processed, core_distance)

    def _get_neighbors(self, point_idx, X):
        distances = torch.norm(X - X[point_idx], dim=1)
        neighbors = torch.where(distances <= self.eps)[0]
        return neighbors.tolist()

    def _update_seeds(self, point_idx, neighbors, X, seeds, reachability, processed, core_distance):
        for neighbor in neighbors:
            if not processed[neighbor]:
                new_reach_dist = max(core_distance[point_idx], torch.norm(X[point_idx] - X[neighbor]))
                if reachability[neighbor] == float('inf'):
                    reachability[neighbor] = new_reach_dist
                    seeds.append(neighbor)
                else:
                    if new_reach_dist < reachability[neighbor]:
                        reachability[neighbor] = new_reach_dist

if __name__ == "__main__":
    X = torch.randn(100, 2)

    optics = OPTICS(2.0, 5)
    optics.fit(X)

    print("有序队列: ", optics.ordering_)
    print("可达距离: ", optics.reachability_)
    print("核心距离: ", optics.core_distances_)

    # 可视化 Reachability 图
    ordered_reachability = optics.reachability_[optics.ordering_]
    plt.figure(figsize=(10, 5))
    plt.plot(ordered_reachability.numpy(), marker='o', linestyle='-', color='b')
    plt.title("OPTICS Reachability Plot")
    plt.xlabel("Points (Ordered)")
    plt.ylabel("Reachability Distance")
    plt.show()
