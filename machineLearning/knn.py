# -*- coding:utf-8 -*-

import heapq
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Node:
    def __init__(self, data=None, father=None, left=None, right=None, feature=None):
        self.data = data
        self.father = father
        self.lchild = left
        self.rchild = right
        self.feature = feature

class KDTree:
    def __init__(self):
        self.root = Node()

    def get_var(self, data, row_indexes, feature_index):
        sum1 = 0
        sum2 = 0
        n = len(row_indexes)
        for i in row_indexes:
            sum1 = sum1 + data[i][feature_index]
            sum2 = sum2 + data[i][feature_index] ** 2
        return sum2 / n - sum2 / n ** 2

    def get_max_variance(self, data, row_indexes):
        max_var = -1
        current_dm = -1
        for dm in range(len(data[0])):
            current_var = self.get_var(data, row_indexes, dm)
            if current_var > max_var:
                max_var = current_var
                current_dm = dm
        return current_dm

    def get_median_location(self, data, row_indexes, feature_index):
        mid = len(row_indexes) // 2
        select_data = [(idx, data[idx][feature_index]) for idx in row_indexes]
        sorted_data = select_data
        sorted(sorted_data, key=(lambda x: x[1]))
        return sorted_data[mid][0]

    def split_feature(self, data, row_indexes, feature_index, mid_index):
        left_subtree = []
        right_subtree = []
        mid_value = data[mid_index][feature_index]
        for idx in row_indexes:
            if idx == mid_index:
                continue
            if data[idx][feature_index] <= mid_value:
                left_subtree.append(idx)
            else:
                right_subtree.append(idx)
        return left_subtree, right_subtree

    def build_KDTree(self, X, Y):
        row_indexes = [i for i in range(len(X))]
        tree_node = self.root
        queue = [(tree_node, row_indexes)]
        while queue:
            sub_node, idx = queue.pop(0)
            if len(idx) == 1:
                sub_node.feature = -1
                sub_node.data = (X[idx[0]], Y[idx[0]])
                continue
            max_varience_index = self.get_max_variance(X, idx)
            mid_index = self.get_median_location(X, idx, max_varience_index)
            ltree, rtree = self.split_feature(X, idx, max_varience_index, mid_index)
            sub_node.feature = max_varience_index
            sub_node.data = (X[mid_index], Y[mid_index])
            if ltree:
                sub_node.lchild = Node()
                sub_node.lchild.father = sub_node
                queue.append((sub_node.lchild, ltree))
            if rtree:
                sub_node.rchild = Node()
                sub_node.rchild.father = sub_node
                queue.append((sub_node.rchild, rtree))

    def distance(self, aim_node, node):
        dis = 0
        for i in range(len(node.data[0])):
            dis += (aim_node[i] - node.data[0][i]) ** 2
        return dis

    def dis_split(self, aim_node, split):
        return aim_node[split.feature] - split.data[0][split.feature]

    def isleaf(self, x):
        if x.feature == -1:
            return 1
        else:
            return 0

    def search_tree(self, aim_node, k=1):
        node = self.root
        search_paths = []
        max_heap = []
        heapq.heappush(max_heap, (float('-inf'), None))
        while node:
            search_paths.append(node)
            if aim_node[node.feature] < node.data[0][node.feature]:
                node = node.lchild
            else:
                node = node.rchild
        while search_paths:
            current_node = search_paths.pop()
            dis_aim_cur = self.distance(aim_node, current_node)
            dis_aim_cur = -dis_aim_cur
            min_dis = max_heap[0][0]
            if len(max_heap) < k:
                heapq.heappush(max_heap, (dis_aim_cur, current_node))  # 入堆
            elif dis_aim_cur > min_dis:
                _ = heapq.heappop(max_heap)
                heapq.heappush(max_heap, (dis_aim_cur, current_node))
            if self.isleaf(current_node): 
                continue
            dis_aim_split = self.dis_split(aim_node, current_node)
            min_dis = max_heap[0][0]
            if len(max_heap) < k or -(dis_aim_split) > min_dis:
                if aim[current_node.feature] < current_node.data[0][current_node.feature]:
                     child_node = current_node.rchild
                else:
                     child_node = current_node.lchild
                while child_node:
                    search_paths.append(child_node)
                    if aim_node[child_node.feature] > child_node.data[0][child_node.feature]:
                        child_node = child_node.rchild
                    else:
                        child_node = child_node.lchild
        return max_heap

    def predict_classification(self, aim_node, K):
        y = self.search_tree(aim_node, K)
        class_count = {}
        for i in y:
            if i[1].data[1] in class_count:
                class_count[i[1].data[1]] += 1
            else:
                class_count[i[1].data[1]] = 1
        class_aim = -1
        max_cnt = -1
        for k, v in class_count.items():
            if v > max_cnt:
                max_cnt = v
                class_aim = k
        return class_aim

if __name__ == '__main__':
    X1 = [[6.27, 5.5], [1.24, -2.86], [17.05, -12.79], [-6.88, -5.4], [-2.96, -0.5], [7.75, -22.68],
        [10.8, -5.03], [-4.6, -10.55], [-4.96, 12.61], [1.75, 12.26], [15.31, -13.16], [7.83, 15.7],[14.63, -0.35]]
    Y1 = [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1]

    T = KDTree()
    T.build_KDTree(X1, Y1)

    aim = [-1, -5]
    K = input("请输入近邻个数：")
    K = int(K)
    h = T.search_tree(aim, K)

    L = [heapq.heappop(h) for i in range(len(h))]
    print(L)

    plt.scatter([x[0] for x in X1], [x[1] for x in X1], c='blue', label='给定数据点')
    plt.scatter(aim[0], aim[1], c='red', label='目标点')
    for i in range(len(L)):
        plt.scatter(L[i][1].data[0][0], L[i][1].data[0][1], c='yellow')
    plt.legend()
    plt.show()

