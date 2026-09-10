# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"系统检测：正在使用 [{device}] 运行计算...")

class BellmanVExplorer:
    def __init__(self, size=5, gamma=0.9):
        self.size = size
        self.gamma = gamma
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上, 下, 左, 右
        self.action_names = ['↑', '↓', '←', '→']
        
        self.goal = (size - 1, size - 1)  # 右下角为终点 (Terminal)
        self.trap = (size // 2, size // 2) # 正中间为陷阱 (Non-terminal)

        self.V = torch.zeros((size, size), dtype=torch.float32, device=device)
        
        self.R = torch.full((size, size), -0.01, device=device) # 每走一步的微小惩罚
        self.R[self.goal] = 1.0   # 终点大奖
        self.R[self.trap] = -1.0  # 陷阱惩罚

    def get_next_state(self, r, c, action):
        nr, nc = r + action[0], c + action[1]
        if 0 <= nr < self.size and 0 <= nc < self.size:
            return nr, nc
        return r, c  

    def run_iteration(self, max_steps=100, tol=1e-6):
        print(f"开始价值迭代 (设备: {self.V.device})...")
        
        for i in range(max_steps):
            v_old = self.V.clone()
            
            for r in range(self.size):
                for c in range(self.size):
                    if (r, c) == self.goal:
                        continue
                    
                    res = []
                    for action in self.actions:
                        nr, nc = self.get_next_state(r, c, action)
                        res.append(self.R[nr, nc] + self.gamma * v_old[nr, nc])
                    
                    self.V[r, c] = torch.max(torch.stack(res))
            
            diff = torch.max(torch.abs(self.V - v_old))
            if diff < tol:
                print(f"算法在第 {i+1} 步收敛。")
                break
        
        return self.V.cpu().numpy()

    def get_policy(self):
        policy = np.full((self.size, self.size), '', dtype=object)
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) == self.goal:
                    policy[r, c] = 'G'
                    continue
                
                best_val = -float('inf')
                best_act = ' '
                for idx, action in enumerate(self.actions):
                    nr, nc = self.get_next_state(r, c, action)
                    
                    val = self.R[nr, nc].item() + self.gamma * self.V[nr, nc].item()
                    
                    if val > best_val:
                        best_val = val
                        best_act = self.action_names[idx]
                        
                policy[r, c] = best_act
                
                if (r, c) == self.trap:
                    policy[r, c] = f'T\n({best_act})'
                    
        return policy

def visualize(v_matrix, policy):
    fig, ax = plt.subplots(figsize=(7, 7))
    cax = ax.matshow(v_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    
    for i in range(v_matrix.shape[0]):
        for j in range(v_matrix.shape[1]):
            val = v_matrix[i, j]
            text = f'{val:.2f}\n{policy[i, j]}'
            color = 'white' if val < -0.3 or val > 0.6 else 'black'
            ax.text(j, i, text, va='center', ha='center', color=color, fontweight='bold')
    
    plt.title("State Value V(s) and Optimal Policy", pad=20)
    ax.set_xticks(range(v_matrix.shape[1]))
    ax.set_yticks(range(v_matrix.shape[0]))
    plt.show()

if __name__ == "__main__":
    # 初始化环境并运行
    solver = BellmanVExplorer(size=5, gamma=0.9)
    final_v = solver.run_iteration()
    optimal_policy = solver.get_policy()
    
    print("\n最终状态价值矩阵 V(s):")
    # 为了在终端打印好看一些，限制小数位数
    np.set_printoptions(precision=3, suppress=True) 
    print(final_v)
    
    # 弹出可视化窗口
    visualize(final_v, optimal_policy)
