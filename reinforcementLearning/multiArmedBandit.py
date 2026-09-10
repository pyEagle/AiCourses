# -*- coding:utf-8 -*-
import torch
import random

device = torch.device("cpu") 

class MultiArmedBandit:
    def __init__(self, k_arms=10):
        self.k_arms = k_arms
        self.true_q_values = torch.randn(k_arms, device=device)
        self.best_action = torch.argmax(self.true_q_values).item()

    def step(self, action):
        # 奖励 R_t = q*(a) + 噪声
        noise = torch.randn((), device=device)
        reward = self.true_q_values[action] + noise
        return reward


class EpsilonGreedyAgent:
    def __init__(self, k_arms, epsilon=0.1, alpha=None):
        self.k_arms = k_arms
        self.epsilon = epsilon
        self.alpha = alpha # 如果为 None，则使用 1/n 样本平均
        
        self.q_estimates = torch.zeros(k_arms, device=device)
        self.action_counts = torch.zeros(k_arms, device=device)

    def choose_action(self):
        rand_prob = torch.rand(1, device=device).item()
        
        if rand_prob < self.epsilon:
            # 探索：随机选择
            action = torch.randint(0, self.k_arms, (1,), device=device).item()
        else:
            # 开发 (贪心选择，并包含随机打破平局机制 Tie-breaking)
            max_q = torch.max(self.q_estimates)
            best_actions = torch.where(self.q_estimates == max_q)[0]
            # 从所有具有最大估计值的臂中随机选一个
            idx = torch.randint(0, len(best_actions), (1,)).item()
            action = best_actions[idx].item()
            
        return action

    def update_estimate(self, action, reward):
        self.action_counts[action] += 1
        
        # 支持动态步长 (1/n) 或 静态步长 (alpha)
        step_size = self.alpha if self.alpha is not None else (1.0 / self.action_counts[action])
        
        # Q_{n+1} = Q_n + step_size * (R - Q_n)
        self.q_estimates[action] += step_size * (reward - self.q_estimates[action])


def train_and_evaluate():
    K_ARMS = 10
    STEPS = 2000
    EPSILON = 0.1

    env = MultiArmedBandit(k_arms=K_ARMS)
    agent = EpsilonGreedyAgent(k_arms=K_ARMS, epsilon=EPSILON)

    print(f"老虎机真实最优臂:  {env.best_action}")
    print(f"老虎机各臂真实价值: {[round(v.item(), 2) for v in env.true_q_values]}\n")
    print("-" * 50)
    print("开始训练...\n")

    optimal_action_count = 0
    total_reward = 0.0

    for step in range(1, STEPS + 1):
        action = agent.choose_action()
        reward = env.step(action)
        
        agent.update_estimate(action, reward)

        if action == env.best_action:
            optimal_action_count += 1
        
        total_reward += reward.item()

        if step % 500 == 0:
            optimal_rate = (optimal_action_count / step) * 100
            avg_reward = total_reward / step
            print(f"步骤 [{step}/{STEPS}] | "
                  f"选择最优臂比例: {optimal_rate:>4.1f}% | "
                  f"平均收益: {avg_reward:.2f}")

    print("-" * 50)
    print("\n训练结束！最终结果：")

    print("臂编号 | 真实期望价值 | 智能体估计价值 | 拉动次数")
    for i in range(K_ARMS):
        true_val = env.true_q_values[i].item()
        est_val = agent.q_estimates[i].item()
        counts = agent.action_counts[i].item()
        
        marker = " (*最优*)" if i == env.best_action else ""
        print(f"  {i:2d}   |    {true_val:>7.2f}    |      {est_val:>7.2f}      |  {int(counts):>4d}{marker}")


if __name__ == "__main__":
    train_and_evaluate()
