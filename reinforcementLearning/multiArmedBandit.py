# -*- coding:utf-8 -*-

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiArmedBandit:
    def __init__(self, k_arms=10):
        self.k_arms = k_arms
        
        self.true_q_values = torch.randn(k_arms, device=device)
        self.best_action = torch.argmax(self.true_q_values).item()

    def step(self, action):
        noise = torch.randn((), device=device)
        reward = self.true_q_values[action] + noise
        return reward


class EpsilonGreedyAgent:
    def __init__(self, k_arms, epsilon=0.1):
        self.k_arms = k_arms
        self.epsilon = epsilon
        
        self.q_estimates = torch.zeros(k_arms, device=device)
        self.action_counts = torch.zeros(k_arms, device=device)

    def choose_action(self):
        rand_prob = torch.rand(1, device=device)
        
        if rand_prob < self.epsilon:
            action = torch.randint(0, self.k_arms, (1,), device=device).item()
        else:
            action = torch.argmax(self.q_estimates).item()
            
        return action

    def update_estimate(self, action, reward):
        self.action_counts[action] += 1
        
        count = self.action_counts[action]
        step_size = 1.0 / count
        
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
                  f"选择最优臂比例: {optimal_rate:.1f}% | "
                  f"平均收益: {avg_reward:.2f}")

    print("-" * 50)
    print("\n训练结束！最终结果：")

    print("臂编号 | 真实期望价值 | 智能体估计价值 | 拉动次数")
    for i in range(K_ARMS):
        true_val = env.true_q_values[i].item()
        est_val = agent.q_estimates[i].item()
        counts = agent.action_counts[i].item()
        
        marker = " (*最优*)" if i == env.best_action else ""
        print(f"  {i:2d}   |    {true_val:>7.2f}    |     {est_val:>7.2f}     |  {int(counts):>4d}{marker}")


if __name__ == "__main__":
    train_and_evaluate()

