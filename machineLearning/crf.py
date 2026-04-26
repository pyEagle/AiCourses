# -*- coding:utf-8 -*-

import numpy as np

class LinearChainCRF:
    def __init__(self, tags, learning_rate=0.01):
        self.tags = tags
        self.tag2id = {tag: i for i, tag in enumerate(tags)}
        self.id2tag = {i: tag for i, tag in enumerate(tags)}
        self.num_tags = len(tags)
        self.lr = learning_rate
        
        self.transition_weights = np.zeros((self.num_tags, self.num_tags))
        
        self.emission_weights = {} 

    def _get_emission_score(self, word, tag_id):
        return self.emission_weights.get((word, tag_id), 0.0)

    def _log_sum_exp(self, vec):
        max_score = np.max(vec)
        return max_score + np.log(np.sum(np.exp(vec - max_score)))

    # ==========================================================
    # 1. 概率计算问题
    # ==========================================================
    def _forward(self, x):
        T = len(x)
        alpha = np.zeros(self.num_tags)
        for i in range(self.num_tags):
            alpha[i] = self._get_emission_score(x[0], i)
        
        for t in range(1, T):
            new_alpha = np.zeros(self.num_tags)
            for curr_tag in range(self.num_tags):
                # score = alpha_{t-1} + transition + emission
                scores = []
                for prev_tag in range(self.num_tags):
                    score = alpha[prev_tag] + \
                            self.transition_weights[prev_tag, curr_tag] + \
                            self._get_emission_score(x[t], curr_tag)
                    scores.append(score)
                new_alpha[curr_tag] = self._log_sum_exp(np.array(scores))
            alpha = new_alpha
            
        return self._log_sum_exp(alpha), alpha # 返回最终 Z 和最后的 alpha 仅作参考

    def get_probability(self, x, y):
        # 1. 计算路径得分
        score = 0
        for t in range(len(x)):
            # 发射分
            score += self._get_emission_score(x[t], self.tag2id[y[t]])
            # 转移分 (从 t=1 开始)
            if t > 0:
                prev = self.tag2id[y[t-1]]
                curr = self.tag2id[y[t]]
                score += self.transition_weights[prev, curr]
        
        # 2. 计算 Z(x)
        log_Z, _ = self._forward(x)
        
        # 3. 计算对数概率然后 exp
        log_prob = score - log_Z
        return np.exp(log_prob)

    # ==========================================================
    # 2. 预测问题 (使用 Viterbi 算法进行 Decoding)
    # ==========================================================
    def predict(self, x):
        T = len(x)
        # dp[t][tag] 存储时刻 t 到达 tag 的最大得分
        dp = np.zeros((T, self.num_tags))
        # path[t][tag] 记录到达该位置的上一个最优 tag，用于回溯
        path = np.zeros((T, self.num_tags), dtype=int)

        for i in range(self.num_tags):
            dp[0][i] = self._get_emission_score(x[0], i)

        # 递推
        for t in range(1, T):
            for curr_tag in range(self.num_tags):
                # 寻找从哪个 prev_tag 转移过来得分最高
                best_score = -float('inf')
                best_prev = -1
                
                emission = self._get_emission_score(x[t], curr_tag)
                
                for prev_tag in range(self.num_tags):
                    score = dp[t-1][prev_tag] + \
                            self.transition_weights[prev_tag, curr_tag] + \
                            emission
                    
                    if score > best_score:
                        best_score = score
                        best_prev = prev_tag
                
                dp[t][curr_tag] = best_score
                path[t][curr_tag] = best_prev

        # 回溯最优路径
        best_path = []
        # 找到最后一个时刻得分最高的tag
        last_tag = np.argmax(dp[T-1])
        best_path.append(last_tag)
        
        for t in range(T-1, 0, -1):
            last_tag = path[t][last_tag]
            best_path.append(last_tag)
            
        best_path.reverse()
        return [self.id2tag[i] for i in best_path]

    # ==========================================================
    # 3. 学习问题
    # ==========================================================
    def _backward(self, x):
        T = len(x)
        beta = np.zeros((T, self.num_tags))
        # 最后一个时刻 beta 设为 0 (log空间中的1)
        # beta[T-1, :] = 0.0  <- 已经是初始化值
        
        for t in range(T-2, -1, -1):
            for curr_tag in range(self.num_tags):
                scores = []
                for next_tag in range(self.num_tags):
                    # beta_{t} = sum(beta_{t+1} * trans * emit)
                    score = beta[t+1][next_tag] + \
                            self.transition_weights[curr_tag, next_tag] + \
                            self._get_emission_score(x[t+1], next_tag)
                    scores.append(score)
                beta[t][curr_tag] = self._log_sum_exp(np.array(scores))
        return beta

    def train_one_step(self, x, y):
        T = len(x)
        
        alpha = np.zeros((T, self.num_tags))
        for i in range(self.num_tags):
            alpha[0][i] = self._get_emission_score(x[0], i)
        for t in range(1, T):
            for curr in range(self.num_tags):
                scores = [alpha[t-1][prev] + self.transition_weights[prev, curr] + \
                          self._get_emission_score(x[t], curr) for prev in range(self.num_tags)]
                alpha[t][curr] = self._log_sum_exp(np.array(scores))
        
        log_Z = self._log_sum_exp(alpha[T-1])
        beta = self._backward(x)

        node_marginals = np.exp(alpha + beta - log_Z)
        
        emission_grads = {}
        for t in range(T):
            true_tag_id = self.tag2id[y[t]]
            key = (x[t], true_tag_id)
            emission_grads[key] = emission_grads.get(key, 0) + 1.0
            
            for tag_id in range(self.num_tags):
                key_model = (x[t], tag_id)
                prob = node_marginals[t][tag_id]
                emission_grads[key_model] = emission_grads.get(key_model, 0) - prob

        for key, grad in emission_grads.items():
            self.emission_weights[key] = self.emission_weights.get(key, 0) + self.lr * grad

        if T > 1:
            for t in range(T-1):
                prev_true = self.tag2id[y[t]]
                curr_true = self.tag2id[y[t+1]]
                self.transition_weights[prev_true, curr_true] += self.lr * 1.0
                
                # log_prob = alpha[t][i] + trans[i,j] + emit[t+1][j] + beta[t+1][j] - log_Z
                for i in range(self.num_tags):
                    for j in range(self.num_tags):
                        log_prob_edge = alpha[t][i] + \
                                        self.transition_weights[i, j] + \
                                        self._get_emission_score(x[t+1], j) + \
                                        beta[t+1][j] - log_Z
                        prob_edge = np.exp(log_prob_edge)
                        self.transition_weights[i, j] -= self.lr * prob_edge

# ==========================================================
# 4. 案例演示
# ==========================================================
if __name__ == "__main__":
    train_x = ["time", "flies", "like", "an", "arrow"]
    train_y = ["N", "V", "V", "N", "N"]
    
    # 实例化模型
    crf = LinearChainCRF(tags=["N", "V"], learning_rate=0.1)
    
    print("--- 1. 训练前预测 ---")
    print(f"Prediction: {crf.predict(train_x)}")
    print(f"Prob of Truth: {crf.get_probability(train_x, train_y):.6f}")

    print("\n--- 2. 开始训练 (Learning Problem) ---")
    # 简单训练 100 轮
    for epoch in range(100):
        crf.train_one_step(train_x, train_y)
        if epoch % 20 == 0:
            prob = crf.get_probability(train_x, train_y)
            print(f"Epoch {epoch}: Probability of true path = {prob:.4f}")

    print("\n--- 3. 训练后结果 ---")
    # 预测 (Prediction Problem)
    pred_path = crf.predict(train_x)
    print(f"Input: {train_x}")
    print(f"Predicted: {pred_path}")
    print(f"True Label: {train_y}")
    
    # 概率计算 (Probability Problem)
    prob_true = crf.get_probability(train_x, train_y)
    prob_wrong = crf.get_probability(train_x, ["V", "N", "N", "V", "V"]) # 一个错误的序列
    print(f"Prob(True Path): {prob_true:.6f}")
    print(f"Prob(Wrong Path): {prob_wrong:.6f}")
    
    # 验证特征学到了什么
    n_id, v_id = crf.tag2id['N'], crf.tag2id['V']
    print("\n--- 4. 模型学到的特征 (权重分析) ---")
    print(f"Transition N->V score: {crf.transition_weights[n_id, v_id]:.2f}")
    print(f"Transition V->V score: {crf.transition_weights[v_id, v_id]:.2f}")
    print(f"Emission (flies -> V): {crf._get_emission_score('flies', v_id):.2f}")
    print(f"Emission (flies -> N): {crf._get_emission_score('flies', n_id):.2f}")
