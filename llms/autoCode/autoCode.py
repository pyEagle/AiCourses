import json
import os
import subprocess
import requests
import re
import numpy as np
import ast
import random

MEMORY_FILE = "strategy_memory.json"


class PPOPolicy:
    def __init__(self, state_dim, emb_dim=32, lr=0.01):
        self.W1 = np.random.randn(state_dim, emb_dim) * 0.1
        self.lr = lr
        self.eps_clip = 0.2

        self.emb_dim = emb_dim
        self.strategy_emb = {}  # 🔥 strategy -> vector

    def get_strategy_emb(self, strategy):
        if strategy not in self.strategy_emb:
            self.strategy_emb[strategy] = np.random.randn(self.emb_dim) * 0.1
        return self.strategy_emb[strategy]

    def forward(self, s, strategies):
        h = np.tanh(s @ self.W1)

        logits = []
        for st in strategies:
            emb = self.get_strategy_emb(st)
            logits.append(np.dot(h, emb))

        logits = np.array(logits)

        exp = np.exp(logits - np.max(logits))
        probs = exp / (np.sum(exp) + 1e-8)
        return probs, h

    def sample(self, s, strategies, weights=None):
        probs, _ = self.forward(s, strategies)

        if weights is not None:
            probs = probs * weights
            probs = probs / (np.sum(probs) + 1e-8)

        idx = np.random.choice(len(strategies), p=probs)
        return idx, probs[idx], probs

    def update(self, states, strategies_list, actions, old_probs, rewards):
        rewards = np.array(rewards)
        advantages = rewards - rewards.mean()

        for s, strategies, a, old_p, adv in zip(states, strategies_list, actions, old_probs, advantages):

            probs, h = self.forward(s, strategies)
            p = probs[a]

            ratio = p / (old_p + 1e-8)
            clipped = np.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            loss = -np.minimum(ratio * adv, clipped * adv)

            # grad logits
            grad_logits = probs.copy()
            grad_logits[a] -= 1

            for i, st in enumerate(strategies):
                emb = self.get_strategy_emb(st)

                grad = grad_logits[i] * loss

                # 更新 embedding
                self.strategy_emb[st] -= self.lr * grad * h

                # 更新 state encoder
                self.W1 -= self.lr * np.outer(
                    s,
                    (1 - h**2) * emb * grad
                )


class StrategyMemory:
    def __init__(self, generator=None):
        self.data = {}
        self.generator = generator

        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                self.data = json.load(f)

    def get_weight(self, strategy):
        stat = self.data.get(strategy, {"success": 1, "fail": 1})
        return stat["success"] / (stat["success"] + stat["fail"])

    def update(self, strategy, success):
        if strategy not in self.data:
            self.data[strategy] = {"success": 1, "fail": 1}

        if success:
            self.data[strategy]["success"] += 1
        else:
            self.data[strategy]["fail"] += 1

    def evolve(self):
        if len(self.data) < 2 or self.generator is None:
            return

        sorted_strats = sorted(
            self.data.items(),
            key=lambda x: x[1]["success"] / (x[1]["success"] + x[1]["fail"]),
            reverse=True
        )

        top_strategies = [s for s, _ in sorted_strats[:3]]

        prompt = f"""
基于这些策略生成新策略:
{top_strategies}

输出JSON:
{{"strategies":["...","..."]}}
"""

        try:
            raw = self.generator.generate(prompt)
            data = json.loads(re.search(r"\{.*\}", raw, re.S).group())
            candidates = data.get("strategies", [])
        except:
            return

        for c in candidates:
            if c not in self.data:
                self.data[c] = {"success": 1, "fail": 1}

    def save(self):
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)


class CodeGenerator:
    def __init__(self, user_description):
        self.user_description = user_description
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "deepseek-r1:latest"

        self.memory = StrategyMemory(generator=self)

        self.strategies = [
            "精准修复错误行",
            "补全缺失变量",
            "简化逻辑",
            "重写函数实现",
            "增加异常处理",
            "替换为更稳定写法"
        ]

        self.policy = PPOPolicy(state_dim=5)

    def generate(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3
        }
        resp = requests.post(self.ollama_url, json=payload, timeout=60)
        return resp.json()["response"]

    def run(self, cmd):
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            return res.returncode == 0, res.stdout + res.stderr
        except Exception as e:
            return False, str(e)

    def get_state(self, err, code, step):
        err_vec = [
            "SyntaxError" in err,
            "NameError" in err,
            "ImportError" in err
        ]
        code_len = len(code) / 500
        return np.array(err_vec + [code_len, step])

    def compute_reward(self, success, log, code):
        if success:
            return 2.0 + min(len(code) / 200, 1.0)

        r = -1.0
        if "SyntaxError" in log:
            r -= 0.5
        if "NameError" in log:
            r -= 0.3
        return r

    def parse_json(self, text):
        match = re.search(r"\{.*\}", text, re.S)
        return json.loads(match.group())

    def run_auto_coder(self, max_steps=5):

        prompt = f"""
生成Python代码(JSON输出):
需求:{self.user_description}
"""

        states, actions, old_probs, rewards = [], [], [], []
        strategies_list = []

        for step in range(max_steps):

            raw = self.generate(prompt)
            data = self.parse_json(raw)

            os.makedirs(data["path"], exist_ok=True)
            code = data["files"][0]["code"]

            with open("./out/main.py", "w") as f:
                f.write(code)

            success, log = self.run(data["main"])

            state = self.get_state(log, code, step)

            all_strategies = list(set(self.strategies) | set(self.memory.data.keys()))

            weights = np.array([self.memory.get_weight(s) for s in all_strategies])

            action, prob, _ = self.policy.sample(state, all_strategies, weights)
            strategy = all_strategies[action]

            reward = self.compute_reward(success, log, code)

            self.memory.update(strategy, success)

            states.append(state)
            strategies_list.append(all_strategies)
            actions.append(action)
            old_probs.append(prob)
            rewards.append(reward)

            if success:
                print("成功")
                break

            prompt = f"""
修复代码:
错误:{log}
代码:{code}
策略:{strategy}
输出JSON
"""

        self.policy.update(states, strategies_list, actions, old_probs, rewards)

        self.memory.evolve()
        self.memory.save()

        return data, log


def main():
    req = "写一个函数计算两个数的和，并打印结果"
    gen = CodeGenerator(req)

    out, log = gen.run_auto_coder()

    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("运行结果:", log)


if __name__ == "__main__":
    main()

