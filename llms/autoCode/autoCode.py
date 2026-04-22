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
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.W1 = np.random.randn(state_dim, 32) * 0.1
        self.W2 = np.random.randn(32, action_dim) * 0.1

        self.lr = lr
        self.eps_clip = 0.2

    def forward(self, s):
        h = np.tanh(s @ self.W1)
        logits = h @ self.W2

        exp = np.exp(logits - np.max(logits))
        probs = exp / (np.sum(exp) + 1e-8)
        return probs

    def sample(self, s, weights=None):
        probs = self.forward(s)

        if weights is not None:
            probs = probs * weights
            probs = probs / (np.sum(probs) + 1e-8)

        action = np.random.choice(len(probs), p=probs)
        return action, probs[action], probs

    def update(self, states, actions, old_probs, rewards):
        rewards = np.array(rewards)
        advantages = rewards - rewards.mean()

        for s, a, old_p, adv in zip(states, actions, old_probs, advantages):
            probs = self.forward(s)
            p = probs[a]

            ratio = p / (old_p + 1e-8)
            clipped = np.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip)

            loss = -np.minimum(ratio * adv, clipped * adv)

            grad_logits = probs.copy()
            grad_logits[a] -= 1

            h = np.tanh(s @ self.W1)

            grad_W2 = np.outer(h, grad_logits) * loss
            grad_W1 = np.outer(
                s,
                (1 - h**2) * (self.W2 @ grad_logits)
            ) * loss

            self.W2 -= self.lr * grad_W2
            self.W1 -= self.lr * grad_W1


class StrategyMemory:
    def __init__(self, generator=None):
        self.data = {}
        self.generator = generator  # 🔥 注入LLM

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
请基于以下优秀代码修复策略，生成3个新的、更具体、更可执行的策略：

已有策略:
{top_strategies}

要求：
1. 必须是具体操作（例如：补全import、处理None、修复索引错误）
2. 不要重复已有策略
3. 每条一句话
4. JSON输出：
{{"strategies": ["...", "..."]}}
"""

        try:
            raw = self.generator.generate(prompt)
            data = json.loads(re.search(r"\{.*\}", raw, re.S).group())
            candidates = data.get("strategies", [])
        except Exception:
            return

        candidates = list(set(candidates))
        candidates = [c for c in candidates if isinstance(c, str) and len(c) < 50]

        if not candidates:
            return

        scored = []
        for c in candidates:
            w = self.get_weight(c) if c in self.data else 0.5
            score = w + random.random() * 0.5
            scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        best_new = scored[0][0]

        if best_new not in self.data:
            self.data[best_new] = {"success": 1, "fail": 1}

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

        self.policy = PPOPolicy(state_dim=5, action_dim=len(self.strategies))

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
你是资深Python工程师，请生成可运行代码。

严格要求：
1. 必须可直接运行
2. 不允许省略
3. 包含main入口
4. 输出JSON格式

格式:
{{
 "path":"./out",
 "files":[{{"pyfile":"main.py","code":"..."}}],
 "main":"python ./out/main.py"
}}

需求:{self.user_description}
"""

        states, actions, old_probs, rewards = [], [], [], []

        for step in range(max_steps):

            raw = self.generate(prompt)
            data = self.parse_json(raw)

            os.makedirs(data["path"], exist_ok=True)
            code = data["files"][0]["code"]

            with open("./out/main.py", "w") as f:
                f.write(code)

            success, log = self.run(data["main"])

            state = self.get_state(log, code, step)

            weights = np.array([self.memory.get_weight(s) for s in self.strategies])

            action, prob, _ = self.policy.sample(state, weights)
            strategy = self.strategies[action]

            reward = self.compute_reward(success, log, code)

            self.memory.update(strategy, success)

            states.append(state)
            actions.append(action)
            old_probs.append(prob)
            rewards.append(reward)

            if success:
                print("成功")
                break

            prompt = f"""
修复代码:

错误:
{log}

代码:
{code}

修复策略:
{strategy}

要求:
1. 修复错误
2. 保持可运行
3. 只输出JSON
"""

        self.policy.update(states, actions, old_probs, rewards)

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


