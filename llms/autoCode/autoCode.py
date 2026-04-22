import json
import os
import subprocess
import requests
import re
import numpy as np

MEMORY_FILE = "strategy_memory.json"

DANGER_PATTERNS = [
    r"os\.remove", r"os\.unlink", r"os\.rmdir", r"os\.removedirs", r"shutil\.rmtree",
    r"os\.system", r"subprocess\.", r"pty\.spawn",
    r"rm\s+", r"mkfs", r"dd\s+", r"> /dev/",
    r"eval\(", r"exec\(",
    r"__import__", r"getattr", r"Popen", r"system\("
]

class PPOPolicy:
    def __init__(self, state_dim, emb_dim=32, lr=0.01):
        self.W1 = np.random.randn(state_dim, emb_dim) * 0.1
        self.lr = lr
        self.eps_clip = 0.2
        self.emb_dim = emb_dim
        self.strategy_emb = {}

    def get_strategy_emb(self, strategy):
        if strategy not in self.strategy_emb:
            self.strategy_emb[strategy] = np.random.randn(self.emb_dim) * 0.1
        return self.strategy_emb[strategy]

    def forward(self, s, strategies):
        h = np.tanh(s @ self.W1)
        logits = np.array([np.dot(h, self.get_strategy_emb(st)) for st in strategies])
        exp = np.exp(logits - np.max(logits))
        probs = exp / (np.sum(exp) + 1e-8)
        return probs, h

    def sample(self, s, strategies, weights=None):
        probs, _ = self.forward(s, strategies)
        if weights is not None:
            probs = probs * (weights + 0.1)
            probs /= (np.sum(probs) + 1e-8)
        idx = np.random.choice(len(strategies), p=probs)
        return idx, probs[idx]

    def update(self, states, strategies_list, actions, old_probs, rewards):
        rewards = np.array(rewards)
        if len(rewards) > 1:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = rewards

        for s, strategies, a, old_p, adv in zip(states, strategies_list, actions, old_probs, advantages):
            probs, h = self.forward(s, strategies)
            p = probs[a]
            ratio = p / (old_p + 1e-8)

            surr1 = ratio * adv
            surr2 = np.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            
            if surr1 > surr2:
                continue

            grad_logits = probs.copy()
            grad_logits[a] -= 1

            grad_W1_total = np.zeros_like(self.W1)
            for i, st in enumerate(strategies):
                emb = self.get_strategy_emb(st)
                grad = grad_logits[i] * adv 
                
                self.strategy_emb[st] -= self.lr * grad * h
                grad_W1_total += np.outer(s, (1 - h**2) * emb * grad)

            self.W1 -= self.lr * grad_W1_total

        if len(self.strategy_emb) > 500:
            self.strategy_emb = dict(list(self.strategy_emb.items())[-300:])

class StrategyMemory:
    def __init__(self, generator=None):
        self.data = {}
        self.generator = generator
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except: self.data = {}

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
        if len(self.data) < 2 or self.generator is None: return
        top = sorted(self.data.items(), key=lambda x: x[1]["success"]/(x[1]["success"]+x[1]["fail"]), reverse=True)[:3]
        prompt = f"基于以下策略生成3个更具体的Python修复动作：{[s for s,_ in top]}。输出JSON格式：{{\"strategies\":[\"...\"]}}"
        try:
            raw = self.generator.generate(prompt)
            data = self.generator.parse_json(raw)
            for s in data.get("strategies", []):
                if isinstance(s, str) and len(s) < 50 and s not in self.data:
                    self.data[s] = {"success": 1, "fail": 1}
        except: pass

    def save(self):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

class CodeGenerator:
    def __init__(self, user_description):
        self.user_description = user_description
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "deepseek-r1:latest"
        self.memory = StrategyMemory(generator=self)
        self.base_strategies = ["精准修复错误行", "补全缺失变量", "增加异常处理", "替换为更稳定写法"]
        self.policy = PPOPolicy(state_dim=5)

    def generate(self, prompt):
        resp = requests.post(self.ollama_url, json={
            "model": self.model_name, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.2}
        }, timeout=120)
        return resp.json()["response"]

    def is_safe(self, code, cmd):
        combined = (code + " " + cmd).lower()
        for pattern in DANGER_PATTERNS:
            if re.search(pattern, combined, re.I):
                return False, pattern
        return True, None

    def parse_json(self, text):
        # 针对 R1 的加固：匹配最后一个 JSON 块，并移除可能存在的 Markdown 标记
        text = re.sub(r"```json|```", "", text)
        matches = re.findall(r"\{.*\}", text, re.S)
        if not matches: raise ValueError("未找到JSON")
        return json.loads(matches[-1].strip())

    def get_state(self, err, code, step):
        return np.array([
            "SyntaxError" in err, "NameError" in err, "ImportError" in err,
            min(len(code)/2000, 1.0), step / 5.0
        ])

    def compute_reward(self, success, log, code):
        if success: return 2.0 + min(len(code)/500, 0.5)
        r = -1.0
        if "Security Violation" in log: r -= 2.0
        if "SyntaxError" in log: r -= 0.5
        return r

    def run_auto_coder(self, max_steps=5):
        current_prompt = f"任务：{self.user_description}\n必须输出JSON，格式如下：\n{{\"path\":\"./out\",\"main\":\"python ./out/main.py\",\"files\":[{{\"path\":\"./out/main.py\",\"code\":\"...\"}}]}}"
        
        states, actions, old_probs, rewards, strategies_list = [], [], [], [], []

        for step in range(max_steps):
            print(f"\n--- 迭代步数 {step+1} ---")
            try:
                raw = self.generate(current_prompt)
                data = self.parse_json(raw)
                f_info = data["files"][0]
                code, cmd = f_info["code"], data["main"]

                # 安全审计
                safe, reason = self.is_safe(code, cmd)
                if not safe:
                    success, log = False, f"Security Violation: {reason}"
                else:
                    os.makedirs(os.path.dirname(f_info["path"]), exist_ok=True)
                    with open(f_info["path"], "w", encoding="utf-8") as fp:
                        fp.write(code)
                    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                    success, log = (res.returncode == 0), res.stdout + res.stderr

                # RL 状态提取与采样
                state = self.get_state(log, code, step)
                all_st = sorted(list(set(self.base_strategies) | set(self.memory.data.keys())))
                weights = np.array([self.memory.get_weight(s) for s in all_st])
                idx, prob = self.policy.sample(state, all_st, weights)
                
                strategy = all_st[idx]
                reward = self.compute_reward(success, log, code)
                self.memory.update(strategy, success)
                
                states.append(state); strategies_list.append(all_st)
                actions.append(idx); old_probs.append(prob); rewards.append(reward)

                if success:
                    print(f"任务成功！结果：\n{log}")
                    break

                print(f"失败，尝试策略: {strategy}")
                current_prompt = f"需求：{self.user_description}\n当前错误：{log}\n代码：\n{code}\n请根据策略“{strategy}”修复并重新输出JSON。"

            except Exception as e:
                print(f"运行异常: {e}")
                break

        if states: self.policy.update(states, strategies_list, actions, old_probs, rewards)
        self.memory.evolve(); self.memory.save()
        return (data if 'data' in locals() else None), (log if 'log' in locals() else "未运行")

if __name__ == "__main__":
    task = "写一个Python脚本，实现两个数相加"
    gen = CodeGenerator(task)
    gen.run_auto_coder()
