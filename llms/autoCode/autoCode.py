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
    r"__import__\(['\"]os['\"]\)\.system",
    r"eval\(", r"exec\("
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
            clipped = np.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            loss = -np.minimum(ratio * adv, clipped * adv)

            grad_logits = probs.copy()
            grad_logits[a] -= 1
            
            grad_W1_total = np.zeros_like(self.W1)
            
            for i, st in enumerate(strategies):
                emb = self.get_strategy_emb(st)
                grad = grad_logits[i] * loss
                
                self.strategy_emb[st] -= self.lr * grad * h
                
                grad_W1_total += np.outer(s, (1 - h**2) * emb * grad)

            self.W1 -= self.lr * grad_W1_total

class StrategyMemory:
    def __init__(self, generator=None):
        self.data = {}
        self.generator = generator
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
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
        top = sorted(self.data.items(), key=lambda x: x[1]["success"]/(x[1]["success"]+x[1]["fail"]), reverse=True)[:3]
        prompt = f"基于这些成功策略生成3个更具体的Python修复策略：{[s for s,_ in top]}。输出JSON: {{\"strategies\":[\"...\",\"...\"]}}"
        try:
            raw = self.generator.generate(prompt)
            data = self.generator.parse_json(raw)
            for s in data.get("strategies", []):
                if s not in self.data: self.data[s] = {"success": 1, "fail": 1}
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
            "options": {"temperature": 0.2} # 稍微降低温度提高稳定性
        }, timeout=120)
        return resp.json()["response"]

    def is_safe(self, code, cmd):
        combined = code + " " + cmd
        for pattern in DANGER_PATTERNS:
            if re.search(pattern, combined, re.I):
                return False, pattern
        return True, None

    def run(self, cmd):
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            return res.returncode == 0, res.stdout + res.stderr
        except Exception as e:
            return False, str(e)

    def parse_json(self, text):
        match = re.search(r"\{.*\}", text, re.S)
        if match: return json.loads(match.group())
        raise ValueError("LLM 未返回有效 JSON")

    def get_state(self, err, code, step):
        return np.array([
            "SyntaxError" in err, "NameError" in err, "ImportError" in err,
            min(len(code)/1000, 1.0), step / 5.0
        ])

    def run_auto_coder(self, max_steps=5):
        current_prompt = f"""
需求：{self.user_description}
请生成Python代码。必须严格按JSON格式输出，不要输出解释文字，不要Markdown块：
{{
  "path": "./out",
  "main": "python ./out/main.py",
  "files": [ {{ "path": "./out/main.py", "code": "..." }} ]
}}
"""
        states, actions, old_probs, rewards, strategies_list = [], [], [], [], []

        for step in range(max_steps):
            print(f"\n--- 步骤 {step+1} ---")
            try:
                raw = self.generate(current_prompt)
                data = self.parse_json(raw)
                file_info = data["files"][0]
                current_code = file_info["code"]
                exec_cmd = data["main"]

                # 安全审计
                safe, reason = self.is_safe(current_code, exec_cmd)
                if not safe:
                    print(f"安全拦截：检测到危险模式 [{reason}]")
                    log = f"Security Violation: Forbidden pattern '{reason}' detected."
                    success = False
                else:
                    target_path = file_info["path"]
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(current_code)
                    success, log = self.run(exec_cmd)

                state = self.get_state(log, current_code, step)
                all_strategies = sorted(list(set(self.base_strategies) | set(self.memory.data.keys())))
                weights = np.array([self.memory.get_weight(s) for s in all_strategies])
                action, prob = self.policy.sample(state, all_strategies, weights)
                strategy = all_strategies[action]

                reward = 2.0 if success else -1.0
                self.memory.update(strategy, success)
                
                states.append(state); strategies_list.append(all_strategies)
                actions.append(action); old_probs.append(prob); rewards.append(reward)

                if success:
                    print("任务达成！")
                    break
                
                print(f"尝试失败，应用策略: {strategy}")
                current_prompt = f"""
需求：{self.user_description}
当前代码：
{current_code}
运行错误：
{log}
修复策略：{strategy}

请修复并继续以JSON格式输出。
"""
            except Exception as e:
                print(f"发生错误: {e}")
                break

        # 统一训练
        if states: 
            self.policy.update(states, strategies_list, actions, old_probs, rewards)
        
        self.memory.evolve()
        self.memory.save()
        return (data if 'data' in locals() else None), (log if 'log' in locals() else "未执行")

if __name__ == "__main__":
    req = "写一个简单的Python脚本，计算 1 到 100 的总和并打印"
    gen = CodeGenerator(req)
    final_out, final_log = gen.run_auto_coder()
    
    if final_out:
        print("\n[最终代码路径]:", final_out["files"][0]["path"])
    print("\n[终端最后日志]:\n", final_log)

