import json
import os
import subprocess
import requests
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MEMORY_FILE = "strategy_memory.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] 运行设备: {device}")

DANGER_PATTERNS = [
    r"os\.remove", r"os\.unlink", r"os\.rmdir", r"os\.removedirs",
    r"shutil\.rmtree", r"os\.system", r"subprocess\.",
    r"pty\.spawn", r"rm\s+", r"mkfs", r"dd\s+",
    r"eval\(", r"exec\("
]

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, emb_dim=32, lr=1e-3):
        super().__init__()
        self.fc = nn.Linear(state_dim, emb_dim)
        self.emb_dim = emb_dim
        self.strategy_embs = nn.ParameterDict()
        self.eps_clip = 0.2

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def init_strategies(self, strategies):
        for s in strategies:
            if s not in self.strategy_embs:
                self.strategy_embs[s] = nn.Parameter(
                    torch.randn(self.emb_dim, device=device) * 0.1
                )

    def forward(self, s, strategies):
        h = torch.tanh(self.fc(s))  # [emb_dim]

        emb_matrix = torch.stack([self.strategy_embs[s] for s in strategies])  # [N, emb]
        logits = torch.matmul(emb_matrix, h)  # [N]

        log_probs = F.log_softmax(logits, dim=0)
        probs = torch.exp(log_probs)
        return probs, log_probs

    def sample(self, s_np, strategies, weights_np=None):
        self.eval()
        with torch.no_grad():
            s = torch.from_numpy(s_np).float().to(device)
            probs, log_probs = self.forward(s, strategies)

            if weights_np is not None:
                w = torch.tensor(weights_np, dtype=torch.float32, device=device)
                probs = probs * (w + 0.1)
                probs = probs / (probs.sum() + 1e-8)
                log_probs = torch.log(probs + 1e-8)

            dist = torch.distributions.Categorical(probs)
            a = dist.sample()

            return a.item(), log_probs[a].item()

    def update(self, states, strategies, actions, old_log_probs, rewards):
        self.train()

        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        if len(rewards) > 1:
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            adv = rewards

        total_loss = 0

        for i in range(len(states)):
            s = torch.from_numpy(states[i]).float().to(device)
            a = actions[i]
            old_log_p = torch.tensor(old_log_probs[i], device=device).detach()
            advantage = adv[i]

            probs, log_probs = self.forward(s, strategies)

            log_p = log_probs[a]
            ratio = torch.exp(log_p - old_log_p)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            total_loss += -torch.min(surr1, surr2)

        if total_loss == 0:
            return

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()


class StrategyMemory:
    def __init__(self):
        self.data = {}
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except:
                self.data = {}

    def get_weight(self, s):
        stat = self.data.get(s, {"success": 1, "fail": 1})
        return stat["success"] / (stat["success"] + stat["fail"])

    def update(self, s, success):
        if s not in self.data:
            self.data[s] = {"success": 1, "fail": 1}
        if success:
            self.data[s]["success"] += 1
        else:
            self.data[s]["fail"] += 1

    def save(self):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)


class CodeGenerator:
    def __init__(self, task):
        self.task = task
        self.url = "http://localhost:11434/api/generate"
        self.model = "deepseek-r1:latest"

        self.memory = StrategyMemory()
        self.base_strategies = ["精准修复错误行", "补全缺失变量", "增加异常处理", "替换为更稳定写法"]

        self.policy = PPOPolicy(5).to(device)

    def generate(self, prompt):
        try:
            r = requests.post(self.url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }, timeout=120)
            return r.json()["response"]
        except Exception as e:
            return str(e)

    def parse_json(self, text):
        text = re.sub(r"```json|```", "", text)

        stack = []
        start = None

        for i, c in enumerate(text):
            if c == '{':
                if not stack:
                    start = i
                stack.append(c)
            elif c == '}':
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        try:
                            return json.loads(text[start:i+1])
                        except:
                            continue

        raise ValueError("JSON解析失败")

    def is_safe(self, code):
        for p in DANGER_PATTERNS:
            if re.search(p, code, re.I):
                return False
        return True

    def build_strategy_space(self):
        return sorted(list(set(self.base_strategies) | set(self.memory.data.keys())))

    def run_auto_coder(self, steps=5):
        prompt = f"任务:{self.task}\n返回JSON格式"

        strategies = self.build_strategy_space()
        self.policy.init_strategies(strategies)

        hist = {"s":[], "a":[], "p":[], "r":[]}

        for step in range(steps):
            print(f"\n[Step {step+1}]")

            raw = self.generate(prompt)

            try:
                data = self.parse_json(raw)
                code = data["files"][0]["code"]
                cmd = data["main"]
            except Exception as e:
                print("解析失败:", e)
                continue

            if not self.is_safe(code):
                success, log = False, "Security violation"
            else:
                with open("tmp.py", "w", encoding="utf-8") as f:
                    f.write(code)

                try:
                    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                    success = res.returncode == 0
                    log = res.stdout + res.stderr
                except Exception as e:
                    success, log = False, str(e)

            state = np.array([
                float("SyntaxError" in log),
                float("NameError" in log),
                float("ImportError" in log),
                min(len(code)/2000, 1),
                step/5
            ], dtype=np.float32)

            weights = [self.memory.get_weight(s) for s in strategies]

            a, logp = self.policy.sample(state, strategies, weights)

            reward = 2.0 if success else -1.0
            if "SyntaxError" in log:
                reward -= 0.5
            if "Security" in log:
                reward -= 1.0

            self.memory.update(strategies[a], success)

            hist["s"].append(state)
            hist["a"].append(a)
            hist["p"].append(logp)
            hist["r"].append(reward)

            if success:
                print("成功:", log)
                break
            else:
                print("失败:", log[:80])
                prompt = f"原任务:{self.task}\n错误:{log}\n代码:\n{code}\n使用策略【{strategies[a]}】修复"

        if hist["s"]:
            self.policy.update(hist["s"], strategies, hist["a"], hist["p"], hist["r"])

        self.memory.save()


if __name__ == "__main__":
    gen = CodeGenerator("写一个Python脚本，计算1到100的和并打印")
    gen.run_auto_coder()
