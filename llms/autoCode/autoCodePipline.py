# -*- coding:utf-8 -*-
import os
import json
import hashlib
import traceback
import subprocess
import requests
import torch
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. DAG执行器（不变）
# ==========================================
class DGARunner:
    def __init__(self, graph):
        self.nodes = graph["nodes"]
        self.edges = graph["edges"]

    def topo_sort(self):
        from collections import defaultdict, deque

        indegree = {n: 0 for n in self.nodes}
        adj = defaultdict(list)

        for u, v in self.edges:
            adj[u].append(v)
            indegree[v] += 1

        q = deque([n for n in self.nodes if indegree[n] == 0])
        order = []

        while q:
            cur = q.popleft()
            order.append(cur)
            for nxt in adj[cur]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    q.append(nxt)

        if len(order) != len(self.nodes):
            raise ValueError("❌ 非法DAG（存在环）")

        return order


# ==========================================
# 2. Bandit（轻微增强：支持context）
# ==========================================
class SemanticBanditCore:
    def __init__(self, capacity=20):
        self.arms = []
        self.capacity = capacity
        self.memory_path = "bandit_memory.json"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.model = SentenceTransformer('./model/LLms/sentence_transformers/', device=self.device)
        except:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        self._load()

    def _get_feature_vector(self, text):
        return self.model.encode(text, convert_to_tensor=True).to(self.device)

    def _calculate_similarity(self, vec1, vec2):
        sim = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
        return sim.item()

    def add_or_update_arm(self, prompt):
        new_vec = self._get_feature_vector(prompt)

        for arm in self.arms:
            if self._calculate_similarity(new_vec, arm["vec"]) > 0.9:
                return arm["id"]

        arm_id = hashlib.md5(prompt.encode()).hexdigest()[:8]
        self.arms.append({
            "id": arm_id,
            "prompt": prompt,
            "vec": new_vec,
            "history": {},
            "score": 0.5
        })

        if len(self.arms) > self.capacity:
            self.arms.sort(key=lambda x: x["score"], reverse=True)
            self.arms.pop()

        self._save()
        return arm_id

    def sample(self, context_key, fallback):
        if not self.arms:
            return fallback

        scores = []
        for arm in self.arms:
            a, b = arm["history"].get(context_key, [1.1, 1.1])
            sample = torch.distributions.Beta(
                torch.tensor(a), torch.tensor(b)
            ).sample().item()
            scores.append(sample)

        return self.arms[int(torch.argmax(torch.tensor(scores)))]["prompt"]

    def update_stats(self, arm_id, context, reward):
        for arm in self.arms:
            if arm["id"] == arm_id:
                if context not in arm["history"]:
                    arm["history"][context] = [1.1, 1.1]
                arm["history"][context][0] += reward
                arm["history"][context][1] += (1 - reward)
                arm["score"] = 0.7 * arm["score"] + 0.3 * reward
        self._save()

    def _save(self):
        data = []
        for a in self.arms:
            temp = a.copy()
            temp["vec"] = temp["vec"].cpu().tolist()
            data.append(temp)
        with open(self.memory_path, "w") as f:
            json.dump(data, f)

    def _load(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path) as f:
                data = json.load(f)
                for a in data:
                    a["vec"] = torch.tensor(a["vec"]).to(self.device)
                self.arms = data


# ==========================================
# 3. 安全执行器（不变）
# ==========================================
class SecureRunner:
    def __init__(self, root="./sandbox"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def run(self, code):
        try:
            path = os.path.join(self.root, "main.py")
            with open(path, "w") as f:
                f.write(code)

            res = subprocess.run(
                "python main.py",
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=10
            )
            return res.returncode == 0, res.stdout if res.returncode == 0 else res.stderr
        except Exception as e:
            return False, str(e)


# ==========================================
# 4. 🔥 ResearchAgent（核心升级：反思+多轮）
# ==========================================
class ResearchAgent:
    def __init__(self, task):
        self.task = task
        self.api_url = "http://localhost:11434/api/generate"
        self.bandit = SemanticBanditCore()
        self.runner = SecureRunner()

    def _llm(self, prompt):
        r = requests.post(self.api_url, json={
            "model": "deepseek-r1:latest",
            "prompt": prompt,
            "stream": False
        })
        return r.json().get("response", "")

    def solve_node(self, node, context_input, max_retry=3):
        context_key = f"{self.task}_{node}"

        base_prompt = f"""
任务: {self.task}
当前模块: {node}
输入: {context_input}

请写Python代码，只输出print最终结果
"""

        arm_id = self.bandit.add_or_update_arm(base_prompt)
        prompt = self.bandit.sample(context_key, base_prompt)

        last_error = ""

        for i in range(max_retry):
            full_prompt = prompt + f"\n错误:{last_error}" if last_error else prompt

            code = self._llm(full_prompt)

            ok, out = self.runner.run(code)

            if ok:
                self.bandit.update_stats(arm_id, context_key, 1.0)
                return out.strip()

            else:
                last_error = out
                self.bandit.update_stats(arm_id, context_key, 0.0)

        raise RuntimeError(f"节点失败: {node}")


# ==========================================
# 5. SSE（升级：逐节点执行）
# ==========================================
class SSEOrchestrator:
    def __init__(self, task):
        self.task = task
        self.api_url = "http://localhost:11434/api/generate"

    def _llm(self, prompt):
        r = requests.post(self.api_url, json={
            "model": "deepseek-r1:latest",
            "prompt": prompt,
            "stream": False,
            "format": "json"
        })
        return r.json().get("response", "{}")

    def analyze(self):
        prompt = f"""
拆解任务为DAG:

{self.task}

输出:
{{
 "modules":[{{"name":""}}],
 "graph":{{"nodes":[],"edges":[]}}
}}
"""
        return json.loads(self._llm(prompt))

    def execute(self, plan):
        dga = DGARunner(plan["graph"])
        order = dga.topo_sort()

        agent = ResearchAgent(self.task)

        memory = {}

        for node in order:
            print(f"\n🚀 执行节点: {node}")

            input_data = {k: memory[k] for k in memory}

            out = agent.solve_node(node, input_data)

            memory[node] = out

            print("✅ 输出:", out)

        return memory


# ==========================================
# 6. 运行
# ==========================================
if __name__ == "__main__":
    task = "生成100个随机数，筛选质数，计算均值和方差"

    sse = SSEOrchestrator(task)

    plan = sse.analyze()
    print("🧠 DAG:", plan)

    result = sse.execute(plan)

    print("\n🏁 最终结果:", result)
