# -*- coding:utf-8 -*-
import os
import json
import hashlib
import traceback
import subprocess
import requests
import torch
from sentence_transformers import SentenceTransformer


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
        best_sim, best_arm = 0.0, None

        for arm in self.arms:
            sim = self._calculate_similarity(new_vec, arm["vec"])
            if sim > best_sim:
                best_sim, best_arm = sim, arm

        if best_sim > 0.85:
            return best_arm["id"]

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

        alphas = [arm["history"].get(context_key, [1.1, 1.1])[0] for arm in self.arms]
        betas = [arm["history"].get(context_key, [1.1, 1.1])[1] for arm in self.arms]

        dist = torch.distributions.Beta(
            torch.tensor(alphas, device=self.device),
            torch.tensor(betas, device=self.device)
        )

        return self.arms[torch.argmax(dist.sample()).item()]["prompt"]

    def update_stats(self, arm_id, context, reward):
        for arm in self.arms:
            if arm["id"] == arm_id:
                if context not in arm["history"]:
                    arm["history"][context] = [1.1, 1.1]
                arm["history"][context][0] += reward
                arm["history"][context][1] += (1.0 - reward)
                arm["score"] = 0.7 * arm["score"] + 0.3 * reward
                break
        self._save()

    def _save(self):
        save_data = []
        for a in self.arms:
            temp = a.copy()
            temp["vec"] = temp["vec"].cpu().tolist()
            save_data.append(temp)

        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

    def _load(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for a in data:
                    a["vec"] = torch.tensor(a["vec"]).to(self.device)
                self.arms = data
            except:
                traceback.print_exc()


class SecureRunner:
    def __init__(self, root="./sandbox"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def run(self, spec):
        try:
            for file_info in spec["files"]:
                with open(os.path.join(self.root, file_info["pyfile"]), "w", encoding="utf-8") as f:
                    f.write(file_info["code"])

            res = subprocess.run(
                spec["main"], shell=True, cwd=self.root,
                capture_output=True, text=True, timeout=15
            )
            return (res.returncode == 0), res.stdout if res.returncode == 0 else res.stderr

        except Exception as e:
            return False, str(e)


class ResearchAgent:
    def __init__(self, task, custom_prompt_template=None):
        self.task = task
        self.custom_prompt_template = custom_prompt_template
        self.bandit = SemanticBanditCore()
        self.runner = SecureRunner()
        self.api_url = "http://localhost:11434/api/generate"

    def _llm_query(self, prompt, is_json=True):
        payload = {
            "model": "deepseek-r1:latest",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        if is_json:
            payload["format"] = "json"

        try:
            r = requests.post(self.api_url, json=payload, timeout=120)
            return r.json().get("response", "")
        except:
            return ""

    def run(self, iterations=3):
        prompt = self.custom_prompt_template

        for _ in range(iterations):
            raw = self._llm_query(prompt)
            try:
                spec = json.loads(raw)
                ok, out = self.runner.run(spec)
                if ok:
                    print("✅ 成功\n", out)
                    return spec
            except:
                pass

        return None


class SSEOrchestrator:
    def __init__(self, task):
        self.task = task
        self.api_url = "http://localhost:11434/api/generate"

    def _llm_query(self, prompt):
        payload = {
            "model": "deepseek-r1:latest",
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        try:
            r = requests.post(self.api_url, json=payload, timeout=120)
            return r.json().get("response", "")
        except:
            return "{}"

    def analyze_and_decompose(self):
        prompt = f"""
拆解任务并构建DAG：

任务：{self.task}

输出：
{{
  "modules": [{{"name":"","desc":"","inputs":"","outputs":""}}],
  "graph": {{
    "nodes": [],
    "edges": []
  }}
}}
"""
        res = self._llm_query(prompt)
        return json.loads(res)

    def assemble_and_execute(self, plan):
        # 👉 DGA执行顺序
        dga = DGARunner(plan["graph"])
        order = dga.topo_sort()

        modules_str = json.dumps(plan["modules"], ensure_ascii=False, indent=2)

        prompt = f"""
根据DGA生成代码：

任务：{self.task}

模块：
{modules_str}

执行顺序：
{order}

要求：
1 严格按顺序执行
2 上一函数输出作为下一函数输入
3 生成main()

只输出JSON
"""

        agent = ResearchAgent(self.task, prompt)
        return agent.run()


# ==========================================
# 6. 运行
# ==========================================
if __name__ == "__main__":
    if os.path.exists("bandit_memory.json"):
        os.remove("bandit_memory.json")

    task = "生成100个随机数，筛选质数，计算均值和方差"

    sse = SSEOrchestrator(task)

    plan = sse.analyze_and_decompose()
    print("🧠 SSE:", plan)

    sse.assemble_and_execute(plan)
