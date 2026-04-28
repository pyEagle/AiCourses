# -*- coding:utf-8 -*-
import os
import json
import hashlib
import traceback
import subprocess
import requests
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict, deque
import re

class DGARunner:
    def __init__(self, graph):
        self.graph = graph
        self.node_map = {node['id']: node for node in graph["nodes"]}
        self.edges = graph["edges"]

    def topo_sort(self):
        indegree = {node['id']: 0 for node in self.graph["nodes"]}
        adj = defaultdict(list)
        
        for edge in self.edges:
            source = edge['source']
            target = edge['target']
            adj[source].append(target)
            indegree[target] += 1

        q = deque([node_id for node_id in indegree if indegree[node_id] == 0])
        order = []
        
        while q:
            cur = q.popleft()
            order.append(cur)
            for nxt in adj[cur]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    q.append(nxt)

        if len(order) != len(self.graph["nodes"]):
            raise ValueError("非法DAG（存在环）")

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


class SecureRunner:
    def __init__(self, root="./sandbox"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def run(self, code):
        try:
            path = os.path.join(self.root, "main.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)

            res = subprocess.run(
                "python main.py",
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8"
            )
            return res.returncode == 0, res.stdout if res.returncode == 0 else res.stderr
        except Exception as e:
            return False, str(e)


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

    # 清理 LLM 返回的 markdown 代码块
    def _clean_code(self, code):
        code = re.sub(r"```python", "", code)
        code = re.sub(r"```", "", code)
        return code.strip()

    def solve_node(self, node_id, context_input, upstream_data=None, max_retry=2):
        node = [n for n in context_input["graph"]["nodes"] if n["id"] == node_id][0]
        context_key = f"{self.task}_{node_id}"

        base_prompt = f"""
任务: {self.task}
节点: {node_id}
节点配置: {node}
上游输入数据: {upstream_data}

请生成一段独立可运行的Python代码，只做一件事：执行该节点功能，用 print 输出结果（列表/数字直接打印）
不要解释，不要用```包裹，只输出纯Python代码。
"""

        arm_id = self.bandit.add_or_update_arm(base_prompt)
        prompt = self.bandit.sample(context_key, base_prompt)

        last_error = ""

        for i in range(max_retry):
            full_prompt = prompt + f"\n上一次错误:\n{last_error}" if last_error else prompt
            code = self._llm(full_prompt)
            code = self._clean_code(code)  # 关键修复

            ok, out = self.runner.run(code)

            if ok:
                self.bandit.update_stats(arm_id, context_key, 1.0)
                return out.strip()

            last_error = out
            self.bandit.update_stats(arm_id, context_key, 0.0)

        raise RuntimeError(f"节点失败: {node_id}\n错误信息:\n{last_error}")


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

输出严格JSON格式:
{{
 "modules": [{{"name": "random"}}, {{"name": "math"}}, {{"name": "statistics"}}],
 "graph": {{
   "nodes": [
     {{"id":"GenerateRandomNumbers","module":"random","function":"random.choices","args":{{"population":list(range(1,101)),"k":100}}}},
     {{"id":"FilterPrimes","module":"math","function":"isprime","args":{{"n":null}}}},
     {{"id":"CalculateMean","module":"statistics","function":"mean","args":{{"data":null}}}},
     {{"id":"CalculateVariance","module":"statistics","function":"variance","args":{{"data":null}}}}
   ],
   "edges": [
     {{"source":"GenerateRandomNumbers","target":"FilterPrimes"}},
     {{"source":"FilterPrimes","target":"CalculateMean"}},
     {{"source":"FilterPrimes","target":"CalculateVariance"}}
   ]
 }}
}}
"""
        return json.loads(self._llm(prompt))

    def execute(self, plan):
        dga = DGARunner(plan["graph"])
        order = dga.topo_sort()
        agent = ResearchAgent(self.task)
        memory = {"graph": plan["graph"]}

        edge_map = defaultdict(list)
        for e in plan["graph"]["edges"]:
            edge_map[e["source"]].append(e["target"])

        for node_id in order:
            print(f"\n执行节点: {node_id}")
            upstream = {src: memory[src] for src in memory if src != "graph" and node_id in edge_map[src]}
            upstream_data = list(upstream.values())[0] if upstream else None

            input_data = {k: memory[k] for k in memory}
            out = agent.solve_node(node_id, input_data, upstream_data)
            memory[node_id] = out
            print("输出:", out)

        return memory


if __name__ == "__main__":
    task = "编写一个Python脚本，找出1000以内最大的质数，代码直接打印最终结果"
    sse = SSEOrchestrator(task)
    plan = sse.analyze()
    print("DAG:", plan)

    result = sse.execute(plan)
    print("\n最终结果:", result)

