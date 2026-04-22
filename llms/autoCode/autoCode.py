import os
import json
import subprocess
import requests
import numpy as np
import re
import math
import random
import ast
from sentence_transformers import SentenceTransformer

DIM = 1024 
MCTS_ITERS = 10  
MAX_CHILDREN = 2
TIMEOUT = 5
MAX_TESTS = 20
RIDGE = 1e-6

class Embedding:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")

    def encode(self, text):
        v = self.model.encode(text, normalize_embeddings=True)
        return v.astype("float32")

class Thompson:
    def __init__(self, dim):
        self.A = np.eye(dim)
        self.b = np.zeros((dim, 1))

    def sample_theta(self):
        A_inv = np.linalg.pinv(self.A + RIDGE * np.eye(self.A.shape[0]))
        mu = A_inv @ self.b
        return np.random.multivariate_normal(mu.flatten(), A_inv)

    def score(self, x):
        theta = self.sample_theta()
        return float(np.dot(theta, x))

    def update(self, x, r):
        x = x.reshape(-1, 1)
        self.A += x @ x.T
        self.b += r * x

class ASTMutator:
    def mutate(self, code):
        try:
            tree = ast.parse(code)
            class Transformer(ast.NodeTransformer):
                def visit_Constant(self, node):
                    if isinstance(node.value, (int, float)):
                        return ast.copy_location(ast.Constant(node.value + random.uniform(-1, 1)), node)
                    return node
            new_tree = Transformer().visit(tree)
            return ast.unparse(new_tree)
        except:
            return None

class Node:
    def __init__(self, code, parent=None):
        self.code = code
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

class MultiAgentMCTS:
    def __init__(self, task):
        self.task = task
        self.embed = Embedding()
        self.bandit = Thompson(DIM)
        self.mutator = ASTMutator()
        self.ollama = "http://localhost:11434/api/generate"
        self.tests = self.init_tests()

    def call_llm(self, prompt):
        try:
            r = requests.post(self.ollama, json={
                "model": "deepseek-r1:latest",
                "prompt": prompt,
                "stream": False
            }, timeout=60)
            return r.json().get("response", "")
        except:
            return ""

    def safe_json(self, text):
        if not text: return None

        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        code_block = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            try: return json.loads(code_block.group(1))
            except: pass

        matches = re.findall(r"\{.*\}", text, re.DOTALL)
        for m in reversed(matches):
            try: return json.loads(m)
            except: continue

        return None

    def init_tests(self):
        return [
            {"input": "1 2\n", "output": "3"},
            {"input": "10 20\n", "output": "30"},
        ]

    def init_code(self):
        prompt = f"任务: {self.task}\n请严格输出JSON格式: {{\"files\":[{{\"code\":\"代码内容\"}}]}}"
        out = self.safe_json(self.call_llm(prompt))
        if out and "files" in out:
            return out["files"][0]["code"]
        return "import sys\nfor line in sys.stdin:\n    print(sum(map(int, line.split())))"

    def mutate_llm(self, code, error_msg=""):
        feedback = f"\n上一次运行报错: {error_msg}" if error_msg else ""
        prompt = f"任务: {self.task}\n当前代码:\n{code}{feedback}\n请修复并优化，严格返回JSON格式。"
        out = self.safe_json(self.call_llm(prompt))
        if out and "files" in out:
            return out["files"][0]["code"]
        return None

    def execute(self, code):
        os.makedirs("out", exist_ok=True)
        path = "out/main.py"
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        
        passed = 0
        error_msg = ""
        for t in self.tests:
            try:
                res = subprocess.run(
                    ["python", path],
                    input=t["input"],
                    text=True, capture_output=True, timeout=TIMEOUT
                )
                if res.returncode != 0:
                    error_msg = res.stderr
                if t["output"].strip() in res.stdout.strip():
                    passed += 1
            except Exception as e:
                error_msg = str(e)
        
        return passed / max(len(self.tests), 1), error_msg

    def critic_score(self, code):
        prompt = f"评估这段代码的健壮性和规范性(0-1分):\n{code}\n返回JSON: {{\"score\": 0.8}}"
        out = self.safe_json(self.call_llm(prompt))
        try: 
            return float(out.get("score", 0.5))
        except: 
            return 0.5

    def score_node(self, node):
        if not node.parent: 
            return 0

        emb = self.embed.encode(node.code)
        ts_score = self.bandit.score(emb)
        uct = (node.value / (node.visits + 1e-6)) + \
              math.sqrt(math.log(node.parent.visits + 1) / (node.visits + 1e-6))

        return 0.7 * ts_score + 0.3 * uct

    def select(self, node):
        while node.children:
            node = max(node.children, key=lambda c: self.score_node(c))
        return node

    def simulate(self, node):
        emb = self.embed.encode(node.code)
        exec_reward, error_msg = self.execute(node.code)
        critic_reward = self.critic_score(node.code)
        reward = 0.7 * exec_reward + 0.3 * critic_reward

        if exec_reward < 1.0:
            fixed_code = self.mutate_llm(node.code, error_msg)
            if fixed_code:
                node.children.append(Node(fixed_code, node))

        self.bandit.update(emb, reward)
        return reward

    def backprop(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def run(self):
        root = Node(self.init_code())
        best_node = root
        best_reward = -1

        for i in range(MCTS_ITERS):
            print(f"迭代 [{i+1}/{MCTS_ITERS}] 搜索中...")
            leaf = self.select(root)
            
            # 扩展阶段
            for _ in range(MAX_CHILDREN):
                new_code = self.mutate_llm(leaf.code)
                if not new_code: new_code = self.mutator.mutate(leaf.code)
                if new_code:
                    child = Node(new_code, leaf)
                    leaf.children.append(child)
                    
                    reward = self.simulate(child)
                    if reward > best_reward:
                        best_reward = reward
                        best_node = child
                    self.backprop(child, reward)

        print("\n" + "="*30 + "\n🏆 最终优选代码:\n" + "="*30)
        print(best_node.code)

if __name__ == "__main__":
    task = "编写一个Python脚本，实现两个数字求和的函数，输出它们的和"
    agent = MultiAgentMCTS(task)
    agent.run()

