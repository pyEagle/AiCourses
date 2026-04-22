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

DIM = 384
MCTS_ITERS = 20
MAX_CHILDREN = 3
TIMEOUT = 5
MAX_TESTS = 20
RIDGE = 1e-6 

class Embedding:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, text):
        v = np.array(self.model.encode(text)).astype("float32")
        return v / (np.linalg.norm(v) + 1e-10)

class Thompson:
    def __init__(self, dim):
        self.A = np.eye(dim)
        self.b = np.zeros((dim, 1))

    def sample_theta(self):
        A_inv = np.linalg.pinv(self.A + RIDGE * np.eye(self.A.shape[0]))
        mu = A_inv @ self.b
        cov = A_inv
        return np.random.multivariate_normal(mu.flatten(), cov)

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
        except:
            return None

        class Transformer(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, (int, float)):
                    return ast.copy_location(
                        ast.Constant(node.value + random.uniform(-1, 1)),
                        node
                    )
                return node

        new_tree = Transformer().visit(tree)
        try:
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
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        matches = re.findall(r"\{.*?\}", text, re.S)
        for m in reversed(matches):
            try:
                return json.loads(m)
            except:
                continue
        return None

    def init_code(self):
        prompt = f"任务:{self.task}\n输出JSON代码"
        out = self.safe_json(self.call_llm(prompt))
        if out and "files" in out:
            return "\n".join(f["code"] for f in out["files"])
        return "print('BMI')"

    def mutate_llm(self, code):
        prompt = f"优化或修复代码:\n{code}\n返回JSON"
        out = self.safe_json(self.call_llm(prompt))
        if out and "files" in out:
            return "\n".join(f["code"] for f in out["files"])
        return None

    def critic_score(self, code):
        prompt = f"""
评估以下代码质量（正确性、鲁棒性、可读性），0~1分：
{code}
返回JSON: {{"score":0.0}}
"""
        out = self.safe_json(self.call_llm(prompt))
        if out and "score" in out:
            try:
                return float(out["score"])
            except:
                return 0.5
        return 0.5

    def breaker_tests(self, code):
        prompt = f"""
代码:
{code}
生成3个极端测试:
{{"tests":[{{"input":"...","output":"..."}}]}}
"""
        out = self.safe_json(self.call_llm(prompt))
        if out and "tests" in out:
            self.tests.extend(out["tests"])

        if len(self.tests) > MAX_TESTS:
            self.tests = self.tests[-MAX_TESTS:]

    def init_tests(self):
        prompt = f"任务:{self.task}\n生成测试JSON"
        out = self.safe_json(self.call_llm(prompt))

        if out and "tests" in out and len(out["tests"]) > 0:
            return out["tests"]

        return [
            {"input": "170 60\n", "output": "正常"},
            {"input": "180 90\n", "output": "超重"}
        ]

    def execute(self, code):
        try:
            os.makedirs("out", exist_ok=True)
            path = "out/main.py"

            with open(path, "w") as f:
                f.write(code)

            passed = 0
            for t in self.tests:
                res = subprocess.run(
                    ["python", path],
                    input=t["input"] + "\n",
                    text=True,
                    capture_output=True,
                    timeout=TIMEOUT
                )

                if res.returncode != 0:
                    print("ERR:", res.stderr)

                if t["output"] in res.stdout:
                    passed += 1

            return passed / max(len(self.tests), 1)
        except Exception as e:
            print("EXEC ERROR:", e)
            return 0

    def score_node(self, node):
        emb = self.embed.encode(node.code)

        ts_score = self.bandit.score(emb)

        uct = (node.value / (node.visits + 1e-6)) + \
              math.sqrt(math.log(node.parent.visits + 1) / (node.visits + 1e-6))

        return 0.7 * ts_score + 0.3 * uct

    def select(self, node):
        while node.children:
            node = max(node.children, key=lambda c: self.score_node(c))
        return node

    def expand(self, node):
        for _ in range(MAX_CHILDREN):
            new_code = self.mutate_llm(node.code)
            if not new_code:
                new_code = self.mutator.mutate(node.code)

            if new_code:
                node.children.append(Node(new_code, node))

    def simulate(self, node):
        emb = self.embed.encode(node.code)

        exec_reward = self.execute(node.code)
        critic_reward = self.critic_score(node.code)

        reward = 0.7 * exec_reward + 0.3 * critic_reward

        if exec_reward < 1.0:
            self.breaker_tests(node.code)

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
        best_score = -1

        for i in range(MCTS_ITERS):
            print(f"\n=== Iter {i+1} ===")

            leaf = self.select(root)
            self.expand(leaf)

            for c in leaf.children:
                r = self.simulate(c)
                print("reward:", r)

                if r > best_score:
                    best_score = r
                    best_node = c

                self.backprop(c, r)

        print("\n🏆 Best Code:\n")
        print(best_node.code)


if __name__ == "__main__":
    task = "创建一个函数，计算两个数的和，并在test.py中调用它。"
    agent = MultiAgentMCTS(task)
    agent.run()

