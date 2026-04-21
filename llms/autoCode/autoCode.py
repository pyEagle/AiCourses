import json
import os
import subprocess
import requests
import re
import math
import random

from collections import Counter
from sentence_transformers import SentenceTransformer

MEMORY_FILE = "memory.json"

correct_example = '{"path": "./", "files": [{"code": "print(1+2)", "pyfile": "test.py"}], "main": "python test.py"}'


class SimpleEmbedding:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.use_transformer = True
        print("✅ 使用 Transformer Embedding")

    def vectorize(self, text):
        return self.model.encode(text).tolist()

    def cosine(self, v1, v2):
        import numpy as np
        v1, v2 = np.array(v1), np.array(v2)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0
        return float(v1 @ v2 / (n1 * n2))


class Memory:
    def __init__(self):
        self.embed = SimpleEmbedding()
        self.data = self.load()

        self.sim_threshold = 0.85
        self.max_size = 50

    def load(self):
        if not os.path.exists(MEMORY_FILE):
            return []
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)

    def save(self):
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def add_case(self, task, code, success):
        v_new = self.embed.vectorize(task)

        for item in self.data:
            v_old = item.get("emb") or self.embed.vectorize(item["task"])
            sim = self.embed.cosine(v_new, v_old)

            if sim > self.sim_threshold:
                if success:
                    item["success"] += 1
                else:
                    item["fail"] += 1

                item["count"] += 1

                if success:
                    item["code"] = code

                item["emb"] = v_old  # 保留embedding

                self.clean()
                self.save()
                return

        self.data.append({
            "task": task,
            "code": code,
            "success": 1 if success else 0,
            "fail": 0 if success else 1,
            "count": 1,
            "emb": v_new   # ⭐存embedding
        })

        self.clean()
        self.save()

    def clean(self):
        if not self.data:
            return

        self.data = [
            x for x in self.data
            if x["success"] >= x["fail"]
        ]

        def score(x):
            return (x["success"] / (x["success"] + x["fail"] + 1)) * math.log(1 + x["count"])

        self.data.sort(key=score, reverse=True)
        self.data = self.data[:self.max_size]

    def retrieve(self, task, top_k=2):
        if not self.data:
            return []

        v1 = self.embed.vectorize(task)
        scores = []

        for item in self.data:
            v2 = item.get("emb") or self.embed.vectorize(item["task"])
            sim = self.embed.cosine(v1, v2)

            alpha = item["success"] + 1
            beta = item["fail"] + 1
            theta = random.betavariate(alpha, beta)

            context_weight = (sim + 1) / 2  # 映射到 [0,1]

            score = sim * theta * context_weight
            scores.append(score)

        indices = list(range(len(self.data)))
        chosen = set()

        while len(chosen) < min(top_k, len(indices)):
            idx = random.choices(indices, weights=scores, k=1)[0]
            chosen.add(idx)

        return [self.data[i] for i in chosen]


class CodeGenerator:
    def __init__(self, user_description):
        self.user_description = user_description
        self.ollama_url = "http://localhost:11434/api/generate"
        self.memory = Memory()

        self.error_prompt = "用户需求: {}\n错误信息: {}\n请修复代码并输出JSON。示例: {}"

    def enhance_prompt(self, base_prompt):
        cases = self.memory.retrieve(self.user_description)

        if not cases:
            return base_prompt

        hint = "\n【历史经验】\n"
        for c in cases:
            hint += f"任务: {c['task']}\n代码:\n{c['code']}\n"

        return base_prompt + hint

    def call_llm(self, prompt):
        payload = {
            "model": "deepseek-r1:latest",
            "prompt": prompt,
            "format": "json",
            "stream": False
        }
        r = requests.post(self.ollama_url, json=payload)
        return r.json().get("response", "")

    def run_code(self, cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return r.returncode == 0, r.stdout + r.stderr

    def debug_code(self, error):
        prompt = self.error_prompt.format(self.user_description, error, correct_example)
        return self.call_llm(prompt)

    def fix_and_retry(self, prompt, max_retries=5):
        prompt = self.enhance_prompt(prompt)

        for i in range(max_retries):
            print("尝试:", i + 1)

            raw = self.call_llm(prompt)

            try:
                out = json.loads(raw)

                cmd = out["main"]

                for f in out["files"]:
                    os.makedirs(out["path"], exist_ok=True)
                    with open(os.path.join(out["path"], f["pyfile"]), "w") as fp:
                        fp.write(f["code"])

                ok, log = self.run_code(cmd)

                code_text = "\n".join(f["code"] for f in out["files"])

                if ok:
                    self.memory.add_case(self.user_description, code_text, True)
                    return out, log
                else:
                    self.memory.add_case(self.user_description, code_text, False)
                    prompt = self.enhance_prompt(self.debug_code(log))

            except Exception as e:
                prompt = self.error_prompt.format(self.user_description, str(e), correct_example)

        raise Exception("失败")


def main():
    task = "写一个函数计算两个数相加并输出"

    prompt = f"""
任务: {task}
要求:
1. Python
2. JSON输出
3. 可执行
示例: {correct_example}
"""

    gen = CodeGenerator(task)
    out, log = gen.fix_and_retry(prompt)

    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(log)


if __name__ == "__main__":
    main()

