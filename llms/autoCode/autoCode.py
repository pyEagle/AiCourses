import json
import os
import subprocess
import requests
import re
import math
import random
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

MEMORY_FILE = "memory.json"
FAISS_INDEX_FILE = "faiss.index"

correct_example = '{"path": "./", "files": [{"code": "print(1+2)", "pyfile": "test.py"}], "main": "python test.py"}'


class SimpleEmbedding:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")

    def vectorize(self, text):
        return np.array(self.model.encode(text)).astype("float32")


class Memory:
    def __init__(self):
        self.embed = SimpleEmbedding()
        self.data = self.load()

        self.dim = 384  # MiniLM维度
        self.index = self.load_faiss()

        self.max_size = 50

    def load(self):
        if not os.path.exists(MEMORY_FILE):
            return []
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)

    def save(self):
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def load_faiss(self):
        if os.path.exists(FAISS_INDEX_FILE):
            return faiss.read_index(FAISS_INDEX_FILE)
        return faiss.IndexFlatIP(self.dim)  # 余弦相似度（需归一化）

    def save_faiss(self):
        faiss.write_index(self.index, FAISS_INDEX_FILE)

    def normalize(self, v):
        return v / np.linalg.norm(v)

    def add_case(self, task, code, success):
        v = self.normalize(self.embed.vectorize(task))

        self.data.append({
            "task": task,
            "code": code,
            "success": 1 if success else 0,
            "fail": 0 if success else 1,
            "count": 1
        })

        self.index.add(np.array([v]))

        self.clean()
        self.save()
        self.save_faiss()

    def clean(self):
        if len(self.data) <= self.max_size:
            return

        def score(x):
            return (x["success"] / (x["success"] + x["fail"] + 1)) * math.log(1 + x["count"])

        self.data.sort(key=score, reverse=True)

        # 截断数据
        self.data = self.data[:self.max_size]

        # 重建 FAISS
        self.rebuild_index()

    def rebuild_index(self):
        self.index = faiss.IndexFlatIP(self.dim)

        vectors = []
        for item in self.data:
            v = self.normalize(self.embed.vectorize(item["task"]))
            vectors.append(v)

        if vectors:
            self.index.add(np.array(vectors))

    def retrieve(self, task, top_k=2):
        if not self.data:
            return []

        v = self.normalize(self.embed.vectorize(task)).reshape(1, -1)

        D, I = self.index.search(v, min(top_k, len(self.data)))

        results = []
        for idx in I[0]:
            if idx < len(self.data):
                results.append(self.data[idx])

        return results


class CodeGenerator:
    def __init__(self, user_description):
        self.user_description = user_description
        self.ollama_url = "http://localhost:11434/api/generate"
        self.memory = Memory()

        self.error_prompt = "用户需求: {}\n错误信息: {}\n请修复代码并输出JSON。示例: {}"

        self.dangerous_patterns = [
            r"rm\s+-rf\s+\/",
            r"rm\s+-rf\s+\*",
            r"os\.system\(.+rm\s+-rf.+\)",
            r"subprocess\..*\(.+rm\s+-rf.+\)",
            r"shutil\.rmtree",
            r"os\.remove\(",
            r"os\.unlink\(",
            r"eval\(",
            r"exec\("
        ]

    def is_dangerous_code(self, code_text):
        for pattern in self.dangerous_patterns:
            if re.search(pattern, code_text):
                return True, pattern
        return False, None

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

                code_text = "\n".join(f["code"] for f in out["files"])

                dangerous, pattern = self.is_dangerous_code(code_text)
                if dangerous:
                    print(f"⚠️ 危险代码: {pattern}")
                    self.memory.add_case(self.user_description, code_text, False)
                    prompt = self.error_prompt.format(
                        self.user_description,
                        f"危险操作: {pattern}",
                        correct_example
                    )
                    continue

                for f in out["files"]:
                    os.makedirs(out["path"], exist_ok=True)
                    with open(os.path.join(out["path"], f["pyfile"]), "w") as fp:
                        fp.write(f["code"])

                ok, log = self.run_code(out["main"])

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
角色：资深人工智能专家
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
