import json
import os
import subprocess
import requests
import re
import numpy as np
from collections import defaultdict
import ast

class Thompson:
    def __init__(self, dim):
        self.A = np.eye(dim)
        self.b = np.zeros((dim, 1))
        self.ridge = 1e-5

    def sample_theta(self):
        A_inv = np.linalg.pinv(self.A + self.ridge * np.eye(self.A.shape[0]))
        mu = A_inv @ self.b
        return np.random.multivariate_normal(mu.flatten(), A_inv)

    def score(self, x):
        theta = self.sample_theta()
        return float(np.dot(theta, x))

    def update(self, x, r):
        x = x.reshape(-1, 1)
        self.A += x @ x.T
        self.b += r * x

class CodeGenerator:
    def __init__(self, user_description):
        self.user_description = user_description
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "deepseek-r1:latest"
        
        self.dangerous_patterns = [
            r"os\.remove|os\.unlink|shutil\.rmtree",
            r"subprocess\.run.*rm|os\.system.*rm",
            r"__import__|eval\(|exec\(",
            r"open\([^)]*w[^)]*\).*[\\/]"
        ]
        
        self.history = defaultdict(list)
        self.thompson = Thompson(dim=3)

    def generate(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1
        }
        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["response"]
        except Exception as e:
            raise Exception(f"模型调用失败：{str(e)}")

    def is_dangerous(self, code):
        for p in self.dangerous_patterns:
            if re.search(p, code, re.IGNORECASE):
                return True
        return False

    def check_syntax(self, code):
        try:
            ast.parse(code)
            return True, "语法正常"
        except SyntaxError as e:
            return False, f"语法错误：{e.msg} (行{e.lineno})"

    def save(self, output):
        path = output["path"]
        os.makedirs(path, exist_ok=True)
        
        for f in output["files"]:
            code = f["code"]
            if self.is_dangerous(code):
                raise Exception("危险代码已拦截：系统禁止文件删除/执行高危命令")
            valid, err = self.check_syntax(code)
            if not valid:
                raise Exception(f"代码语法无效：{err}")
            
            file_path = os.path.join(path, f["pyfile"])
            with open(file_path, "w", encoding="utf-8") as fp:
                fp.write(code)
        return output["main"]

    def run(self, cmd):
        try:
            res = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=15
            )
            if res.returncode == 0:
                return True, res.stdout
            else:
                return False, res.stderr
        except Exception as e:
            return False, f"执行异常：{str(e)}"

    def get_features(self, err, code):
        err_type = 0
        if "SyntaxError" in err: err_type = 1
        elif "ImportError" in err: err_type = 2
        elif "NameError" in err: err_type = 3
        elif "IndexError|KeyError" in err: err_type = 4
        code_len = min(len(code) / 500, 5)
        retries = len(self.history["errors"])

        return np.array([err_type, code_len, retries])

    def choose_strategy(self, feats):
        strategies = [
            "精准修复错误行，保持其他代码不变",
            "补充缺失依赖/变量，保证代码可运行",
            "简化逻辑，保持功能完整，减少错误"
        ]
        scores = [self.thompson.score(feats) for _ in strategies]
        return strategies[np.argmax(scores)]

    def parse_json(self, text):
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)
            data = json.loads(text)
            required = ["path", "files", "main"]
            for k in required:
                if k not in data:
                    raise ValueError(f"缺少字段：{k}")
            return data
        except:
            raise ValueError("JSON格式无效")

    def run_auto_coder(self, max_retries=4):
        base_prompt = f"""
你是专业Python工程师，只输出合法JSON，不解释、不闲聊、不加注释。
用户需求：{self.user_description}
输出格式必须严格如下：
{{
    "path": "./output",
    "files": [{{"pyfile": "main.py", "code": "完整代码"}}],
    "main": "python ./output/main.py"
}}
要求：
1. 代码可直接运行
2. 无语法错误
3. 实现业务逻辑
4. 不包含高危操作
"""
        prompt = base_prompt

        for attempt in range(max_retries):
            print(f"\n==== 第 {attempt+1} 次生成 ====")
            try:
                raw = self.generate(prompt)
                output = self.parse_json(raw)
                main_cmd = self.save(output)
                success, log = self.run(main_cmd)

                feats = self.get_features(log, output["files"][0]["code"])
                self.thompson.update(feats, 1.0 if success else 0.0)

                if success:
                    print("代码运行成功！")
                    return output, log

                print(f"执行失败：{log[:200]}")
                strategy = self.choose_strategy(feats)
                prompt = f"""
{base_prompt}
错误信息：{log}
修复策略：{strategy}
请只修复错误，不要改变原有功能，输出严格JSON。
"""
            except Exception as err:
                prompt = f"{base_prompt}\n错误：{err}\n请严格输出JSON，修复问题。"

        raise Exception("达到最大重试次数，生成失败")

def main():
    requirement = "创建一个函数，计算两个数的和，并在main.py中调用并打印结果。"
    gen = CodeGenerator(requirement)
    
    try:
        final_code, run_log = gen.run_auto_coder()
        print("\n" + "="*50)
        print("最终生成结构：")
        print(json.dumps(final_code, indent=2, ensure_ascii=False))
        print("\n运行结果：")
        print(run_log)
    except Exception as e:
        print(f"\n最终失败：{e}")

if __name__ == "__main__":
    main()

