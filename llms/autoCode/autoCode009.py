# -*- coding:utf-8 -*-

import os
import json
import time
import hashlib
import traceback
import subprocess
import random
import ast

import torch
import requests

from sentence_transformers import SentenceTransformer


class SemanticBanditCore:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.memory_path = "bandit_memory.json"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            './model/LLms/sentence_transformers/',
            device=self.device
        )

        self.arms = []
        self.decay = 0.995

        self._load()

    def _get_feature_vector(self, text):
        vec = self.model.encode(text, convert_to_tensor=True)

        return vec.to(self.device)

    def _calculate_similarity(self, vec1, vec2):
        sim = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
        return sim.item()

    def get_task_cluster(self, task):
        task = task.lower()
        if "flask" in task or "fastapi" in task or "api" in task:
            return "python_api"

        if "爬虫" in task or "crawler" in task:
            return "crawler"

        if "机器学习" in task or "pytorch" in task:
            return "ml"

        return "general"

    def add_arm(self, prompt_template, cluster, arm_type="base"):
        arm_id = hashlib.md5(
            prompt_template.encode()
        ).hexdigest()[:8]
        vec = self._get_feature_vector(prompt_template)
        arm = {
            "id": arm_id,
            "cluster": cluster,
            "type": arm_type,
            "prompt_template": prompt_template,
            "vec": vec,
            "score": 0.5,
            "success": {},
            "failure": {},
            "created_at": time.time()
        }

        self.arms.append(arm)
        self._prune()
        self._save()

        return arm_id

    def retrieve_similar_arms(self,prompt, cluster, topk=5):
        if not self.arms:
            return []

        query_vec = self._get_feature_vector(prompt)
        scored = []
        for arm in self.arms:
            if arm["cluster"] != cluster:
                continue

            sim = self._calculate_similarity(query_vec, arm["vec"])
            scored.append((sim, arm))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        for sim, arm in scored[:topk]:
            if sim > 0.65:
                result.append(arm)

        return result

    def sample(self,context_key, candidates, fallback_prompt):
        if not candidates:
            return {"id": None, "prompt_template": fallback_prompt}

        alpha_list = []
        beta_list = []
        for arm in candidates:
            suc = arm["success"].get(context_key, 1.1)
            fail = arm["failure"].get(context_key, 1.1)

            alpha_list.append(suc)
            beta_list.append(fail)

        alpha_tensor = torch.tensor(alpha_list, device=self.device)
        beta_tensor = torch.tensor(beta_list, device=self.device)

        dist = torch.distributions.Beta(alpha_tensor, beta_tensor)
        samples = dist.sample()
        idx = torch.argmax(samples).item()
        return candidates[idx]

    def update_stats(self, arm_id, context_key, reward):
        if arm_id is None:
            return

        for arm in self.arms:
            if arm["id"] == arm_id:
                if context_key not in arm["success"]:
                    arm["success"][context_key] = 1.1
                if context_key not in arm["failure"]:
                    arm["failure"][context_key] = 1.1
                arm["success"][context_key] += reward
                arm["failure"][context_key] += (1.0 - reward) * 0.2
                arm["score"] = 0.95 * arm["score"] + 0.05 * reward
                break

        self._apply_decay()
        self._save()

    def _apply_decay(self):
        for arm in self.arms:
            for ctx in arm["success"]:
                arm["success"][ctx] *= self.decay

            for ctx in arm["failure"]:
                arm["failure"][ctx] *= self.decay

    def _prune(self):
        if len(self.arms) <= self.capacity:
            return

        self.arms.sort(key=lambda x: x["score"],reverse=True)
        self.arms = self.arms[:self.capacity]

    def _save(self):
        save_data = []
        for arm in self.arms:
            temp = arm.copy()
            if torch.is_tensor(temp["vec"]):
                temp["vec"] = temp["vec"].cpu().tolist()

            save_data.append(temp)

        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

    def _load(self):
        if not os.path.exists(self.memory_path):
            return

        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for arm in data:
                arm["vec"] = torch.tensor(arm["vec"], device=self.device)

            self.arms = data
            print(f"[Bandit] loaded={len(self.arms)}")
        except:
            traceback.print_exc()

class SecureRunner:
    def __init__(self, root="./sandbox"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.blacklist = [
            "os.system",
            "subprocess.Popen",
            "subprocess.call",
            "pty.spawn",
            "shutil.rmtree",
            "rm -rf",
            "__import__",
            "eval(",
            "exec("
        ]

    def _ast_check(self, code):
        try:
            tree = ast.parse(code)
            banned_nodes = ast.ImportFrom
            for node in ast.walk(tree):
                if isinstance(node, banned_nodes):
                    if getattr(node, "module", "") == "os":
                        return False

            return True
        except:
            return False

    def _safe_check(self, code):
        lower = code.lower()
        for bad in self.blacklist:
            if bad.lower() in lower:
                return False

        return self._ast_check(code)

    def run(self, spec):
        try:
            if not all(k in spec for k in ["path", "files", "main"]):
                return False, "JSON字段错误"

            for file_info in spec["files"]:
                if ("code" not in file_info or "pyfile" not in file_info):
                    return False, "文件字段缺失"

                code = file_info["code"]
                if not self._safe_check(code):
                    return False, "检测到危险代码"

                file_path = os.path.join(self.root, file_info["pyfile"])

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)

            result = subprocess.run(
                spec["main"],
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode == 0:
                return True, result.stdout
            return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, str(e)


class ResearchAgent:
    def __init__(self, task):
        self.task = task
        self.bandit = SemanticBanditCore()
        self.runner = SecureRunner()
        self.api_url = "http://localhost:11434/api/generate"

    def _llm_query(self,prompt, is_json=True, temp=0.2):
        payload = {
            "model": "deepseek-r1:latest",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp
            }
        }
        if is_json:
            payload["format"] = "json"

        try:
            r = requests.post(self.api_url, json=payload, timeout=120)
            r.raise_for_status()
            txt = r.json().get("response", "")
            return txt
        except Exception as e:
            print(f"[LLM ERROR] {e}")

            return ""

    def _get_error_type(self, log):
        upper = log.upper()
        if "JSON" in upper:
            return "json_error"
        if "SYNTAX" in upper:
            return "syntax_error"
        if "TIMEOUT" in upper:
            return "timeout"
        if "危险" in log:
            return "unsafe"
        return "runtime_error"

    def _build_context(self, last_log):
        cluster = self.bandit.get_task_cluster(self.task)
        if not last_log:
            return f"{cluster}_initial"

        err = self._get_error_type(last_log)
        return f"{cluster}_{err}"

    def _compute_reward(self,success, log):
        if success:
            return 1.0

        upper = log.upper()
        if "JSON" in upper:
            return 0.2
        if "SYNTAX" in upper:
            return 0.4
        if "TIMEOUT" in upper:
            return 0.3
        if "危险" in log:
            return 0.1

        return 0.5

    def _build_base_prompt(self):
        return f"""
你是专业Python工程师。
任务：
{self.task}
严格只输出标准JSON。
格式：
{{
    "path":"./",
    "files":[
        {{
            "code":"完整Python代码",
            "pyfile":"main.py"
        }}
    ],
    "main":"python main.py"
}}

要求：
1. 必须完整可运行
2. 禁止markdown
3. 必须包含完整import
4. 禁止危险操作
5. 必须通过语法检查
"""

    def _build_repair_prompt(self, last_log, failed_code):
        return f"""
任务：
{self.task}
之前代码失败。
错误信息：
{last_log}
错误代码：
{failed_code}
请修复问题。
严格要求：
1. 只输出JSON
2. 不允许markdown
3. 必须完整可运行
4. 必须包含完整import
5. 禁止危险操作
"""

    def run(self, iterations=5, max_fixes=3):
        base_prompt = self._build_base_prompt()
        last_log = ""
        cluster = self.bandit.get_task_cluster(self.task)
        for round_id in range(iterations):
            print(f"\n[迭代: {round_id+1}]")
            context = self._build_context(last_log)
            candidates = self.bandit.retrieve_similar_arms(base_prompt, cluster, topk=5)

            explore = random.random() < 0.15
            if explore or not candidates:
                arm_id = self.bandit.add_arm(base_prompt, cluster, arm_type="base")
                active_prompt = base_prompt
                print("[Bandit] new arm")
            else:
                chosen = self.bandit.sample(context, candidates, base_prompt)
                arm_id = chosen["id"]
                active_prompt = chosen["prompt_template"]
                print(f"[Bandit] reuse={arm_id}")

            success = False
            spec = None
            for fix_step in range(max_fixes + 1):
                print(f"[Fix {fix_step}]")
                raw = self._llm_query(active_prompt)

                if not raw:
                    last_log = "LLM_EMPTY"
                    continue
                try:
                    spec = json.loads(raw)
                except Exception as e:
                    success = False
                    last_log = f"JSON_ERROR:{e}"
                    continue

                success, last_log = self.runner.run(spec)
                print(last_log)
                if success:
                    reward = self._compute_reward(True, last_log)
                    self.bandit.update_stats(arm_id, context, reward)
                    print(f"[SUCCESS] reward={reward}")
                    return spec

                failed_code = ""
                try:
                    failed_code = spec["files"][0]["code"]
                except:
                    failed_code = "NO_CODE"

                repair_prompt = self._build_repair_prompt(last_log, failed_code)

                if random.random() < 0.3:
                    self.bandit.add_arm(repair_prompt, cluster, arm_type="repair")
                active_prompt = repair_prompt

            reward = self._compute_reward(False, last_log)

            self.bandit.update_stats(arm_id, context, reward)
            print(f"[FAILED] reward={reward}")

        return None



if __name__ == "__main__":
    agent = ResearchAgent(
        "编写一个Python脚本，找出1000以内最大的质数，并打印结果"
    )
    result = agent.run()
    print("\nFINAL RESULT:\n")
    print(result)
