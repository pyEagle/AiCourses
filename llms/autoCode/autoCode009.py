# -*- coding:utf-8 -*-
import os
import re
import json
import hashlib
import traceback
import subprocess
import requests
import torch
from sentence_transformers import SentenceTransformer

class SemanticBanditCore:
    def __init__(self, capacity=20):
        self.arms = []
        self.capacity = capacity
        self.memory_path = "bandit_memory.json"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [系统] 初始化设备: {self.device}")
        try:
            self.model = SentenceTransformer('/usr/songzs/model/LLms/sentence_transformers/', device=self.device)
        except:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self._load()

    def _get_feature_vector(self, text):
        return self.model.encode(text, convert_to_tensor=True).to(self.device)

    def _calculate_similarity(self, vec1, vec2):
        sim = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
        return sim.item()

    def add_or_update_arm(self, prompt):
        if not prompt.strip():
            prompt = "生成可运行Python代码"
        new_vec = self._get_feature_vector(prompt)
        best_sim, best_arm = 0.0, None
        for arm in self.arms:
            sim = self._calculate_similarity(new_vec, arm["vec"])
            if sim > best_sim:
                best_sim, best_arm = sim, arm
        if best_sim > 0.85:
            return best_arm["id"]
        arm_id = hashlib.md5(prompt.encode()).hexdigest()[:8]
        self.arms.append({"id": arm_id, "prompt": prompt, "vec": new_vec, "history": {}, "score": 0.5})
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
        dist = torch.distributions.Beta(torch.tensor(alphas, device=self.device), torch.tensor(betas, device=self.device))
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
            if torch.is_tensor(temp["vec"]):
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
        self.not_safe_code_list = [
            "__import__('os').system", "subprocess.call", "pty.spawn",
            "rm -rf", "os.system", "shutil.rmtree", "os.popen", "eval", "exec"
        ]

    def run(self, spec):
        try:
            if not all(k in spec for k in ["path", "files", "main"]):
                return False, "JSON格式错误"
            for file_info in spec["files"]:
                if "code" not in file_info or "pyfile" not in file_info:
                    return False, "文件字段缺失"
                code = file_info["code"]
                if any(bad in code for bad in self.not_safe_code_list):
                    return False, "安全风险：包含高危系统操作"
                file_path = os.path.join(self.root, file_info["pyfile"])
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
            res = subprocess.run(
                spec["main"], shell=True, cwd=self.root,
                capture_output=True, text=True, timeout=15
            )
            return (res.returncode == 0), res.stdout if res.returncode == 0 else res.stderr
        except Exception as e:
            return False, f"执行异常：{str(e)}"

class ResearchAgent:
    def __init__(self, task):
        self.task = task
        self.bandit = SemanticBanditCore()
        self.runner = SecureRunner()
        self.api_url = "http://localhost:11434/api/generate"

    def _llm_query(self, prompt, is_json=True, temp=0.2):
        payload = {
            "model": "deepseek-r1:latest",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temp}
        }
        if is_json:
            payload["format"] = "json"
        try:
            r = requests.post(self.api_url, json=payload, timeout=120)
            r.raise_for_status()
            txt = r.json().get("response", "")
            print(txt)
            return txt
        except Exception as e:
            print(f"LLM请求失败：{e}")
            return ""

    def _get_context(self, log):
        if not log:
            return "领域代码|状态_初始"
        if "JSON" in log.upper() or "解析" in log:
            err = "格式错误"
        elif "SYNTAX" in log.upper() or "语法" in log:
            err = "语法错误"
        else:
            err = "运行失败"
        return f"领域代码|状态_{err}"

    def run(self, iterations=5, max_fixes=3):
        # 🔥 修复：使用双层大括号转义，解决 f-string 冲突
        current_prompt = f"""你是专业Python工程师，任务：{self.task}
严格只输出标准JSON，无任何多余内容
格式：{{"path":"./","files":[{{"code":"完整可运行代码","pyfile":"main.py"}}],"main":"python main.py"}}
要求：代码必须通过Python语法检查，逻辑正确，只使用安全操作"""

        last_log = ""

        for i in range(iterations):
            ctx = self._get_context(last_log)
            arm_id = self.bandit.add_or_update_arm(current_prompt)
            strategy_prompt = self.bandit.sample(ctx, fallback=current_prompt)

            print(f"\n--- [第 {i+1} 轮迭代] Arm ID: {arm_id} ---")
            active_prompt = strategy_prompt
            success = False

            for fix_step in range(max_fixes + 1):
                if fix_step > 0:
                    print(f"🛠️  第{fix_step}次自动修复中...")

                raw_res = self._llm_query(active_prompt)
                if not raw_res:
                    success, last_log = False, "LLM未返回数据"
                    continue

                try:
                    spec = json.loads(raw_res)
                    success, last_log = self.runner.run(spec)
                except Exception as e:
                    success, last_log = False, f"JSON解析失败: {str(e)[:50]}"
                    spec = None

                if success:
                    print(f"✅ 执行成功！")
                    self.bandit.update_stats(arm_id, ctx, 1.0)
                    print(f"📊 运行结果：\n{last_log}")
                    return spec

                failed_code = spec["files"][0]["code"] if (spec and "files" in spec) else "无有效代码"
                active_prompt = f"""任务：{self.task}
错误信息：{last_log}
错误代码：{failed_code}
请直接输出修复后的完整标准JSON，不要任何解释文字
要求：代码必须正确可运行，仅使用基础Python语法，无高危操作"""

            print(f"❌ 本轮策略失败，正在进化提示词...")
            self.bandit.update_stats(arm_id, ctx, 0.0)
            evo_prompt = f"任务：{self.task}，错误：{last_log}，生成一个能让代码一次运行成功的严格提示词，只输出提示词本身"
            current_prompt = self._llm_query(evo_prompt, is_json=False, temp=0.1) or current_prompt

if __name__ == "__main__":
    if os.path.exists("bandit_memory.json"):
        os.remove("bandit_memory.json")
        
    agent = ResearchAgent("编写一个Python脚本，找出1000以内最大的质数，代码直接打印最终结果")
    agent.run()

