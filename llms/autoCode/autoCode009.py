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
    def __init__(self, capacity=15):
        self.arms = [] 
        self.capacity = capacity
        self.memory_path = "bandit_memory.json"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [系统] 初始化设备: {self.device}")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
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
        if best_sim > 0.92: return best_arm["id"]
        arm_id = hashlib.md5(prompt.encode()).hexdigest()[:8]
        self.arms.append({"id": arm_id, "prompt": prompt, "vec": new_vec, "history": {}, "score": 0.5})
        if len(self.arms) > self.capacity:
            self.arms.sort(key=lambda x: x["score"], reverse=True)
            self.arms.pop()
        self._save()
        return arm_id

    def sample(self, context_key, fallback):
        if not self.arms: return fallback
        alphas = [arm["history"].get(context_key, [1.1, 1.1])[0] for arm in self.arms]
        betas = [arm["history"].get(context_key, [1.1, 1.1])[1] for arm in self.arms]
        dist = torch.distributions.Beta(torch.tensor(alphas, device=self.device), torch.tensor(betas, device=self.device))
        return self.arms[torch.argmax(dist.sample()).item()]["prompt"]

    def update_stats(self, arm_id, context, reward):
        for arm in self.arms:
            if arm["id"] == arm_id:
                if context not in arm["history"]: arm["history"][context] = [1.1, 1.1]
                arm["history"][context][0] += reward
                arm["history"][context][1] += (1.0 - reward)
                arm["score"] = 0.8 * arm["score"] + 0.2 * reward
                break
        self._save()

    def _save(self):
        save_data = []
        for a in self.arms:
            temp = a.copy()
            if torch.is_tensor(temp["vec"]): temp["vec"] = temp["vec"].cpu().tolist()
            save_data.append(temp)
        with open(self.memory_path, "w") as f: json.dump(save_data, f)

    def _load(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f:
                    data = json.load(f)
                    for a in data: a["vec"] = torch.tensor(a["vec"]).to(self.device)
                    self.arms = data
            except: traceback.print_exc()

class SecureRunner:
    def __init__(self, root="./sandbox"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.not_safe_code_list = ["__import__('os').system", "subprocess.call", "pty.spawn", "rm -rf"]

    def run(self, spec):
        try:
            for file_info in spec["files"]:
                code = file_info["code"]
                if any(x in code for x in self.not_safe_code_list): return False, "安全风险：拦截到非法指令"
                with open(os.path.join(self.root, file_info["pyfile"]), "w", encoding="utf-8") as f: f.write(code)
            res = subprocess.run(spec["main"], shell=True, cwd=self.root, capture_output=True, text=True, timeout=10)
            return (res.returncode == 0), (res.stdout if res.returncode == 0 else res.stderr)
        except Exception as e: return False, str(e)

class ResearchAgent:
    def __init__(self, task):
        self.task = task
        self.bandit = SemanticBanditCore()
        self.runner = SecureRunner()
        self.api_url = "http://localhost:11434/api/generate"

    def _llm_query(self, prompt, is_json=True, temp=0.2):
        payload = {"model": "deepseek-r1:latest", "prompt": prompt, "stream": False, "options": {"temperature": temp}}
        if is_json: payload["format"] = "json"
        try:
            r = requests.post(self.api_url, json=payload, timeout=90)
            txt = r.json().get("response", "")
            return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
        except: return ""

    def _get_context(self, log):
        if not log: return "领域代码|状态_初始"
        if "JSON" in log.upper(): err = "格式错误"
        elif "SYNTAX" in log.upper(): err = "语法错误"
        else: err = "运行失败"
        return f"领域代码|状态_{err}"

    def run(self, iterations=5, max_fixes=3):
        current_prompt = (
            f"角色：资深 Python 开发工程师\n任务：{self.task}\n"
            f"要求：输出合法 JSON：{{\"path\":\"./\", \"files\":[{{\"code\":\"代码\", \"pyfile\":\"文件名.py\"}}], \"main\":\"运行命令\"}}\n"
            f"核心：代码必须完整，输出结果必须符合预期。"
        )
        last_log = ""

        for i in range(iterations):
            ctx = self._get_context(last_log)
            arm_id = self.bandit.add_or_update_arm(current_prompt)
            strategy_prompt = self.bandit.sample(ctx, fallback=current_prompt)
            
            print(f"\n--- [第 {i+1} 轮策略] Arm: {arm_id} ---")
            
            active_prompt = strategy_prompt
            success = False
            
            for fix_step in range(max_fixes + 1):
                if fix_step > 0:
                    print(f"🛠️  正在进行第 {fix_step} 次自我修正推理...")
                
                raw_res = self._llm_query(active_prompt)
                try:
                    spec = json.loads(raw_res)
                    success, last_log = self.runner.run(spec)
                except:
                    success, last_log = False, "JSON解析失败"

                if success:
                    print(f"✅ 成功完成！步骤: {fix_step}")
                    self.bandit.update_stats(arm_id, ctx, 1.0)
                    print(f"输出结果：\n{last_log}")
                    return spec
                
                failed_code = spec["files"][0]["code"] if 'spec' in locals() and 'files' in spec else "未知"
                active_prompt = (
                    f"### 任务执行失败\n任务：{self.task}\n"
                    f"### 错误代码\n```python\n{failed_code}\n```\n"
                    f"### 错误信息\n{last_log}\n"
                    f"### 指令\n请基于 DeepSeek-R1 的推理能力，分析失败原因并修复代码。请输出修复后的完整 JSON 对象。"
                )

            print(f"❌ 策略尝试失败，正在总结教训进化 Prompt...")
            self.bandit.update_stats(arm_id, ctx, 0.1)
            
            evo_prompt = (
                f"### 失败总结\n任务：{self.task}\n最终错误：{last_log}\n"
                f"你现在的角色是顶级提示词工程师。请分析由于什么策略失误导致多次修正仍失败，"
                f"并编写一个更强力的初始系统提示词来避免此类问题。仅输出提示词内容。"
            )
            current_prompt = self._llm_query(evo_prompt, is_json=False, temp=0.7)

if __name__ == "__main__":
    agent = ResearchAgent("编写一个 Python 脚本，找出 1000 以内最大的质数。")
    agent.run()

