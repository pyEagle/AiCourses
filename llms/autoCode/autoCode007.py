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
        print(f"[System] Initialized on device: {self.device}")
        
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
        self._load()

    def _get_feature_vector(self, text):
        return self.model.encode(text, convert_to_tensor=True).to(self.device)

    def _calculate_similarity(self, vec1, vec2):
        sim = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
        return sim.item()

    def add_or_update_arm(self, prompt):
        new_vec = self._get_feature_vector(prompt)
        best_sim = 0.0
        best_arm = None

        for arm in self.arms:
            sim = self._calculate_similarity(new_vec, arm["vec"])
            if sim > best_sim:
                best_sim = sim
                best_arm = arm

        if best_sim > 0.92: 
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
        if not self.arms: return fallback
        
        alphas, betas = [], []
        for arm in self.arms:
            a, b = arm["history"].get(context_key, [1.1, 1.1])
            alphas.append(a)
            betas.append(b)
        
        dist = torch.distributions.Beta(torch.tensor(alphas, device=self.device), 
                                       torch.tensor(betas, device=self.device))
        samples = dist.sample()
        return self.arms[torch.argmax(samples).item()]["prompt"]

    def update_stats(self, arm_id, context, reward):
        for arm in self.arms:
            if arm["id"] == arm_id:
                if context not in arm["history"]:
                    arm["history"][context] = [1.1, 1.1]
                arm["history"][context][0] += reward
                arm["history"][context][1] += (1.0 - reward)
                arm["score"] = 0.8 * arm["score"] + 0.2 * reward
                break
        self._save()

    def _save(self):
        save_data = []
        for a in self.arms:
            temp = a.copy()
            if torch.is_tensor(temp["vec"]):
                temp["vec"] = temp["vec"].cpu().tolist()
            save_data.append(temp)
        with open(self.memory_path, "w") as f:
            json.dump(save_data, f)

    def _load(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f:
                    data = json.load(f)
                    for a in data:
                        a["vec"] = torch.tensor(a["vec"]).to(self.device)
                    self.arms = data
            except Exception as e:
                traceback.print_exc()

class SecureRunner:
    def __init__(self, root="./sandbox"):
        self.root = root
        os.makedirs(root, exist_ok=True)

        self.not_safe_code_list = ["__import__('os').system", "subprocess.call", "pty.spawn"]

    def run(self, spec):
        try:
            for file_info in spec["files"]:
                code = file_info["code"]
                if any(x in code for x in self.not_safe_code_list):
                    return False, "安全风险: 触发系统级调用"
                
                with open(os.path.join(self.root, file_info["pyfile"]), "w", encoding="utf-8") as f:
                    f.write(code)

            res = subprocess.run(
                spec["main"], shell=True, cwd=self.root,
                capture_output=True, text=True, timeout=10
            )
            return (res.returncode == 0), (res.stdout if res.returncode == 0 else res.stderr)
        except Exception as e:
            return False, f"运行错误: {str(e)}"

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
        if is_json: payload["format"] = "json"
        
        try:
            r = requests.post(self.api_url, json=payload, timeout=60)
            txt = r.json().get("response", "")
            return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
        except: return ""

    def _get_context(self, log):
        if not log: return "domain_code|err_none"
        log_upper = log.upper()
        if "JSON" in log_upper: err = "format"
        elif "SYNTAX" in log_upper: err = "syntax"
        elif "TIMEOUT" in log_upper: err = "timeout"
        else: err = "logic"
        return f"domain_code|err_{err}"

    def run(self, iterations=5):
        current_prompt = (
            f"任务：{self.task}。\n"
            f"要求：输出严格的JSON格式：{{\"path\":\"./\", \"files\":[{{\"code\":str, \"pyfile\":str}}], \"main\":str}}。\n"
            f"注意：Python代码要包含完整的逻辑，确保直接运行 main 字段的命令可以输出结果。"
        )
        last_log = ""

        for i in range(iterations):
            ctx = self._get_context(last_log)
            arm_id = self.bandit.add_or_update_arm(current_prompt)
            selected_prompt = self.bandit.sample(ctx, fallback=current_prompt)
            
            print(f"\n[轮次 {i+1}] Device: {self.bandit.device} | Arm: {arm_id}")
            raw_res = self._llm_query(selected_prompt)
            
            try:
                spec = json.loads(raw_res)
                success, last_log = self.runner.run(spec)
            except:
                success, last_log = False, "JSON_Parse_Error"

            reward = 1.0 if success else (0.2 if "JSON_Parse_Error" not in last_log else 0.0)
            self.bandit.update_stats(arm_id, ctx, reward)
            print(f"结果: {'成功' if success else '失败'} | 奖励: {reward:.1f}")

            if success:
                print(f"\n 完成！输出：\n{last_log}")
                return spec

            # 提示词进化逻辑
            evo_prompt = (
                f"代码执行失败。任务是：{self.task}。\n"
                f"错误：\n{last_log}\n"
                f"请写一段新的系统提示词引导 AI 避开此错误并保持JSON格式输出。只输出文本。"
            )
            current_prompt = self._llm_query(evo_prompt, is_json=False, temp=0.7)

if __name__ == "__main__":
    agent = ResearchAgent("编写一个Python脚本，找出1000以内最大的质数。")
    agent.run()

