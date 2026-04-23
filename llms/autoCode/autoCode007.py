# -*- coding:utf-8 -*-

import json
import os
import subprocess
import requests
import re
import hashlib
import numpy as np

from collections import Counter

class SemanticBanditCore:
    def __init__(self, capacity=15):
        self.arms = [] # List of dicts
        self.capacity = capacity
        self.memory_path = "v4_bandit_memory.json"
        self._load()

    def _get_feature_vector(self, text):
        text = text.lower()
        tokens = re.findall(r"\b(json|file|recursive|search|regex|sort|count|math|handle|exception)\b", text)
        ngrams = [text[i:i+3] for i in range(len(text)-3)]
        return Counter(tokens + ngrams)

    def _calculate_similarity(self, vec1, vec2):
        all_keys = set(vec1.keys()) | set(vec2.keys())
        if not all_keys: return 0.0
        v1 = np.array([vec1.get(k, 0) for k in all_keys])
        v2 = np.array([vec2.get(k, 0) for k in all_keys])
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.dot(v1, v2) / norm if norm > 0 else 0.0

    def add_or_update_arm(self, prompt):
        new_vec = self._get_feature_vector(prompt)
        best_sim = 0
        best_arm = None

        for arm in self.arms:
            sim = self._calculate_similarity(new_vec, arm["vec"])
            if sim > best_sim:
                best_sim = sim
                best_arm = arm

        if best_sim > 0.9:
            return best_arm["id"]

        arm_id = hashlib.md5(prompt.encode()).hexdigest()[:8]
        self.arms.append({
            "id": arm_id,
            "prompt": prompt,
            "vec": new_vec,
            "history": {}, # context -> [alpha, beta]
            "score": 0.5
        })

        if len(self.arms) > self.capacity:
            self.arms.sort(key=lambda x: x["score"], reverse=True)
            self.arms.pop()
        
        self._save()
        return arm_id

    def sample(self, context_key, fallback):
        if not self.arms: return fallback
        
        samples = []
        for arm in self.arms:
            a, b = arm["history"].get(context_key, [1.1, 1.1])
            samples.append(np.random.beta(a, b))
        
        return self.arms[np.argmax(samples)]["prompt"]

    def update_stats(self, arm_id, context, reward):
        """
        [V4 升级] 严格 Bernoulli 奖励更新
        """
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
            temp.pop("vec")
            save_data.append(temp)
        with open(self.memory_path, "w") as f:
            json.dump(save_data, f)

    def _load(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f:
                    data = json.load(f)
                    for a in data:
                        a["vec"] = self._get_feature_vector(a["prompt"])
                    self.arms = data
            except: pass

class SecureRunner:
    def __init__(self, root="./v4_sandbox"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def run(self, spec):
        try:
            for file in spec["files"]:
                code = file["code"]
                if any(x in code for x in ["__import__('os').system", "subprocess.call", "pty.spawn"]):
                    return False, "Security: Banned Syscall Detected"
                
                with open(os.path.join(self.root, file["pyfile"]), "w") as f:
                    f.write(code)

            res = subprocess.run(
                spec["main"], shell=True, cwd=self.root,
                capture_output=True, text=True, timeout=8
            )
            return (res.returncode == 0), (res.stdout if res.returncode == 0 else res.stderr)
        except Exception as e:
            return False, f"Runner Error: {str(e)}"

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
            r = requests.post(self.api_url, json=payload, timeout=60)
            txt = r.json().get("response", "")
            return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
        except: return ""

    def _get_context(self, log):
        err_type = "none"
        if log:
            if "JSON" in log: err_type = "format"
            elif "Syntax" in log: err_type = "syntax"
            elif "Timeout" in log: err_type = "timeout"
            else: err_type = "logic"
        return f"domain_code|err_{err_type}"

    def solve(self, iterations=5):
        current_prompt = (
            f"Solve: {self.task}. Output JSON: {{\"path\":\"./\", \"files\":[{{\"code\":str, \"pyfile\":str}}], \"main\":str}}. "
            "Write robust Python code."
        )
        last_log = ""

        for i in range(iterations):
            ctx = self._get_context(last_log)
            arm_id = self.bandit.add_or_update_arm(current_prompt)
            
            selected_prompt = self.bandit.sample(ctx, fallback=current_prompt)
            print(f"\n[Iter {i+1}] Context: {ctx} | Selected Arm: {arm_id}")

            raw_res = self._llm_query(selected_prompt)
            try:
                spec = json.loads(raw_res)
                success, last_log = self.runner.run(spec)
            except:
                success, last_log = False, "JSON_Parse_Error"

            reward = 1.0 if success else 0.0
            if not success:
                if "JSON_Parse_Error" not in last_log: reward += 0.2
                if "SyntaxError" not in last_log: reward += 0.2
            
            self.bandit.update_stats(arm_id, ctx, reward)
            print(f"Result: {'SUCCESS' if success else 'FAIL'} | Reward: {reward:.1f}")

            if success: return spec

            evo_prompt = (
                f"Your last prompt failed. Task: {self.task}. Error: {last_log}. "
                "Design a new system prompt that is CONCISE and avoids the previous error. "
                "Constraint: No conversation, just the prompt text. Must maintain JSON output requirement."
            )
            current_prompt = self._llm_query(evo_prompt, is_json=False, temp=0.7)

if __name__ == "__main__":
    agent = ResearchAgent("Write a python script that finds the largest prime number under 1000.")
    agent.solve()

