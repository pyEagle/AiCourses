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
            except:
                traceback.print_exc()

class SecureRunner:
    def __init__(self, root="./sandbox"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.not_safe_code_list = ["__import__('os').system", "subprocess.call", "pty.spawn", "rm -rf"]

    def run(self, spec):
        try:
            for file_info in spec["files"]:
                code = file_info["code"]
                if any(x in code for x in self.not_safe_code_list):
                    return False, "安全风险：检测到系统级调用屏蔽词。"
                
                with open(os.path.join(self.root, file_info["pyfile"]), "w", encoding="utf-8") as f:
                    f.write(code)

            res = subprocess.run(
                spec["main"], shell=True, cwd=self.root,
                capture_output=True, text=True, timeout=10
            )
            return (res.returncode == 0), (res.stdout if res.returncode == 0 else res.stderr)
        except Exception as e:
            return False, f"运行时错误: {str(e)}"

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
            r = requests.post(self.api_url, json=payload, timeout=90)
            txt = r.json().get("response", "")
            return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
        except: return ""

    def _get_context(self, log):
        if not log: return "领域代码|错误_无"
        log_upper = log.upper()
        if "JSON" in log_upper: err = "格式错误"
        elif "SYNTAX" in log_upper: err = "语法错误"
        elif "TIMEOUT" in log_upper: err = "执行超时"
        else: err = "逻辑错误"
        return f"领域代码|错误_{err}"

    def run(self, iterations=5):
        current_prompt = (
            f"角色：资深 Python 开发工程师\n"
            f"当前任务：{self.task}\n"
            f"输出要求：请直接输出一个合法的 JSON 对象，不要包含任何 Markdown 代码块格式。格式如下：\n"
            f"{{\"path\":\"./\", \"files\":[{{\"code\":\"Python代码字符串\", \"pyfile\":\"文件名.py\"}}], \"main\":\"运行主文件的shell命令\"}}\n"
            f"核心原则：代码必须逻辑完整、自包含，确保执行 main 中的命令能直接得出结果。"
        )
        last_log = ""
        last_failed_code = ""

        for i in range(iterations):
            ctx = self._get_context(last_log)
            arm_id = self.bandit.add_or_update_arm(current_prompt)
            selected_prompt = self.bandit.sample(ctx, fallback=current_prompt)
            
            print(f"\n--- [第 {i+1} 轮迭代] Arm ID: {arm_id} ---")
            raw_res = self._llm_query(selected_prompt)
            
            spec = {}
            try:
                spec = json.loads(raw_res)
                success, last_log = self.runner.run(spec)
                last_failed_code = spec["files"][0]["code"] if spec.get("files") else ""
            except Exception:
                success, last_log = False, f"JSON解析失败。收到内容前100字: {raw_res[:100]}..."
                last_failed_code = "无法获取代码（JSON解析错误）"

            reward = 1.0 if success else (0.3 if "JSON解析失败" not in last_log else 0.0)
            self.bandit.update_stats(arm_id, ctx, reward)
            print(f"状态: {'✅ 运行成功' if success else '❌ 运行失败'} | 奖励得分: {reward:.1f}")

            if success:
                print(f"\n✨ 任务圆满完成！\n程序输出结果：\n{last_log}")
                return spec

            print(f"💡 正在基于错误信息进化提示词...")
            evo_prompt = (
                f"### 诊断与进化任务\n"
                f"上一次尝试解决任务“{self.task}”失败了。\n\n"
                f"### 失败的代码片段\n```python\n{last_failed_code}\n```\n\n"
                f"### 错误反馈信息\n{last_log}\n\n"
                f"### 提示词工程师指令\n"
                f"你现在的身份是顶级提示词工程师。请完成以下步骤：\n"
                f"1. 深刻分析代码失败的原因（是库缺失、逻辑漏洞、语法错误还是环境问题？）。\n"
                f"2. 编写一段全新的、更具指导性的系统提示词（System Prompt），通过在指令中加入预警或详细规范，引导 AI 避开上述错误。\n"
                f"3. 新提示词必须依然要求 AI 输出严格的 JSON 格式。\n"
                f"4. 注意：请【仅输出】新的提示词文本内容，不要包含任何分析过程或开场白。"
            )
            current_prompt = self._llm_query(evo_prompt, is_json=False, temp=0.7)

if __name__ == "__main__":
    agent = ResearchAgent("编写一个 Python 脚本，找出 1000 以内最大的质数。")
    agent.run()

