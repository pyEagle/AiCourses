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
        
        self.model = SentenceTransformer('/usr/songzs/model/LLms/sentence_transformers/', device=self.device)
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

        if best_sim > 0.75: 
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
        self.dangerous_patterns = [
            r'os\.remove\(.+\)', r'os\.unlink\(.+\)', r'shutil\.rmtree\(.+\)',
            r'subprocess\.run\(["\']rm ["\'].*\)', r'__import__\(\'os\'\)\.system',
            r'subprocess\.call', r'pty\.spawn', r'rm -rf'
        ]

    def run(self, spec):
        try:
            required_fields = ["path", "files", "main"]
            if not all(k in spec for k in required_fields):
                return False, f"JSON格式错误：缺少必需字段 {', '.join(required_fields)}"

            for file_info in spec["files"]:
                if not all(k in file_info for k in ["code", "pyfile"]):
                    return False, "文件格式错误：缺少code/pyfile字段"

                code = file_info["code"]
                for pattern in self.dangerous_patterns:
                    if re.search(pattern, code):
                        return False, f"安全风险：检测到危险操作 `{pattern}`"
                
                file_path = os.path.join(self.root, spec["path"], file_info["pyfile"])
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)

            exec_path = os.path.join(self.root, spec["path"])
            res = subprocess.run(
                spec["main"], shell=True, cwd=exec_path,
                capture_output=True, text=True, timeout=10
            )
            return (res.returncode == 0), (res.stdout if res.returncode == 0 else res.stderr)
        except Exception as e:
            return False, f"运行时错误: {str(e)} + {traceback.format_exc()[:100]}"

class ResearchAgent:
    def __init__(self, task):
        self.task = task
        self.bandit = SemanticBanditCore()
        self.runner = SecureRunner()
        self.api_url = "http://localhost:11434/api/generate"
        # 优化：复用autoCode001的正确示例，提升JSON格式准确性
        self.correct_example = '{"path": "./", "files": [{"code": "def print_str(): return \\"hello, word\\n\\"", "pyfile": "main.py"}], "main": "python main.py"}'

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
            r = requests.post(self.api_url, json=payload, timeout=90)
            r.raise_for_status()  # 优化：主动抛出HTTP错误
            txt = r.json().get("response", "").strip()
            return txt
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return ""

    def _get_context(self, log):
        # 优化：简化错误分类，提升bandit采样准确性
        if not log: return "初始生成"
        log_upper = log.upper()
        if "JSON" in log_upper: 
            return "格式错误-JSON"
        elif "安全风险" in log_upper: 
            return "格式错误-安全"
        elif "SYNTAX" in log_upper: 
            return "代码错误-语法"
        else: 
            return "代码错误-逻辑"

    def run(self, iterations=5):
        # 优化：初始prompt复用autoCode001的结构化模板，提升生成准确性
        current_prompt = f"""
角色：资深 Python 开发工程师，深耕领域20年
任务：{self.task}
输出要求：
1. 严格输出JSON格式（无任何额外内容），参考示例：{self.correct_example}
2. 代码必须完整可运行，执行`python main.py`能直接输出结果
3. JSON字段说明：
   - path：代码文件存放路径（建议用./）
   - files：数组，包含code（完整Python代码）、pyfile（文件名，建议main.py）
   - main：执行命令（建议python main.py）
"""
        
        last_log = ""
        last_failed_code = ""

        for i in range(iterations):
            # 1. Bandit选臂
            ctx = self._get_context(last_log)
            arm_id = self.bandit.add_or_update_arm(current_prompt)
            selected_prompt = self.bandit.sample(ctx, fallback=current_prompt)
            
            print(f"\n--- [第 {i+1} 轮迭代] Arm ID: {arm_id} ---")
            # 2. 调用LLM生成代码
            raw_res = self._llm_query(selected_prompt)
            if not raw_res:
                success = False
                last_log = "LLM返回空内容"
                reward = 0.0
                self.bandit.update_stats(arm_id, ctx, reward)
                print(f"状态: ❌ 运行失败 | 奖励得分: {reward:.1f}")
                print(f"失败原因: {last_log}")
                continue

            # 3. 解析JSON+运行代码
            spec = {}
            try:
                spec = json.loads(raw_res)
                success, last_log = self.runner.run(spec)
                last_failed_code = spec["files"][0]["code"] if spec.get("files") else ""
            except json.JSONDecodeError:
                success = False
                last_log = f"JSON解析失败：{raw_res[:150]}..."
                last_failed_code = "无法解析JSON"
            except Exception as e:
                success = False
                last_log = f"解析/运行异常：{str(e)}"
                last_failed_code = "解析异常"

            if success:
                reward = 1.0  # 运行成功：满分
            elif "JSON" in last_log:
                reward = 0.0  # JSON格式错：0分
            elif "安全风险" in last_log:
                reward = 0.1  # 安全风险：低奖励
            else:
                reward = 0.2  # 语法/逻辑错：低奖励
            self.bandit.update_stats(arm_id, ctx, reward)

            print(f"状态: {'✅ 运行成功' if success else '❌ 运行失败'} | 奖励得分: {reward:.1f}")
            if not success:
                print(f"失败原因: {last_log[:200]}")  # 截断过长日志

            if success:
                print(f"\n✨ 任务完成！\n程序输出：\n{last_log}")
                return spec

            print(f"💡 基于错误优化提示词...")
            evo_prompt = f"""
用户需求：{self.task}
上一次生成代码：{raw_res}
错误信息：{last_log}
正确示例：{self.correct_example}
要求：
1. 修复错误，保证代码可运行
2. 严格输出JSON格式（无额外内容）
3. 执行`python main.py`能输出正确结果
"""
            current_prompt = self._llm_query(evo_prompt, is_json=False, temp=0.7)
            if not current_prompt:
                current_prompt = f"生成Python代码实现：{self.task}，严格输出JSON格式参考 {self.correct_example}，代码可直接运行"

        raise Exception(f"❌ 达到最大迭代次数 {iterations}，无法生成有效代码")

if __name__ == "__main__":
    if os.path.exists("bandit_memory.json"):
        os.remove("bandit_memory.json")
    if os.path.exists("./sandbox"):
        import shutil
        shutil.rmtree("./sandbox", ignore_errors=True)
    
    task = "创建一个函数，打印小于100的素数。并在main.py中调用它，输出计算结果"
    agent = ResearchAgent(task)
    final_spec = agent.run()
    print("\n最终生成的JSON：")
    print(json.dumps(final_spec, indent=2, ensure_ascii=False))

