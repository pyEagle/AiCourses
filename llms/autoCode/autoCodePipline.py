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
            self.model = SentenceTransformer('./model/LLms/sentence_transformers/', device=self.device)
        except:
            print("⚠️ 未找到本地模型，自动下载默认模型 'all-MiniLM-L6-v2'...")
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
    def __init__(self, task, custom_prompt_template=None):
        self.task = task
        self.custom_prompt_template = custom_prompt_template
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
            return txt
        except Exception as e:
            print(f"LLM请求失败：{e}")
            return ""

    def _get_context(self, log):
        if not log: return "领域代码|状态_初始"
        if "JSON" in log.upper() or "解析" in log: return "领域代码|状态_格式错误"
        elif "SYNTAX" in log.upper() or "语法" in log: return "领域代码|状态_语法错误"
        else: return "领域代码|状态_运行失败"

    def run(self, iterations=5, max_fixes=3):
        # 如果有传入自定义系统级模板，则使用；否则使用默认单任务模板
        current_prompt = self.custom_prompt_template or f"""你是专业Python工程师，任务：{self.task}
严格只输出标准JSON，无任何多余内容
格式：{{"path":"./","files":[{{"code":"完整可运行代码","pyfile":"main.py"}}],"main":"python main.py"}}
要求：代码必须通过Python语法检查，逻辑正确，只使用安全操作"""

        last_log = ""
        final_spec = None

        for i in range(iterations):
            ctx = self._get_context(last_log)
            arm_id = self.bandit.add_or_update_arm(current_prompt)
            strategy_prompt = self.bandit.sample(ctx, fallback=current_prompt)

            print(f"\n--- [第 {i+1} 轮迭代] Arm ID: {arm_id} ---")
            active_prompt = strategy_prompt
            success = False

            for fix_step in range(max_fixes + 1):
                if fix_step > 0: print(f"🛠️  第{fix_step}次自动修复中...")

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
                    print(f"📊 运行结果：\n{last_log.strip()}")
                    return spec  # 成功，返回解析好的 JSON 格式化代码规范

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
        
        return None

class SSEOrchestrator:
    def __init__(self, task):
        self.task = task
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
            r = requests.post(self.api_url, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            print(f"LLM架构设计请求失败：{e}")
            return "{}"

    def analyze_and_decompose(self):
        """阶段1：概念模型拆解"""
        print(f"\n🧠 [SSE 阶段 1] 开始软系统工程概念建模与拆解...")
        prompt = f"""作为系统架构师，请使用软系统工程(SSE)方法拆解以下任务：
任务：{self.task}
请将此任务拆解为 2-4 个简单、高内聚低耦合的Python子函数，并规划一个串联它们的主流程。
严格只输出标准JSON，格式如下（不要输出其他文本）：
{{
    "modules": [
        {{"name": "函数名1", "desc": "子任务1详细描述", "inputs": "输入说明", "outputs": "输出说明"}},
        {{"name": "函数名2", "desc": "子任务2详细描述", "inputs": "输入说明", "outputs": "输出说明"}}
    ],
    "main_flow": "描述主函数 main() 如何按顺序调用并传递上述模块的数据，最后打印结果。"
}}"""
        res = self._llm_query(prompt, is_json=True, temp=0.3)
        try:
            plan = json.loads(res)
            modules_count = len(plan.get('modules', []))
            print(f"✅ 系统拆解完成：成功划分为 {modules_count} 个子模块。")
            return plan
        except Exception as e:
            print(f"❌ 拆解失败 ({e})，启用降级方案...")
            return {
                "modules": [{"name": "process_task", "desc": self.task, "inputs": "None", "outputs": "打印结果"}],
                "main_flow": "调用 process_task()"
            }

    def assemble_and_execute(self, plan):
        """阶段2：蓝图装配与执行"""
        print(f"⚙️  [SSE 阶段 2] 开始根据蓝图装配代码，并注入 Bandit 智能体运行...")
        
        modules_json_str = json.dumps(plan['modules'], ensure_ascii=False, indent=2)
        
        # 将 SSE 设计图注入为智能体的全局指令
        sse_prompt_template = f"""你是专业Python架构师与开发工程师。已完成软系统工程拆解，请根据以下【系统设计蓝图】生成完整可运行代码：

目标任务：{self.task}
【子模块定义】：
{modules_json_str}
【主流程设计】：
{plan['main_flow']}

严格只输出标准JSON，无任何多余文字或Markdown标记。格式：
{{"path":"./","files":[{{"code":"包含所有子函数、主函数 main()，以及 if __name__ == '__main__': main() 的完整代码","pyfile":"main.py"}}],"main":"python main.py"}}
要求：
1. 代码必须完全遵循上述【系统设计蓝图】拆解的模块。
2. 逻辑严密，自带错误处理。
3. 纯净代码，禁止高危系统调用。"""

        # 调用带 Bandit 记忆的 ResearchAgent 执行装配与自我修复
        agent = ResearchAgent(task=self.task, custom_prompt_template=sse_prompt_template)
        final_spec = agent.run(iterations=4, max_fixes=3)

        if final_spec:
            print("\n🎉 软系统工程(SSE)任务圆满完成！最终生成的装配代码如下：")
            print("="*50)
            print(final_spec['files'][0]['code'])
            print("="*50)
        else:
            print("\n💥 系统最终装配失败，未能通过沙盒安全测试。请检查日志或优化任务描述。")


if __name__ == "__main__":
    if os.path.exists("bandit_memory.json"):
        os.remove("bandit_memory.json")
        
    complex_task = "生成100个随机正整数(1-1000)，过滤出其中的质数，计算这些质数的平均值和方差，最后打印结果报告。"
    
    orchestrator = SSEOrchestrator(complex_task)
    
    system_plan = orchestrator.analyze_and_decompose()
    
    orchestrator.assemble_and_execute(system_plan)

