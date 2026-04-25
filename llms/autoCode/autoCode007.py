# -*- coding:utf-8 -*-

import json
import re

from autoCode007 import ResearchAgent


class PipelineAgent:
    def __init__(self, main_task):
        self.main_task = main_task
        self.orchestrator = ResearchAgent(main_task)
        self.nodes = []
        self.implemented_modules = []

    def decompose_to_flow(self):
        print("\n[Step 1] 正在生成流程图节点...")
        prompt = (
            f"请将任务：'{self.main_task}' 拆解为多个功能节点。\n"
            "输出严格 JSON 数组，每个元素必须包含 id, name, description。\n"
        )
        res = self.orchestrator._llm_query(prompt, is_json=True)

        if res:
            res = re.sub(r'^```json|```$', '', res).strip()
            data = json.loads(res)

            if isinstance(data, dict):
                self.nodes = [data]
            elif isinstance(data, list):
                self.nodes = data

            for node in self.nodes:
                for k in ["id", "name", "description"]:
                    node.setdefault(k, "unknown")

        return self.nodes

    def describe_nodes(self):
        print("\n[Step 2] 正在细化节点 I/O 契约...")
        for node in self.nodes:
            prompt = (
                f"针对节点：{node['name']} ({node['description']})\n"
                "请输出JSON，必须包含 input, output, logic, test_case 四个字段。"
            )
            try:
                detail = self.orchestrator._llm_query(prompt, is_json=True)
                detail = re.sub(r'^```json|```$', '', detail).strip()
                node.update(json.loads(detail))
            except:
                pass
        return self.nodes

    def generate_node_code(self):
        print("\n[Step 3] 生成各节点代码...")

        for node in self.nodes:
            node_task = (
                f"【全局任务】{self.main_task}\n"
                f"【当前模块】{node.get('name')}\n"
                f"描述：{node.get('description')}\n"
                f"输入：{node.get('input')}\n"
                f"输出：{node.get('output')}\n"
                f"逻辑：{node.get('logic')}\n"
                "要求：函数形式实现，不写main函数"
            )

            print(f"\n>>> 节点: {node.get('id')}")
            agent = ResearchAgent(node_task)

            try:
                spec = agent.run(iterations=3)
                if spec:
                    self.implemented_modules.append(spec)
            except Exception as e:
                print(f"⚠️ 失败: {e}")

    def build_master_node(self):
        print("\n[Step 3.5] 构建主控模块...")

        modules_desc = "\n".join([
            f"{n.get('name')}：{n.get('description')}"
            for n in self.nodes
        ])

        prompt = f"""
全局任务：{self.main_task}

已有模块：
{modules_desc}

请生成 main.py 串联所有模块并输出最终结果
"""

        agent = ResearchAgent(prompt)
        spec = agent.run(iterations=3)
        if spec:
            self.implemented_modules.append(spec)

    def assemble(self):
        print("\n[Step 4] 系统组装...")

        file_map = {}

        for spec in self.implemented_modules:
            if isinstance(spec, dict) and "files" in spec:
                for file in spec["files"]:
                    file_map[file["pyfile"]] = file["code"]

        with open("final_system_integrated.py", "w", encoding="utf-8") as f:
            for fname, code in file_map.items():
                f.write(f"\n# ===== {fname} =====\n{code}\n")

        print("✅ 已生成 final_system_integrated.py")

    def run_pipeline(self):
        self.decompose_to_flow()
        self.describe_nodes()
        self.generate_node_code()
        self.build_master_node()
        self.assemble()


if __name__ == "__main__":
    task = "编写一个本地文本分析系统：读取txt，清洗非中文，统计词频Top10，输出JSON"
    PipelineAgent(task).run_pipeline()
