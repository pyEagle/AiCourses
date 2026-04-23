# -*- coding:utf-8 -*-

import json
import os
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
            "输出严格的 JSON 数组格式，每个元素包含：id, name, description。\n"
            "例如：[{\"id\": \"data_loader\", \"name\": \"数据加载\", \"description\": \"...\"}]"
        )
        res = self.orchestrator._llm_query(prompt, is_json=True)
        self.nodes = json.loads(res)
        return self.nodes

    def describe_nodes(self):
        print("\n[Step 2] 正在细化节点 I/O 契约...")
        refined_nodes = []
        for node in self.nodes:
            prompt = (
                f"针对节点：{node['name']} ({node['description']})\n"
                "请详细定义其：\n1. 输入参数(Input)\n2. 输出结果(Output)\n3. 核心算法逻辑(Logic)\n4. 测试用例(Test Case)\n"
                "输出 JSON 格式，包含字段：input, output, logic, test_case。"
            )
            detail = self.orchestrator._llm_query(prompt, is_json=True)
            node.update(json.loads(detail))
            refined_nodes.append(node)
        self.nodes = refined_nodes
        return self.nodes

    def generate_node_code(self):
        print("\n[Step 3] 启动 autoCode007 语义进化生成各节点代码...")
        
        for node in self.nodes:
            node_task = (
                f"实现功能模块：{node['name']}\n"
                f"逻辑描述：{node['logic']}\n"
                f"输入：{node['input']} | 输出：{node['output']}\n"
                f"必须包含测试用例：{node['test_case']}\n"
                "要求：输出符合 autoCode007 格式的 JSON，确保模块可独立运行。"
            )
            
            print(f"\n>>> 正在处理节点: {node['id']}...")
            # 为每个节点创建一个独立的老虎机实例
            node_agent = ResearchAgent(node_task)
            # 迭代 3 次以确保代码质量
            node_spec = node_agent.run(iterations=3)
            
            if node_spec:
                # 保存每个节点生成的代码信息
                self.implemented_modules.append(node_spec)
        
        return self.implemented_modules

    def assemble(self):
        """最终组装：将所有代码块合并并生成主入口"""
        print("\n[Step 4] 正在进行全系统组装与集成...")
        all_code = ""
        for spec in self.implemented_modules:
            for file in spec["files"]:
                all_code += f"\n# File: {file['pyfile']}\n{file['code']}\n"
        
        # 保存到本地文件
        with open("final_system_integrated.py", "w", encoding="utf-8") as f:
            f.write(all_code)
        
        print("\n✨ 任务完成！所有模块已集成至: final_system_integrated.py")

    def run_pipeline(self):
        self.decompose_to_flow()
        self.describe_nodes()
        self.generate_node_code()
        # 额外步骤：组装
        self.step4_assemble()

if __name__ == "__main__":
    task = "编写一个本地文本分析系统：1.读取txt文件 2.清洗非中文字符 3.统计词频并返回Top10 4.将结果存为JSON"
    
    pipeline = PipelineAgent(task)
    pipeline.run_pipeline()

