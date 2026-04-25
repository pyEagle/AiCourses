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
            "例如：[{\"id\":\"load\",\"name\":\"加载文件\",\"description\":\"读取txt内容\"}]"
        )
        res = self.orchestrator._llm_query(prompt, is_json=True)

        if res:
            res = re.sub(r'^```json|```$', '', res).strip()
            try:
                data = json.loads(res)

                # 自动兼容 单个字典 / 数组
                if isinstance(data, dict):
                    self.nodes = [data]
                elif isinstance(data, list):
                    self.nodes = data
                else:
                    raise ValueError("返回格式无效")

                # 字段校验
                for node in self.nodes:
                    for k in ["id", "name", "description"]:
                        node.setdefault(k, "unknown")

            except Exception as e:
                raise ValueError(f"解析失败：{e}\n返回：{res}")
        return self.nodes

    def describe_nodes(self):
        print("\n[Step 2] 正在细化节点 I/O 契约...")
        refined_nodes = []
        for node in self.nodes:
            prompt = (
                f"针对节点：{node['name']} ({node['description']})\n"
                "请输出JSON，必须包含 input, output, logic, test_case 四个字段。"
            )
            try:
                detail = self.orchestrator._llm_query(prompt, is_json=True)
                if detail:
                    detail = re.sub(r'^```json|```$', '', detail).strip()
                    detail_dict = json.loads(detail)
                    node.update(detail_dict)
            except:
                # 解析失败也不崩溃，自动赋空值
                pass

            refined_nodes.append(node)
        self.nodes = refined_nodes
        return self.nodes

    def generate_node_code(self):
        print("\n[Step 3] 启动 autoCode007 语义进化生成各节点代码...")
        for node in self.nodes:
            # ✅ 关键修复：用 .get() 安全取值，不存在不会报错
            node_task = (
                f"实现功能模块：{node.get('name', '未知模块')}\n"
                f"描述：{node.get('description', '无')}\n"
                f"输入：{node.get('input', '无')}\n"
                f"输出：{node.get('output', '无')}\n"
                f"逻辑：{node.get('logic', '无')}\n"
                "输出 autoCode007 标准格式 JSON"
            )

            print(f"\n>>> 处理节点: {node.get('id', 'unknown')}")
            node_agent = ResearchAgent(node_task)
            try:
                node_spec = node_agent.run(iterations=3)
                if node_spec:
                    self.implemented_modules.append(node_spec)
            except Exception as e:
                print(f"⚠️ 节点生成失败：{e}")
        return self.implemented_modules

    def assemble(self):
        print("\n[Step 4] 正在进行全系统组装与集成...")
        all_code = ""
        for spec in self.implemented_modules:
            if isinstance(spec, dict) and "files" in spec:
                for file in spec.get("files", []):
                    all_code += f"\n# {file.get('pyfile', 'module.py')}\n{file.get('code', '')}\n"
        with open("final_system_integrated.py", "w", encoding="utf-8") as f:
            f.write(all_code)
        print("\n✨ 完成！已生成：final_system_integrated.py")

    def run_pipeline(self):
        self.decompose_to_flow()
        self.describe_nodes()
        self.generate_node_code()
        self.assemble()

if __name__ == "__main__":
    task = "编写一个本地文本分析系统：1.读取txt文件 2.清洗非中文字符 3.统计词频返回Top10 4.结果存为JSON"
    pipeline = PipelineAgent(task)
    pipeline.run_pipeline()

