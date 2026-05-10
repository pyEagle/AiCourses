# -*- coding:utf-8 -*-

from tools.calculator import Calculator

class ToolManager:
    def __init__(self):
        # 注册所有工具
        self.tools = {
            Calculator().name: Calculator()
        }

    def get_all_tools(self):
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]

    def call_tool(self, tool_name, **kwargs):
        if tool_name not in self.tools:
            return f"未知工具：{tool_name}"
        
        tool = self.tools[tool_name]
        return tool.call(** kwargs)

    def get_tool_prompt(self):
        tools_info = self.get_all_tools()
        prompt = "可用工具列表：\n"
        for tool in tools_info:
            prompt += f"- {tool['name']}: {tool['description']}\n"
            prompt += f"  参数：{tool['parameters']}\n"
        return prompt

