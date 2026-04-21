import re

from core.llm_client import OllamaClient
from core.memory import RedisMemory
from core.skill_manager import SkillManager
from core.tool_manager import ToolManager
from core.knowledge_base import KnowledgeBase

class Agent:
    def __init__(self):
        self.llm_client = OllamaClient()
        self.memory = RedisMemory()
        self.skill_manager = SkillManager()
        self.tool_manager = ToolManager()
        self.knowledge_base = KnowledgeBase()
        
        self.call_pattern = re.compile(r"\[\[(.*?)\s+(.*?)\]\]")

    def _build_system_prompt(self):
        system_prompt = """你是一名企业级智能助手。
你可使用的工具：{self.tool_manager.get_tool_prompt()}
你可使用的技能：{self.skill_manager.get_skill_prompt()}
请严格遵守以下规则：
1.优先检索知识库，有匹配答案则直接使用。
2.若需使用技能，按格式调用：[[skill_name param1=value1 param2=value2]]
3.若需最新信息，调用网页搜索：[[web_search query = 关键词]]
4.无需工具/技能时，直接简洁、准确回答。
5.所有回答必须简洁、准确、基于给定信息，不编造内容。
"""
        
        system_prompt += "\n\n" + self.skill_manager.get_skill_prompt()
        system_prompt += "\n\n" + self.tool_manager.get_tool_prompt()
        
        return system_prompt

    def _parse_call_command(self, content):
        matches = self.call_pattern.findall(content)
        commands = []
        for m in matches:
            call_type, params_str = m
            params = {}
            # 解析参数（key=value格式）
            for param in params_str.split():
                if "=" in param:
                    key, value = param.split("=", 1)
                    params[key] = value
            commands.append({
                "type": call_type,
                "params": params
            })
        return commands

    def chat(self, session_id, query):
        knowledge_answer = self.knowledge_base.retrieve(query)
        
        system_prompt = self._build_system_prompt()
        if knowledge_answer:
            prompt = f"知识库信息：\n{knowledge_answer}\n\n用户问题：{query}"
        else:
            prompt = f"用户问题：{query}"
        
        history = self.memory.get_history(session_id)
        if not history:
            history.insert(0, {"role": "system", "content": system_prompt})
        
        llm_response = self.llm_client.generate(prompt, history)
        commands = self._parse_call_command(llm_response)
        if commands:
            final_answer = []
            for cmd in commands:
                call_type = cmd["type"]
                params = cmd["params"]
                
                if call_type in self.skill_manager.skills:
                    skill_result = self.skill_manager.execute_skill(call_type, **params)
                    final_answer.append(f"【{call_type}】：{skill_result}")
                elif call_type in self.tool_manager.tools:
                    tool_result = self.tool_manager.call_tool(call_type, **params)
                    final_answer.append(f"【{call_type}】：{tool_result}")
                else:
                    final_answer.append(f"未知调用类型：{call_type}")
            
            final_answer = "\n\n".join(final_answer)
        else:
            final_answer = llm_response
        
        self.memory.save_message(session_id, "user", query)
        self.memory.save_message(session_id, "assistant", final_answer)
        
        return final_answer

