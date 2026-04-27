import re

from core.llmClient import OllamaClient
from core.memory import RedisMemory
from core.skillManager import SkillManager
from core.toolManager import ToolManager
from core.knowledgeManager import KnowledgeManger

class Agent:
    def __init__(self):
        self.llm_client = OllamaClient()
        self.memory = RedisMemory()
        self.skill_manager = SkillManager()
        self.tool_manager = ToolManager()
        self.knowledge = KnowledgeManger()
        
        # 技能/工具调用正则（匹配格式：[[skill_name param1=value1 param2=value2]]）
        self.call_pattern = re.compile(r"\[\[(.*?)\s+(.*?)\]\]")

    def _build_system_prompt(self):
        """构建系统提示词"""
        system_prompt = """你是一名企业级智能助手。
你可使用的工具：{}
你可使用的技能：{}
请严格遵守以下规则：
1.优先检索知识库，有匹配答案则直接使用。
2.**对于所有涉及技能列表中的任务（如摘要、翻译等），必须强制使用工具调用格式，绝对禁止直接生成结果。**
3.若需使用工具或技能，按格式调用：[[skill_name param1=value1 param2=value2]]
4.无需工具/技能时，直接简洁、准确回答。
5.所有回答必须简洁、准确、基于给定信息，不编造内容。
"""
        
        # 添加技能和工具信息
        system_prompt = system_prompt.format(self.tool_manager.get_tool_prompt(),
                             self.skill_manager.get_skill_prompt())

        return system_prompt

    def _parse_call_command(self, content):
        """解析技能/工具调用命令"""
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
        res, knowledge_answer = self.knowledge.retrieve(query)
        
        system_prompt = self._build_system_prompt()
        if knowledge_answer:
            if res == 'qa':
                self.memory.save_message(session_id, "user", query)
                self.memory.save_message(session_id, "assistant", knowledge_answer)
                return knowledge_answer
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

