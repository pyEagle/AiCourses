# -*- coding:utf-8 -*-
 
import redis
import uuid
import json
import logging
import traceback
 
import requests
 
from typing import List, Dict, Tuple, Optional
 
 
class DialogManager:
    MAX_ROUNDS = 10  # 最大对话轮次（10轮 = 20条消息）
    DIALOG_TTL = 7 * 24 * 3600  # 7天（单位：秒）
 
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
 
    def create_dialog(self):
        dialog_id = str(uuid.uuid4())
        self.redis.setex(f"dialog:{dialog_id}", self.DIALOG_TTL, json.dumps([]))
        return dialog_id
 
    def add_message(self, dialog_id, user_msg, ai_msg):
        dialog_key = f"dialog:{dialog_id}"
        current_history = self.redis.get(dialog_key)
        if not current_history:
            raise ValueError(f"Dialog {dialog_id} not found or expired")
        
        history = json.loads(current_history)
        history.extend([user_msg, ai_msg])
        
        # 滑动窗口：如果超过最大轮次，保留最新的 N 轮
        if len(history) > 2 * self.MAX_ROUNDS:
            history = history[-2 * self.MAX_ROUNDS:]
        
        self.redis.setex(dialog_key, self.DIALOG_TTL, json.dumps(history))
        return history
 
    def get_dialog_history(self, dialog_id):
        dialog_key = f"dialog:{dialog_id}"
        if not self.redis.exists(dialog_key):
            return []
        raw_data = self.redis.get(dialog_key)
        return json.loads(raw_data) if raw_data else []
 
class ToolRegistry:
    def __init__(self):
        self.tools = {}
 
    def register(self, name, func):
        self.tools[name] = func
 
    def execute(self, tool_name, *args, **kwargs):
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not registered")
        return self.tools[tool_name](*args, **kwargs)
 
class EnterpriseAgent:
    """
    企业级智能体：感知理解 + 长短时记忆(Redis) + Ollama(DeepSeek-R1)决策 + 工具执行
    """
    def __init__(self, ollama_host='http://localhost:11434', model_name='deepseek-r1:latest', max_memory=10):
        self.ollama_url = f"{ollama_host}/api/chat"
        self.model_name = model_name
        
        self.dialog_manager = DialogManager()
        
        self.tool_registry = ToolRegistry()
        self._register_default_tools()
        
        self.memory_window = max_memory  # 传递给LLM的上下文轮次
 
    def _register_default_tools(self):
        self.tool_registry.register("calculator", self._calculator)
 
    def _calculator(self, a, b):
        try:
            return str(float(a) + float(b))
        except ValueError:
            return "Error: Invalid numbers for calculation."
 
    def _build_messages(self, history, user_input):
        messages = []
        
        system_prompt = (
            "你是一名专业的企业助理。请严格遵守以下规则："
            "1.全程使用中文回复。"
            "2.回复需简洁、专业。"
            "3.调用工具时，严格按照此格式输出：TOOL_CALL:工具名称(参数1=值1, 参数2=值2)"
            "4.可调用工具如下："
            "4.1 calculator(a, b),功能描述：完整两个数的加法运算"
        )
        messages.append({"role": "system", "content": system_prompt})
 
        start_index = max(0, len(history) - 2 * self.memory_window)
        
        for i in range(start_index, len(history), 2):
            if i + 1 < len(history):
                messages.append({"role": "user", "content": history[i]})
                messages.append({"role": "assistant", "content": history[i+1]})
 
        messages.append({"role": "user", "content": user_input})
        
        return messages
 
    def _parse_tool_call(self, response_text):
        if "TOOL_CALL:" in response_text:
            try:
                parts = response_text.split("TOOL_CALL:", 1)[1].strip()
                if '(' in parts and ')' in parts:
                    tool_name = parts.split('(')[0].strip()
                    args_str = parts.split('(')[1].rstrip(')').strip()
                    
                    args = {}
                    if args_str:
                        for kv in args_str.split(','):
                            if '=' in kv:
                                k, v = kv.split('=', 1)
                                args[k.strip()] = v.strip().strip('"').strip("'")
                    return tool_name, args
            except Exception as e:
                import traceback
                traceback.print_exc()
        
        return "", {}
 
    def generate_response(self, dialog_id, user_input):
        history = self.dialog_manager.get_dialog_history(dialog_id)
        
        messages = self._build_messages(history, user_input)
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            llm_response = result['message']['content']
        except:
            traceback.print_exc()
            return None
 
        tool_name, tool_args = self._parse_tool_call(llm_response)
        
        final_response = ""
        if tool_name:
            try:
                tool_result = self.tool_registry.execute(tool_name, **tool_args)
                final_response = f"正在为您调用 {tool_name}... 执行结果：{tool_result}。"
            except Exception as e:
                traceback.print_exc()
        else:
            final_response = llm_response.strip()
            if final_response.startswith("Assistant:"):
                final_response = final_response[len("Assistant:"):].strip()
 
        self.dialog_manager.add_message(dialog_id, user_input, final_response)
        
        return final_response
 
if __name__ == "__main__":
    print("--- 企业级智能体启动 (基于 Ollama + DeepSeek-R1) ---")
    print("输入: exit 或 quit 退出\n")
 
    try:
        agent = EnterpriseAgent()
        dialog_id = agent.dialog_manager.create_dialog()
        
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("再见！")
                    break
                
                if not user_input.strip():
                    continue
 
                response = agent.generate_response(dialog_id, user_input)
                print(f"Assistant: {response}\n")
            except Exception as e:
                traceback.print_exc()
 
    except Exception as e:
        traceback.print_exc()
