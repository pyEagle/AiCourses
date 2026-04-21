# -*- coding:utf-8 -*-
 
import re
import json

import requests
 
def calculate(expression):
    expression = re.sub(r'[^0-9+\-*/(). ]', '', expression)
    try:
        return eval(expression)
    except Exception as e:
        return f"错误: {str(e)}"

SYSTEM_PROMPT = """
你是一个具备计算能力的 AI 助手。
当用户提出的问题涉及数学计算时，你必须遵循以下流程：
1. 思考：简要描述你为什么要计算。
2. 行动：输出格式为 calculate(你的表达式)，例如 calculate(123*456)。
3. 暂停：停止输出，等待系统返回计算结果。
当你得到计算结果（Observation）后，再给出最终回答（Final Answer）。
示例：
用户：15乘以24是多少？
Thought: 我需要计算 15 * 24。
Action: calculate(15*24)
（此处你会收到 Observation: 360）
Final Answer: 15乘以24的结果是360。
"""

def call_ollama(messages):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "deepseek-r1:latest",
        "messages": messages,
        "stream": False
    }
    response = requests.post(url, json=payload)

    return response.json()['message']['content']


def run_agent(user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    response = call_ollama(messages)
    print(f"AI 回复:")
    print(response)

    # 存入上下文
    messages.append({"role": "assistant", "content": response})

    # 匹配 Action: calculate(...)
    match = re.search(r'Action:\s*calculate\((.*?)\)', response)

    if match:
        expression = match.group(1)
        print(f"检测到工具调用: 计算 {expression}")

        # 执行本地函数
        result = calculate(expression)
        print(f"本地计算结果: {result}")

        # 把结果作为 Observation 反馈给 AI
        messages.append({"role": "user", "content": f"Observation: {result}"})


if __name__ == "__main__":
    run_agent("如果我有 127 个苹果，每个卖 15.5 元，再减去 250 元的摊位费，我最后剩多少钱？")

