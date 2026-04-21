# -*- coding:utf-8 -*-

import uuid
from core.agent import Agent


def main():
    agent = Agent()
    
    session_id = str(uuid.uuid4())
    print("=== 智能体对话系统 ===")
    print(f"会话ID：{session_id}")
    print("输入 'exit' 退出对话\n")
    
    while True:
        # 获取用户输入
        query = input("你：")
        if query.lower() == "exit":
            print("智能体：再见！")
            break
        
        # 调用智能体
        try:
            answer = agent.chat(session_id, query)
            print(f"智能体：{answer}\n")
        except Exception as e:
            print(f"智能体：抱歉，处理你的问题时出错了：{str(e)}\n")

if __name__ == "__main__":
    main()

