import sys
from client.mcp_client import MCPClient


def client_chat():
    client = MCPClient(session_id="test_001")
    print("\nMCP客户端已启动（会话ID：test_001），输入 exit 退出")

    while True:
        user_input = input("\n🗣️ 请输入问题：")
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            client.clear_session()
            continue
        print(client.chat(user_input))


if __name__ == '__main__':
    client_chat()
