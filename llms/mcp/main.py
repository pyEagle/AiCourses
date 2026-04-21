import sys
from server.mcp_server import run_mcp_server
from client.mcp_client import MCPClient


def show_help():
    """帮助说明"""
    print("=" * 50)
    print("迷你 MCP（模型上下文协议）教学版")
    print("命令：")
    print("  server  - 启动MCP协议服务端")
    print("  client  - 启动MCP协议客户端")
    print("  clear   - 清空当前会话上下文")
    print("  exit    - 退出程序")
    print("=" * 50)


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
    show_help()

    while True:
        cmd = input("\n请输入命令（server/client）：").strip().lower()
        if cmd == "server":
            print("正在启动 MCP 服务端（http://0.0.0.0:5000）...")
            run_mcp_server()
        elif cmd == "client":
            client_chat()
        elif cmd == "exit":
            print("退出程序")
            sys.exit(0)
        else:
            print("无效命令，请重新输入")

