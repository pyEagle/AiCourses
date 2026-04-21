import sys
from server.mcp_server import run_mcp_server


if __name__ == '__main__':
    print("正在启动 MCP 服务端（http://0.0.0.0:5000）...")
    run_mcp_server()
