import requests

MCP_SERVER_URL = "http://127.0.0.1:5000"


class MCPClient:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id  # MCP会话标识

    def chat(self, user_input):
        url = f"{MCP_SERVER_URL}/mcp/chat"
        # MCP协议规定的请求格式
        payload = {
            "session_id": self.session_id,
            "user_input": user_input
        }

        try:
            response = requests.post(url, json=payload)
            result = response.json()

            if result["code"] == 200:
                return f"\n🤖 模型回答：{result['data']['response']}"
            else:
                return f"❌ MCP错误：{result['msg']}"
        except Exception as e:
            return f"❌ 连接MCP服务失败：{str(e)}"

    def clear_session(self):
        url = f"{MCP_SERVER_URL}/mcp/session/clear"
        payload = {"session_id": self.session_id}
        requests.post(url, json=payload)
        print(f"\� MCP会话 {self.session_id} 已清空")

