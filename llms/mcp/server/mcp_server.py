from flask import Flask, request, jsonify
from server.ollama_client import chat_with_ollama

app = Flask(__name__)

mcp_sessions = {}

@app.route('/mcp/chat', methods=['POST'])
def mcp_chat():
    data = request.get_json()
    session_id = data.get("session_id", "default")  # 会话ID（MCP标识）
    user_input = data.get("user_input", "")

    if not user_input:
        return jsonify({
            "code": 400,
            "msg": "MCP错误：用户输入不能为空",
            "data": None
        })

    if session_id not in mcp_sessions:
        mcp_sessions[session_id] = []
    session_history = mcp_sessions[session_id]

    model_response = chat_with_ollama(user_input, session_history)

    session_history.append({"role": "user", "content": user_input})
    session_history.append({"role": "assistant", "content": model_response})

    return jsonify({
        "code": 200,
        "msg": "MCP请求成功",
        "data": {
            "session_id": session_id,
            "response": model_response,
            "context_length": len(session_history)  # 上下文长度（MCP元数据）
        }
    })


@app.route('/mcp/session/clear', methods=['POST'])
def mcp_clear_session():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    if session_id in mcp_sessions:
        mcp_sessions[session_id] = []
    return jsonify({"code": 200, "msg": "MCP会话已清空", "data": {"session_id": session_id}})


def run_mcp_server(host="0.0.0.0", port=5000):
    app.run(host=host, port=port, debug=True)

