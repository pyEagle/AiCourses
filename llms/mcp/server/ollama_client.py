import requests

OLLAMA_BASE_URL = "http://localhost:11434/api"
MODEL_NAME = "deepseek-r1:latest"


system_prompt = [{'role':'system', 
                  'content':"""你是一名企业级智能助手。在回答问题时，始终用中文回复，且所有回答必须简洁、准确、基于给定信息，不编造内容。\n"""}]
def chat_with_ollama(prompt, history):
    url = f"{OLLAMA_BASE_URL}/chat"
    # 拼接上下文（MCP核心：上下文管理）
    messages = system_prompt+history + [{"role": "user", "content": prompt}]
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        return f"模型调用失败：{str(e)}"

