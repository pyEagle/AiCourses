import os
import json
import requests
 
 
class OllamaClient:
    def __init__(self, host=None, model=None):
        self.host = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")
        self.timeout = 120
    
    def chat(self, messages, tools=None, temperature=0.3):
        url = f"{self.host}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
        
        if tools:
            payload["tools"] = tools
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # 返回统一格式
            return {
                "message": data.get("message", {}),
                "done": data.get("done", True)
            }
        except Exception as e:
            return {"error": f"Ollama请求失败：{str(e)}"}
    
    def check_health(self):
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except:
            return False
