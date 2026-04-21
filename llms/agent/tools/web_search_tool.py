import requests

from config.settings import TOOL_CONFIG
from tools.base_tool import BaseTool

class WebSearchTool(BaseTool):
    @property
    def name(self):
        return "web_search"

    @property
    def description(self):
        return "执行网页搜索，获取最新信息"

    @property
    def parameters(self):
        return {
            "query": "搜索关键词"
        }

    def call(self, **kwargs):
        if not self.check_parameters(** kwargs):
            return "参数错误，缺少搜索关键词"
        
        query = kwargs.get("query")
        try:
            url = f"https://api.serper.dev/search"
            headers = {
                "X-API-KEY": TOOL_CONFIG.get("web_search_api", ""),
                "Content-Type": "application/json"
            }
            data = {"q": query, "num": 5}
            
            if not TOOL_CONFIG.get("web_search_api"):
                return f"【模拟搜索结果】关键词：{query}\n1. 搜索结果1\n2. 搜索结果2\n3. 搜索结果3"
            
            response = requests.post(url, json=data, headers=headers, timeout=TOOL_CONFIG["timeout"])
            if response.status_code == 200:
                results = response.json()
                snippets = [item.get("snippet", "") for item in results.get("organic", [])[:3]]
                return "\n".join([f"{i+1}. {snippet}" for i, snippet in enumerate(snippets)])
            return f"搜索失败，状态码：{response.status_code}"
        except Exception as e:
            return f"调用搜索工具失败: {str(e)}"

