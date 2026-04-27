import requests
import re

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
        if not self.check_parameters(**kwargs):
            return "参数错误，缺少搜索关键词"
        
        query = kwargs.get("query")
        
        # 检测是否为天气查询
        if "天气" in query:
            return self._get_weather_info(query)
        else:
            return self._perform_generic_search(query)

    def _get_weather_info(self, query):
        """获取天气信息（使用和风天气公开测试接口）"""
        # 提取城市名称
        city = self._extract_city_name(query)
        if not city:
            return "无法识别查询中的城市名称，请提供明确的城市名"
        
        # 调用和风天气API
        weather_data = self._fetch_weather_data(city)
        
        # 格式化输出
        return self._format_weather_output(city, weather_data)

    def _extract_city_name(self, query):
        """从查询中提取城市名称"""
        # 移除"天气"关键词，保留城市名称
        city = re.sub(r'天气', '', query).strip()
        # 进一步清理可能的标点符号
        city = re.sub(r'[，。、；]', '', city).strip()
        return city if city else None

    def _fetch_weather_data(self, city):
        """调用和风天气开发版API获取数据"""
        # 修复点1: 使用开发版域名 (devapi) 而不是正式版
        # 修复点2: 使用公开测试 Key (public_test_key)
        API_URL = "https://devapi.qweather.com/v7/weather/now"
        PUBLIC_API_KEY = "public_test_key" 
        
        try:
            response = requests.get(
                API_URL,
                params={
                    "location": city,
                    "key": PUBLIC_API_KEY # Key 放在参数中
                },
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "200":
                    return data["now"]
                else:
                    # 返回具体的错误信息帮助调试
                    return f"天气API错误: {data.get('message', '未知错误')}"
            return f"天气API请求失败，状态码: {response.status_code}"
        except Exception as e:
            return f"获取天气数据失败: {str(e)}"

    def _format_weather_output(self, city, weather_data):
        """格式化天气输出"""
        # 修复：增加对错误信息的判断，防止字符串拼接报错
        if isinstance(weather_data, str) and not weather_data.startswith("温度"):
            return f"【天气查询】\n地点：{city}\n状态：{weather_data}"
            
        if not weather_data or "temp" not in weather_data:
            return f"【天气查询】\n地点：{city}\n状态：无法获取天气信息"

        temperature = weather_data.get("temp", "N/A")
        weather_condition = weather_data.get("text", "N/A")
        
        return f"【天气查询】\n地点：{city}\n温度：{temperature}°C\n天气状况：{weather_condition}"

    def _perform_generic_search(self, query):
        """执行通用搜索（保留原有的Serper逻辑）"""
        try:
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": TOOL_CONFIG.get("web_search_api", ""),
                "Content-Type": "application/json"
            }
            data = {"q": query, "num": 5}
            
            response = requests.post(url, json=data, headers=headers, timeout=TOOL_CONFIG["timeout"])
            if response.status_code == 200:
                results = response.json()
                snippets = [item.get("snippet", "") for item in results.get("organic", [])[:3]]
                return "\n".join([f"{i+1}. {snippet}" for i, snippet in enumerate(snippets)])
            return f"搜索失败，状态码：{response.status_code}"
        except Exception as e:
            return f"调用搜索工具失败: {str(e)}"
