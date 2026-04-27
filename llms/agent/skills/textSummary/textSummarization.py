import json
import requests

from skills.baseSkill import BaseSkill

class TextSummarization(BaseSkill):
    def __init__(self):
        self.OLLAMA_API_URL = "http://localhost:11434/api/generate"
        self.MODEL_NAME = "deepseek-r1:latest"
        self.system_prompt = """
# 任务：文本摘要提取
请你对下方的文本进行专业、精炼的摘要总结，严格遵守以下规则：
1. 保留核心观点、关键数据、主要事件，不添加任何额外信息
2. 语言简洁通顺，逻辑清晰，长度控制在原文的10%-20%
3. 纯文本输出，不要使用markdown、列表、标题等格式
4. 客观中立，不做主观评价

待摘要文本：
{text}
"""

    @property
    def name(self):
        return "textSummarization"

    @property
    def description(self):
        return "使用deepseek完成文本摘要生成"

    @property
    def parameters(self):
        return {
            "text": "需要生成摘要的文本内容"
        }

    def execute(self, **kwargs):
        # 取出参数
        text = kwargs.get("text", "")
        if not text:
            return "错误：未提供需要摘要的文本"

        system_prompt = self.system_prompt.format(text=text)

        payload = {
            "model": self.MODEL_NAME,
            "prompt": system_prompt,
            "stream": False,
            "temperature": 0.1,
            "top_p": 0.9
        }

        try:
            response = requests.post(
                self.OLLAMA_API_URL, 
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            return result["response"].strip()

        except Exception as e:
            return f"调用模型失败：{str(e)}"

