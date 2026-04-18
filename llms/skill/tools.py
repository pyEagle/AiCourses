import json
import requests
 
from config import config
 
class ToolRegistry:
    _tools = {}
    @classmethod
    def register(cls, name, description):
        def decorator(func):
            cls._tools[name] = {
                "name": name,
                "description": description,
                "function": func
            }
            return func
        return decorator
    
    @classmethod
    def get_tools_schema(cls):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "需要处理的文本内容"
                            },
                            "max_length": {
                                "type": "integer",
                                "description": "摘要最大长度（字数）",
                                "default": 200
                            }
                        },
                        "required": ["text"]
                    }
                }
            }
            for tool in cls._tools.values()
        ]
    
    @classmethod
    def execute(cls, name, **kwargs):
        if name not in cls._tools:
            raise ValueError(f"工具 {name} 未注册")
        return cls._tools[name]["function"](**kwargs)
 
 
@ToolRegistry.register(
    name="summarize_text",
    description="对给定文本进行摘要提取，返回简洁的摘要内容"
)
def summarize_text(text, max_length = 200):
    """文本摘要工具"""
    from skills.text_summary.script import TextSummarizer
    
    summarizer = TextSummarizer()
    result = summarizer.summarize(text, max_length)
    
    return {
        "success": True,
        "summary": result["summary"],
        "original_length": result["original_length"],
        "summary_length": result["summary_length"],
        "compression_rate": result["compression_rate"]
    }
 
 
@ToolRegistry.register(
    name="extract_keywords",
    description="从文本中提取关键词"
)
def extract_keywords(text, top_k=5):
    import re
    from collections import Counter
    
    words = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', text)
    keywords = Counter(words).most_common(top_k)
    
    return {
        "success": True,
        "keywords": [kw[0] for kw in keywords]
    }
