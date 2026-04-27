import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_CONFIG = {
    "model": "deepseek-r1:14b",
    #"model": "deepseek-r1:latest",
    "base_url": "http://localhost:11434",
    "temperature": 0.1,
    "max_tokens": 4096
}

REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "db": int(os.getenv("REDIS_DB", 0)),
    "password": os.getenv("REDIS_PASSWORD", ""),
    "expire_seconds": 3600 * 24  # 对话过期时间：24小时
}

KNOWLEDGE_CONFIG = {
    "faq_path": "../agent/knowledge/faq.json"
}

TOOL_CONFIG = {
    "web_search_api": os.getenv("SEARCH_API", ""),  # 可替换为实际搜索API
    "timeout": 10
}

TOOL_CONFIG = {
    "qweather_api_key": "YOUR_QWEATHER_API_KEY",
    "web_search_api": "serper_api_key_实际有效的密钥",
    "timeout": 10
}
