import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_CONFIG = {
    "model": "deepseek-r1:latest",
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
    "faq_path": "./knowledge/faq.json"
}

TOOL_CONFIG = {
    "web_search_api": os.getenv("SEARCH_API", ""),  
    "timeout": 10
}

DEVICE = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
