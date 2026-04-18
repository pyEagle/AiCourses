import os
from dotenv import load_dotenv
from pathlib import Path
 
load_dotenv()
 
class Config:
    # Ollama 配置
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
    OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")
    OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
    
    # 目录配置
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "./skill_demo"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output/summaries"))
    LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
    
    # 模型参数
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    @classmethod
    def init_dirs(cls):
        """初始化必要目录"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
 
config = Config()
Config.init_dirs()
