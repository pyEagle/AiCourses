import json
import redis
from config.settings import REDIS_CONFIG

class RedisMemory:
    def __init__(self, max_rounds: int = 50):
        self.client = redis.Redis(
            host=REDIS_CONFIG["host"],
            port=REDIS_CONFIG["port"],
            db=REDIS_CONFIG["db"],
            password=REDIS_CONFIG["password"],
            decode_responses=True
        )
        self.expire_seconds = REDIS_CONFIG["expire_seconds"]
        self.max_rounds = max_rounds

    def get_history(self, session_id):
        try:
            history_str = self.client.get(f"agent:session:{session_id}")
            if not history_str:
                return []
            return json.loads(history_str)
        except Exception as e:
            print(f"获取对话历史失败: {str(e)}")
            return []

    def save_message(self, session_id, role, content):
        """
        :param session_id: 会话ID
        :param role: 角色（user/assistant）
        :param content: 消息内容
        """
        try:
            history = self.get_history(session_id)
            history.append({"role": role, "content": content})
            
            if len(history) > self.max_rounds * 2:
                history = history[-self.max_rounds * 2:]  # 保留最后n轮
            
            self.client.setex(
                f"agent:session:{session_id}",
                self.expire_seconds,
                json.dumps(history, ensure_ascii=False)
            )
        except Exception as e:
            print(f"保存消息失败: {str(e)}")

    def clear_history(self, session_id):
        self.client.delete(f"agent:session:{session_id}")

