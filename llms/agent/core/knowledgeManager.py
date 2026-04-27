import json
from config.settings import KNOWLEDGE_CONFIG

class KnowledgeManger:
    def __init__(self):
        self.faq_data = self._load_faq()

    def _load_faq(self):
        try:
            with open(KNOWLEDGE_CONFIG["faq_path"], "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"FAQ文件未找到，路径：{KNOWLEDGE_CONFIG['faq_path']}")
            return []
        except json.JSONDecodeError:
            print("FAQ文件格式错误")
            return []

    def retrieve(self, query):
        if not self.faq_data:
            return ""
        
        # 简单的关键词匹配（可替换为向量检索等更优方案）
        query_lower = query.lower()
        matched = []
        
        for item in self.faq_data:
            question = item.get("question", "").lower()
            # 关键词匹配
            if any(word in query_lower for word in question.split()):
                matched.append(item.get("answer", ""))
        
        if matched:
            return "\n\n".join(matched[:2])  # 返回前2个匹配结果
        return ""

