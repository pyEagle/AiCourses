import json

from sentence_transformers import SentenceTransformer

from config.settings import KNOWLEDGE_CONFIG, DEVICE

class KnowledgeBase:
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

    def retrieve(self, query, topk=2):
        if not self.faq_data:
            return ""
        
        # TODO: 适合教学
        query_lower = query.lower()
        matched = []
        
        for item in self.faq_data:
            question = item.get("question", "").lower()
            if self.compute_similarity(query_lower, question) > 0.5:
                matched.append(item.get("answer", ""))
        
        if matched:
            return "\n\n".join(matched[:topk])
        return ""
 
    def compute_similarity(self, query, question): # TODO: question的embedding预先加载
        embedding1 = model.encode(query, convert_to_tensor=True).to(DEVICE)
        embedding2 = model.encode(question, convert_to_tensor=True).to(DEVICE)
    
        cosine_similarity = torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        ).item()
    
        return cosine_similarity
