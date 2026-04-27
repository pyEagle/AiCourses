import json
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from config.settings import KNOWLEDGE_CONFIG

class KnowledgeManger:
    def __init__(self):
        self.model = SentenceTransformer(KNOWLEDGE_CONFIG['model_file'])
        self.faq_data, self.question_embeddings = self._load_faq()

    def _load_faq(self):
        try:
            with open(KNOWLEDGE_CONFIG["faq_path"], "r", encoding="utf-8") as f:
                faq_data = json.load(f)
        except FileNotFoundError:
            print(f"FAQ文件未找到，路径：{KNOWLEDGE_CONFIG['faq_path']}")
            return [], []
        except json.JSONDecodeError:
            print("FAQ文件格式错误")
            return [], []

        questions = [item["question"] for item in faq_data]
        question_embeddings = self.model.encode(questions)

        return faq_data, question_embeddings

    def retrieve(self, query, topK=2):
        if not self.faq_data:
            return ""

        query_embedding = self.model.encode([query])[0].reshape(1, -1)

        similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
        top_indices = np.argsort(similarities)[-topK:][::-1]

        matched = [self.faq_data[i]["answer"] for i in top_indices]
        return "\n\n".join(matched)
