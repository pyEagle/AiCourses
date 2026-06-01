# -*- coding:utf-8 -*-

import re
import os
import joblib
import numpy as np
import unicodedata
import requests

from sentence_transformers import SentenceTransformer
from lightgbm import LGBMClassifier
from collections import defaultdict


class EdgeIntentEngine:
    def __init__(self, confidence_threshold=0.45, similarity_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.clf = LGBMClassifier(
            random_state=42,
            class_weight='balanced',
            verbose=-1,
            n_jobs=1,
        )

        self.model_path="./model/edge_model.pkl"
        self.exact_match_cache = {}
        self.api_mapping = {}
        self.inverted_index = defaultdict(set)
        self.text_vectors = {}

        # TODO: Agent
        self.agent_flag = True
        self.agent_url = "http://127.0.0.1:9090/api/chat"

    def set_agent_flag(self, flag):
        self.agent_flag = flag

    @staticmethod
    def clean_text(text):
        temp =''.join(
            c for c in text 
            if not unicodedata.category(c).startswith('P')
        )

        return re.sub(r'\s+', ' ', temp).strip()

    def _build_inverted_index(self, texts):
        for text in texts:
            words = list(self.clean_text(text))
            for word in words:
                self.inverted_index[word].add(text)

    def _cosine_similarity(self, vec1, vec2):
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot / (norm + 1e-8)

    def train(self, dataset):
        texts = [item['描述'] for item in dataset]
        labels = [item['API'] for item in dataset]

        for item in dataset:
            self.exact_match_cache[item['描述']] = item['API']
            self.api_mapping[item['API']] = item

        self._build_inverted_index(texts)
        features = self.encoder.encode(texts)
        for text, vec in zip(texts, features):
            self.text_vectors[text] = vec

        self.clf.fit(features, labels)

        joblib.dump({
            'exact_match_cache': self.exact_match_cache,
            'api_mapping': self.api_mapping,
            'encoder': self.encoder,
            'clf': self.clf,
            'inverted_index': self.inverted_index,
            'text_vectors': self.text_vectors
        }, self.model_path)
        print(f"模型及缓存已成功保存至: {self.model_path}")

    def set_model_file(self, model_file):
        self.model_path = model_file

    def load(self):
        try:
            data = joblib.load(self.model_path)
            self.exact_match_cache = data['exact_match_cache']
            self.api_mapping = data['api_mapping']
            self.encoder = data['encoder']
            self.clf = data['clf']
            self.inverted_index = data.get('inverted_index', defaultdict(set))
            self.text_vectors = data.get('text_vectors', {})
            print(f"[*] 成功加载模型组件")
        except Exception as e:
            print(f"[!] 加载失败: {e}")

    def predict(self, text):
        text = self.clean_text(text)
        if text in self.exact_match_cache:
            return self._build_response(True, self.exact_match_cache[text], 1.0, "exact_match")

        query_words = text.split() if ' ' in text else list(text)
        candidate_texts = set()
        for word in query_words:
            if word in self.inverted_index:
                candidate_texts.update(self.inverted_index.get(word))

        if candidate_texts:
            query_vec = self.encoder.encode(text)
            best_sim = 0.0
            best_text = None
            for cand_text in candidate_texts:
                cand_vec = self.text_vectors.get(cand_text)
                if cand_vec is None:
                    continue
                sim = self._cosine_similarity(query_vec, cand_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_text = cand_text
            if best_sim >= self.similarity_threshold and best_text:
                api = self.exact_match_cache[best_text]
                return self._build_response(True, api, best_sim, "inverted_similarity")

        vector = self.encoder.encode([text])
        prob = self.clf.predict_proba(vector)[0]
        max_idx = np.argmax(prob)
        max_p = prob[max_idx]

        if max_p >= self.confidence_threshold:
            return self._build_response(True, self.clf.classes_[max_idx], max_p, "ml_model")
        else:
            if self.agent_flag:
                response = self.ask_agent(text)
                if response.status_code == 200:
                    result = response.json()
                    return self._build_response(True, result['data']['reply'], 1, "ai_agent")

        return self._build_response(False, "未能识别", 0.0, "none")

    def _build_response(self, success, result, confidence, indent_path):
        print('意图识别路径', indent_path, )

        intent_info = self.api_mapping.get(result, {}) if success else {}
        return {
            "api": result if success else None,
            "intent": intent_info.get("意图", "未知"),
            "confidence": round(float(confidence), 4),
        }

    def ask_agent(self, message):
        payload = {
            "message": message,
            "session_id": "session_001",
            }
        
        return requests.post(self.agent_url, json=payload, timeout=5)

