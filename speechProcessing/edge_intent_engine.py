# -*- coding:utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NO_MPS"] = "1"

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from lightgbm import LGBMClassifier

"""
training_data = [
    {'意图': '灯光控制', '描述': '开灯', 'API': '/light/on'},
    {'意图': '灯光控制', '描述': '关灯', 'API': '/light/off'},
]
"""
class EdgeIntentEngine:
    def __init__(self, confidence_threshold=0.45):
        self.confidence_threshold = confidence_threshold
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.clf = LGBMClassifier(
            random_state=42, 
            class_weight='balanced', 
            verbose=-1,
            n_jobs=1, # for mac os 
        )
        self.exact_match_cache = {}
        self.api_mapping = {}

    def train(self, dataset, model_path="edge_model.pkl"):
        texts = [item['描述'] for item in dataset]
        labels = [item['API'] for item in dataset]
        
        for item in dataset:
            self.exact_match_cache[item['描述']] = item['API']
            self.api_mapping[item['API']] = item

        print("[*] 语义特征提取...")
        features = self.encoder.encode(texts)
        
        print("[*] 训练LightGBM 分类器...")
        self.clf.fit(features, labels)

        joblib.dump({
            'exact_match_cache': self.exact_match_cache,
            'api_mapping': self.api_mapping,
            'encoder': self.encoder,
            'clf': self.clf
        }, model_path)
        print(f"[*] 模型及缓存已成功保存至: {model_path}")

    def load(self, model_path="edge_model.pkl"):
        try:
            data = joblib.load(model_path)
            self.exact_match_cache = data['exact_match_cache']
            self.api_mapping = data['api_mapping']
            self.encoder = data['encoder']
            self.clf = data['clf']
            print(f"[*] 成功加载模型组件")
        except Exception as e:
            print(f"[!] 加载失败: {e}")

    def predict(self, text):
        text = text.strip()
        if text in self.exact_match_cache:
            return self._build_response(True, self.exact_match_cache[text], 1.0, "cache")
        
        # 提取特征
        vector = self.encoder.encode([text])
        prob = self.clf.predict_proba(vector)[0]
        max_idx = np.argmax(prob)
        max_p = prob[max_idx]
        
        if max_p >= self.confidence_threshold:
            return self._build_response(True, self.clf.classes_[max_idx], max_p, "ml_model")
        
        return self._build_response(False, "未能识别", 0.0, "none")

    def _build_response(self, success, result, confidence, source):
        intent_info = self.api_mapping.get(result, {}) if success else {}
        return {
            "status": "success" if success else "fail",
            "api": result if success else None,
            "intent": intent_info.get("意图", "未知"),
            "confidence": round(float(confidence), 4),
            "source": source
        }


