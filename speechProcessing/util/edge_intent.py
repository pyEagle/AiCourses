# -*- coding:utf-8 -*-

import re
import os
import joblib
import numpy as np
import unicodedata
import requests
import jieba

from lightgbm import LGBMClassifier
from collections import defaultdict
from rank_bm25 import BM25Okapi

# 引入量化与轻量级推理所需库
import torch
from transformers import AutoTokenizer, AutoModel
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


class ONNXMiniLM:
    """
    轻量级 INT8 语义向量提取器 (无缝替换 SentenceTransformer)
    自动将 HuggingFace 模型转换为 ONNX INT8，专为 RK3588 ARM 架构优化。
    """
    def __init__(self, model_name, cache_dir="./model"):
        self.model_dir = os.path.join(cache_dir, model_name)
        self.onnx_path = os.path.join(self.model_dir, "model.onnx")
        self.quant_onnx_path = os.path.join(self.model_dir, "model_int8.onnx")
        
        os.makedirs(self.model_dir, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/' + model_name)
        
        # 首次运行：自动导出 ONNX 并进行 INT8 量化
        if not os.path.exists(self.quant_onnx_path):
            print(f"[意图引擎] 首次运行：正在将 {model_name} 导出并量化为 INT8 (仅执行一次)...")
            self._export_and_quantize('sentence-transformers/' + model_name)
            print(f"[意图引擎] INT8 量化完成！极速模型已保存至: {self.quant_onnx_path}")
            
        # 挂载 ONNX Runtime (CPUExecutionProvider 针对 ARM 进行了深度汇编优化)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2  # 限制线程，防止与 YOLO 抢占资源
        
        self.session = ort.InferenceSession(self.quant_onnx_path, sess_options, providers=['CPUExecutionProvider'])
        print("[意图引擎] ONNX INT8 极速语义引擎加载完毕。")

    def _export_and_quantize(self, hf_model_id):
        # 1. 下载并加载原始 PyTorch 模型
        model = AutoModel.from_pretrained(hf_model_id)
        model.eval()
        
        # 2. 导出为 ONNX (FP32)
        dummy_text = ["你好"]
        inputs = self.tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)
        
        # 动态提取当前模型架构所需的 input names
        input_names = list(inputs.keys())
        export_inputs = tuple(inputs[k] for k in input_names)
        
        dynamic_axes = {k: {0: 'batch_size', 1: 'sequence_length'} for k in input_names}
        dynamic_axes['last_hidden_state'] = {0: 'batch_size', 1: 'sequence_length'}

        torch.onnx.export(
            model,
            export_inputs,
            self.onnx_path,
            input_names=input_names,
            output_names=['last_hidden_state', 'pooler_output'],
            dynamic_axes=dynamic_axes,
            opset_version=13
        )
        
        # 3. 动态量化为 INT8
        quantize_dynamic(
            model_input=self.onnx_path,
            model_output=self.quant_onnx_path,
            weight_type=QuantType.QUInt8
        )
        # 清理中间文件
        if os.path.exists(self.onnx_path):
            os.remove(self.onnx_path)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            
        # 预处理
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='np')
        
        # 动态映射 inputs 到 ONNX
        ort_inputs = {k: v.astype(np.int64) for k, v in encoded.items()}
        
        # ONNX 推理
        ort_outputs = self.session.run(['last_hidden_state'], ort_inputs)
        token_embeddings = ort_outputs[0]
        
        # Mean Pooling 平均池化 (忽略 Padding 部分)
        attention_mask = ort_inputs['attention_mask']
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        
        sentence_embeddings = sum_embeddings / sum_mask
        
        # L2 归一化
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        sentence_embeddings = sentence_embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        
        # 如果输入是单条文本，返回 1D numpy array 以保持接口兼容
        if len(texts) == 1:
            return sentence_embeddings[0]
        return sentence_embeddings


class EdgeIntentEngine:
    def __init__(self, confidence_threshold=0.45, similarity_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        
        self.encoder = ONNXMiniLM('paraphrase-multilingual-MiniLM-L12-v2')
        
        self.clf = LGBMClassifier(
            random_state=42,
            class_weight='balanced',
            verbose=-1,
            n_jobs=1,
        )

        self.model_path = "./model/edge_model.pkl"
        self.exact_match_cache = {}
        self.api_mapping = {}
        self.text_vectors = {}
        
        self.bm25 = None
        self.corpus_texts = []

        self.agent_flag = True
        self.agent_url = "http://127.0.0.1:9090/api/chat"

    def set_agent_flag(self, flag):
        self.agent_flag = flag

    @staticmethod
    def clean_text(text):
        temp = ''.join(
            c for c in text 
            if not unicodedata.category(c).startswith('P')
        )
        return re.sub(r'\s+', ' ', temp).strip()
        
    def _tokenize(self, text):
        text = self.clean_text(text)
        return [word for word in jieba.lcut(text) if word.strip()]

    def _build_bm25_index(self, texts):
        self.corpus_texts = texts
        tokenized_corpus = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _cosine_similarity(self, vec1, vec2):
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot / (norm + 1e-8)

    def train(self, dataset):
        texts = []
        labels = []

        for item in dataset:
            c_text = self.clean_text(item['句子'])
            texts.append(c_text)
            labels.append(item['指令'])

            self.exact_match_cache[c_text] = item['指令']
            
            if item['指令'] not in self.api_mapping:
                self.api_mapping[item['指令']] = {"意图": item.get("意图", "未知")}

        self._build_bm25_index(texts)
        
        features = [self.encoder.encode(text) for text in texts]
        for text, vec in zip(texts, features):
            self.text_vectors[text] = vec

        self.clf.fit(features, labels)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        joblib.dump({
            'exact_match_cache': self.exact_match_cache,
            'api_mapping': self.api_mapping,
            'clf': self.clf,
            'bm25': self.bm25,
            'corpus_texts': self.corpus_texts,
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
            
            self.clf = data['clf']
            self.bm25 = data.get('bm25', None)
            self.corpus_texts = data.get('corpus_texts', [])
            self.text_vectors = data.get('text_vectors', {})
            print(f"[*] 成功加载模型组件")
        except Exception as e:
            print(f"[!] 加载失败: {e}")

    def predict(self, text, session_id="default_001"):
        clean_text = self.clean_text(text)
        if not clean_text:
            return self._build_response(False, "未能识别", 0.0, "none")
            
        query_vec = None
        
        # 精确匹配
        if clean_text in self.exact_match_cache:
            return self._build_response(True, self.exact_match_cache[clean_text], 1.0, "exact_match")

        # BM25召回，向量重排
        if self.bm25 is not None and self.corpus_texts:
            query_tokens = self._tokenize(text)
            scores = self.bm25.get_scores(query_tokens)
            
            top_n_idx = np.argsort(scores)[::-1][:5]
            candidate_texts = [self.corpus_texts[i] for i in top_n_idx if scores[i] > 0]

            if candidate_texts:
                query_vec = self.encoder.encode(clean_text)
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
                    return self._build_response(True, api, best_sim, "bm25_similarity")

        if not hasattr(self.clf, "classes_") or self.clf.classes_ is None:
            return self._build_response(False, "未能识别", 0.0, "none")

        if query_vec is None:
            query_vec = self.encoder.encode(clean_text)
            
        vector_2d = np.array([query_vec]) if len(query_vec.shape) == 1 else query_vec
        
        prob = self.clf.predict_proba(vector_2d)[0]
        max_idx = np.argmax(prob)
        max_p = prob[max_idx]

        if max_p >= self.confidence_threshold:
            return self._build_response(True, self.clf.classes_[max_idx], max_p, "ml_model")

        return self._build_response(False, "未能识别", 0.0, "none")

    def _build_response(self, success, result, confidence, indent_path):
        print(f'意图识别路径: {indent_path}')
        intent_info = self.api_mapping.get(result, {}) if success else {}
        return {
            "指令": result if success else None,
            "意图": intent_info.get("意图", "未知"),
            "置信度": round(float(confidence), 4),
        }

    def ask_agent(self, message, session_id):
        payload = {}
        return requests.post(self.agent_url, json=payload, timeout=5)


# =====================================================================
# 测试执行模块
# =====================================================================

def main():
    print("\n>>> 初始化 EdgeIntentEngine 测试环境...")
    engine = EdgeIntentEngine()

    engine.set_agent_flag(False)

    train_dataset = [
        {"句子": "开点阿莫西林", "指令": "开药", "意图": "处方开具"},
        {"句子": "去心血管内科挂号", "指令": "挂号", "意图": "科室预约"},
        {"句子": "测一下血压", "指令": "测血压", "意图": "体征测量"},
        {"句子": "我要办理出院", "指令": "出院结算", "意图": "住院办理"},
        {"句子": "你好", "指令": "看来今天心情不错哦", "意图": "闲聊"},
        {"句子": "小明小明", "指令": "在的，想聊什么呢", "意图": "闲聊"},
        {"句子": "打开垃圾箱", "指令": "OpenDustbin", "意图": "打开垃圾箱"},
        {"句子": "核查", "指令": "Check", "意图": "台账"},
        {"句子": "你能做什么", "指令": "我是智能助理，可以做很多", "意图": "问答"},
    ]

    print("\n>>> 开始训练引擎...")
    engine.train(train_dataset)

    print("\n>>> 验证模型加载...")
    engine.load()

    test_queries = [
        "开点阿莫西林",                 # 测试场景 1: 精确匹配 (exact_match)
        "我想去心血管内科挂号",         # 测试场景 2: 相似度召回 (bm25_similarity)
        "帮我量一下血压吧",             # 测试场景 3: 机器学习分类推断 (ml_model)
        "今天天气怎么样？"              # 测试场景 4: 无法识别，走兜底 (none / ai_agent)
    ]

    print("\n>>> 开始意图预测测试...")
    for query in test_queries:
        print("-" * 50)
        print(f"👤 用户输入: {query}")
        result = engine.predict(query)
        print(f"🤖 引擎输出: {result}")
    print("-" * 50)


if __name__ == "__main__":
    main()
