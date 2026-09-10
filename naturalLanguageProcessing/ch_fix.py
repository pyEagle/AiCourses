# -*- coding: utf-8 -*-

import re
import os
import math
import pickle
import pypinyin
import numpy as np

from collections import defaultdict
from rank_bm25 import BM25Okapi

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("未检测到 faiss 库，将以纯 Python 模式运行向量检索。")

try:
    import ahocorasick
    HAS_AC = True
except ImportError:
    HAS_AC = False
    print("未检测到 pyahocorasick 库，将以纯滑动窗口模式运行。")

try:
    import kenlm
    HAS_KENLM = True
except ImportError:
    HAS_KENLM = False
    print("未检测到 kenlm 库，将使用内置 Python N-gram 模型作为降级方案。")

try:
    from gensim.models import FastText
    HAS_FASTTEXT = True
except ImportError:
    HAS_FASTTEXT = False
    print("未检测到 gensim 库，FastText 将以模拟演示模式运行。")


def levenshtein_ratio(s1, s2):
    m, n = len(s1), len(s2)
    if m == 0 and n == 0:
        return 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return 1 - dp[m][n] / max(m, n)


class FastTextWrapper:
    def __init__(self, model_path=None):
        self.index = None
        self.indexed_vocab = []
        self.faiss_to_vocab = {}
        
        # 【性能与精度兼顾修复】：添加本地缓存字典
        self._sim_cache = {}
        self._miss_cache = set()

        if HAS_FASTTEXT and model_path and os.path.exists(model_path):
            self.model = FastText.load(model_path)
            self.vector_dim = self.model.vector_size
            print(f"已成功加载 FastText 语义引擎: {model_path}")
        else:
            self.model = None
            self.vector_dim = 64
            print("【语义兜底引擎】未找到 FastText 模型，已启动降级模式。")

    def build_embedding(self, vocab_list, save_dir="./model"):
        self.indexed_vocab = list(vocab_list)
        if not HAS_FAISS or not self.model:
            print("缺少 FAISS 库或 FastText 模型，跳过向量索引持久化构建。")
            return

        os.makedirs(save_dir, exist_ok=True)
        vectors = []
        self.faiss_to_vocab = {}
        faiss_idx = 0
        for vocab_idx, word in enumerate(self.indexed_vocab):
            try:
                vec = self.model.wv[word]
                vectors.append(vec)
                self.faiss_to_vocab[faiss_idx] = vocab_idx
                faiss_idx += 1
            except KeyError:
                continue

        if vectors:
            vectors = np.array(vectors).astype('float32')
            faiss.normalize_L2(vectors)
            self.index = faiss.IndexFlatIP(self.vector_dim)
            self.index.add(vectors)

            index_path = os.path.join(save_dir, "medical_faiss.index")
            mapping_path = os.path.join(save_dir, "faiss_mapping.pkl")
            faiss.write_index(self.index, index_path)
            with open(mapping_path, "wb") as f:
                pickle.dump({
                    "faiss_to_vocab": self.faiss_to_vocab,
                    "indexed_vocab":  self.indexed_vocab,
                    "vector_dim":     self.vector_dim,
                }, f)
            print(f"FAISS 向量索引已构建并持久化，共载入 {self.index.ntotal} 个实体向量。")

    def load_embedding(self, save_dir="./model"):
        if not HAS_FAISS:
            return False
        index_path = os.path.join(save_dir, "medical_faiss.index")
        mapping_path = os.path.join(save_dir, "faiss_mapping.pkl")
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            return False
        try:
            self.index = faiss.read_index(index_path)
            with open(mapping_path, "rb") as f:
                data = pickle.load(f)
            self.faiss_to_vocab = data["faiss_to_vocab"]
            self.indexed_vocab = data["indexed_vocab"]
            self.vector_dim = data.get("vector_dim", self.vector_dim)
            return True
        except Exception as e:
            print(f"加载 FAISS 索引失败: {e}，将重新构建。")
            return False

    def build_index(self, vocab_list):
        self.build_embedding(vocab_list)

    def similarity(self, w1, w2):
        if not self.model: 
            return 0.0
            
        cache_key = tuple(sorted([w1, w2]))
        if cache_key in self._sim_cache:
            return self._sim_cache[cache_key]
            
        if w1 in self._miss_cache or w2 in self._miss_cache:
            return 0.0

        try:
            sim = float(self.model.wv.similarity(w1, w2))
            self._sim_cache[cache_key] = sim
            return sim
        except Exception:
            if w1 not in self.model.wv: self._miss_cache.add(w1)
            if w2 not in self.model.wv: self._miss_cache.add(w2)
            return 0.0

    def get_most_similar_idx(self, text, vocab_list, threshold=0.85):
        if not self.model or text in self._miss_cache:
            return None, 0.0

        if self.index is not None and self.model:
            try:
                query_vec = self.model.wv[text].reshape(1, -1).astype('float32')
                faiss.normalize_L2(query_vec)
                scores, indices = self.index.search(query_vec, 1)
                best_score = float(scores[0][0])
                if best_score >= threshold and int(indices[0][0]) != -1:
                    return self.faiss_to_vocab[int(indices[0][0])], best_score
            except Exception:
                # 记录无法进行 FAISS 检索的坏词
                self._miss_cache.add(text)
                pass

        best_idx, max_sim = None, -1.0
        for i, word in enumerate(vocab_list):
            sim = self.similarity(text, word)
            if sim > max_sim:
                max_sim = sim
                best_idx = i
        return best_idx, max_sim if max_sim >= threshold else (None, 0.0)


class KenLMWrapper:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"KenLM 模型未找到: {model_path}")
        self.model = kenlm.Model(model_path)

    def score(self, text):
        text_no_punct = re.sub(r'[^\w\u4e00-\u9fa5]', '', text)
        if not text_no_punct:
            return -9999.0
        chars = list(text_no_punct)
        return self.model.score(" ".join(chars)) / max(1, len(chars))


class EdgeNgramLM:
    def __init__(self, corpus_texts):
        if isinstance(corpus_texts, str) and os.path.exists(corpus_texts):
            with open(corpus_texts, 'r', encoding='utf-8') as f:
                corpus_texts = [line.strip() for line in f if line.strip()]

        self.bigrams = defaultdict(int)
        self.unigrams = defaultdict(int)

        for text in corpus_texts:
            text_clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', text)
            chars = ["<BOS>"] + list(text_clean) + ["<EOS>"]
            for i in range(len(chars) - 1):
                self.bigrams[(chars[i], chars[i + 1])] += 1
                self.unigrams[chars[i]] += 1
            self.unigrams[chars[-1]] += 1
        self.vocab_size = len(self.unigrams)

    def score(self, text):
        text_clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', text)
        chars = ["<BOS>"] + list(text_clean) + ["<EOS>"]
        log_prob = 0.0
        for i in range(len(chars) - 1):
            w1, w2 = chars[i], chars[i + 1]
            prob = (self.bigrams.get((w1, w2), 0) + 1) / (self.unigrams.get(w1, 0) + self.vocab_size)
            log_prob += math.log10(prob)
        return log_prob / max(1, (len(chars) - 1))


class BM25MedicalASRCorrector:
    def __init__(self, medical_vocab, lm_scorer=None, semantic_scorer=None, confidence_threshold=0.8):
        if isinstance(medical_vocab, str) and os.path.exists(medical_vocab):
            with open(medical_vocab, 'r', encoding='utf-8') as f:
                self.medical_vocab = [line.strip() for line in f if line.strip()]
        else:
            self.medical_vocab = medical_vocab
            
        self.max_window_size = 0
        self.corpus_tokens = []
        self.lm = lm_scorer
        self.semantic = semantic_scorer
        self.confidence_threshold = confidence_threshold

        self.ac = ahocorasick.Automaton() if HAS_AC else None

        for idx, word in enumerate(self.medical_vocab):
            if len(word) > self.max_window_size:
                self.max_window_size = len(word)
            tokens = self._get_pinyin_tokens(word)
            self.corpus_tokens.append(tokens)

            if self.ac:
                self.ac.add_word(word, (idx, word))

        self.bm25 = BM25Okapi(self.corpus_tokens)

        if self.ac:
            self.ac.make_automaton()
            print("AC 自动机 精确匹配快车道已挂载。")

        if self.semantic and hasattr(self.semantic, 'load_embedding'):
            if not self.semantic.load_embedding():
                self.semantic.build_embedding(self.medical_vocab)
        elif self.semantic and hasattr(self.semantic, 'build_index'):
            self.semantic.build_index(self.medical_vocab)

    def _get_pinyin_tokens(self, text):
        pinyin_list = pypinyin.pinyin(text, style=pypinyin.Style.NORMAL)
        normalized_tokens = []
        for py_seq in pinyin_list:
            py = py_seq[0]
            py = re.sub(r'^zh', 'z', py)
            py = re.sub(r'^ch', 'c', py)
            py = re.sub(r'^sh', 's', py)
            py = re.sub(r'^l', 'n', py)
            py = re.sub(r'ing$', 'in', py)
            py = re.sub(r'eng$', 'en', py)
            normalized_tokens.append(py)
        return normalized_tokens

    def _update_ac_intervals(self, text):
        intervals = set()
        if self.ac:
            for end_idx, (vocab_idx, word) in self.ac.iter(text):
                start_idx = end_idx - len(word) + 1
                intervals.add((start_idx, len(word)))
        return intervals

    def correct(self, text, confidences=None, confidence_threshold=None):
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
        if confidences is not None:
            if len(confidences) != len(text):
                raise ValueError(f"严重错误：传入的 confidences 长度({len(confidences)})与 text({len(text)}) 不一致。")
            current_conf = list(confidences)
        else:
            current_conf = [1.0] * len(text)

        exact_match_intervals = self._update_ac_intervals(text)
        proposals = []
        
        i = 0
        while i < len(text):
            ac_matched = [length for (start, length) in exact_match_intervals if start == i]
            if ac_matched:
                jump_length = max(ac_matched)
                i += jump_length  
                continue
                
            for n in range(1, self.max_window_size + 3):
                if i + n > len(text): continue
                
                window_text = text[i: i+n]
                
                if re.search(r'[^\w\u4e00-\u9fa5]', window_text):
                    continue
                
                if min(current_conf[i : i+n]) >= threshold:
                    continue
                    
                window_tokens = self._get_pinyin_tokens(window_text)
                
                final_candidates = set()
                scores = self.bm25.get_scores(window_tokens)
                for idx in np.argsort(scores)[::-1][:3]:
                    if scores[idx] > 0:
                        final_candidates.add(idx)
                        
                if self.semantic:
                    sem_idx, _ = self.semantic.get_most_similar_idx(window_text, self.medical_vocab, threshold=0.85)
                    if sem_idx is not None:
                        final_candidates.add(sem_idx)
                        
                for best_idx in final_candidates: 
                    target_word = self.medical_vocab[best_idx]
                    target_tokens = self.corpus_tokens[best_idx]
                    
                    len_diff = abs(len(window_text) - len(target_word))
                    if len_diff > 2: 
                        continue
                    if len(window_text) <= 1 and len_diff > 0: 
                        continue
                    
                    best_ratio = levenshtein_ratio(window_tokens, target_tokens)
                    allowed_ratio = max(0.60, 1.0 - (1.5 / max(len(window_tokens), len(target_tokens))))
                    semantic_sim = self.semantic.similarity(window_text, target_word) if self.semantic else 0.0
                    
                    is_pinyin_ok = (best_ratio >= allowed_ratio)
                    is_semantic_ok = (semantic_sim >= 0.85)
                    
                    if (is_pinyin_ok or is_semantic_ok) and window_text != target_word:
                        candidate_text = text[:i] + target_word + text[i+n:]
                        orig_score = self.lm.score(text) if self.lm else -1
                        cand_score = self.lm.score(candidate_text) if self.lm else 0
                        
                        base_conf = max(best_ratio, semantic_sim)
                        
                        if self.lm is None or cand_score > orig_score or (cand_score >= orig_score and base_conf >= 0.85):
                            score_gain = cand_score - orig_score
                            rank_score = base_conf - (len_diff * 0.15)
                            
                            proposals.append({
                                'start': i,
                                'end': i + n,
                                'target': target_word,
                                'rank_score': rank_score,
                                'score_gain': score_gain,
                                'is_semantic': is_semantic_ok and not is_pinyin_ok,
                                'pinyin_ratio': best_ratio,
                                'semantic_sim': semantic_sim,
                                'len_diff': len_diff
                            })
            i += 1  

        proposals.sort(key=lambda x: (x['rank_score'], x['score_gain'], len(x['target'])), reverse=True)
        
        accepted_proposals = []
        for prop in proposals:
            overlap = False
            for acc in accepted_proposals:
                if not (prop['end'] <= acc['start'] or prop['start'] >= acc['end']):
                    overlap = True
                    break
            if not overlap:
                accepted_proposals.append(prop)
                
        accepted_proposals.sort(key=lambda x: x['start'], reverse=True)
        
        corrected_text = text
        for prop in accepted_proposals:
            start, end, target = prop['start'], prop['end'], prop['target']
            window_text = text[start:end]
            
            diff_type = "等长" if prop['len_diff'] == 0 else ("多字" if len(window_text) > len(target) else "漏字")
            
            if prop['is_semantic']:
                print(f"[语义兜底纠错] '{window_text}' -> '{target}' (打分: {prop['semantic_sim']:.2f})")
            else:
                print(f"[拼音常规纠错 - {diff_type}修复] '{window_text}' -> '{target}' (打分: {prop['pinyin_ratio']:.2f})")
                
            corrected_text = corrected_text[:start] + target + corrected_text[end:]
            
        return corrected_text


def test_ch_fix():
    correct_dictionary = [
        "医生", "阿莫西林", "布洛芬", "奥美拉唑", "医用棉", "脱脂棉",
        "心血管内科", "神经外科", "呼吸科",
        "静脉注射", "切开", "清创缝合"
    ]
    
    training_corpus = [
        "医 生 ， 请 给 他 开 一 点 阿 莫 西 林 。",
        "医 生 ， 请 给 他 开 一 包 脱 脂 棉 。",
        "患 者 家 属 去 心 血 管 内 科 挂 号 。",
        "去 呼 吸 科 查 一 下 肺 功 能 。",
        "准 备 好 工 具 ， 马 上 进 行 清 创 缝 合 。",
        "立 刻 在 手 臂 上 做 切 开 处 理 。",
        "他 需 要 马 上 做 那 个 静 脉 注 射 。",
        "普 通 的 感 冒 ， 多 喝 水 就 行 。",
        "请 给 那 个 病 人 找 一 下 医 生 。"
    ]

    fasttext_engine = FastTextWrapper(model_path="./model/fasttext_model.bin")  

    lm_model_path = "./model/medical_lm.bin"
    if HAS_KENLM and os.path.exists(lm_model_path):
        lm = KenLMWrapper(lm_model_path)
    else:
        lm = EdgeNgramLM(training_corpus)

    corrector = BM25MedicalASRCorrector(
        medical_vocab=correct_dictionary, 
        lm_scorer=lm, 
        semantic_scorer=fasttext_engine, 
        confidence_threshold=0.99 
    )

    test_sentences = [
        "医生，请给他开一包医用棉。",         
        "患者去新血管内科开点阿莫洗林吧。",   
        "患者去心血管科开点阿莫林吧。",       
        "患者去新血管内内科开点阿莫西洗林。", 
        "立刻在手臂上做划开处理。",
        "请给那个病人找一下大夫。"
    ]

    print("\n---------- 开始测试 (自适应漏字、多字、错字、语义全域修复) ----------")
    for sentence in test_sentences:
        conf = [0.3] * len(sentence) 
        result = corrector.correct(sentence, confidences=conf)
        print(f"ASR原文: {sentence} \n-> 纠错后: {result}\n")


if __name__ == "__main__":
    HAS_FAISS = True
    HAS_AC = True
    HAS_KENLM = True
    HAS_FASTTEXT = True

    test_ch_fix()
