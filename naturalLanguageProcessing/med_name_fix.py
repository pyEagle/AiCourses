# -*- coding:utf-8 -*-

import os
import math
import difflib
from collections import defaultdict

try:
    import pypinyin
    HAS_PINYIN = True
except ImportError:
    HAS_PINYIN = False
    print("提示: 未安装 pypinyin，将关闭音近字纠错增强 (建议 pip install pypinyin)")

try:
    import ahocorasick
    HAS_AC = True
except ImportError:
    HAS_AC = False
    print("提示: 未安装 pyahocorasick，将关闭 AC 自动机快车道 (建议 pip install pyahocorasick)")


class EdgeDrugCorrector:
    def __init__(self, image_dir, k1=1.5, b=0.75):
        self.image_dir = image_dir
        self.drug_names = set()
        self.inverted_index = defaultdict(set)

        self.doc_count = 0
        self.avg_doc_len = 0
        self.doc_lengths = {}
        self.df = defaultdict(int)
        self.idf = {}
        self.k1 = k1
        self.b = b

        self.ac = ahocorasick.Automaton() if HAS_AC else None

    def _get_features(self, text):
        features = []
        for i in range(len(text)):
            features.append(text[i])
            if i < len(text) - 1:
                features.append(text[i:i+2])

        if HAS_PINYIN:
            pys = pypinyin.lazy_pinyin(text)
            for py in pys:
                features.append("py_" + py)
            first_letters = pypinyin.lazy_pinyin(text, style=pypinyin.Style.FIRST_LETTER)
            features.append("p_" + "".join(first_letters))

        return features

    def load_index(self):
        if not os.path.exists(self.image_dir):
            return

        total_length = 0
        for filename in os.listdir(self.image_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            parts = filename.rsplit('.', 1)[0].split('_')
            drug_name = None

            for p in parts:
                if p and p != "未识别" and not p.isascii() and "ocr" not in p:
                    drug_name = p
                    break

            if drug_name and drug_name not in self.drug_names:
                self.drug_names.add(drug_name)
                self.doc_count += 1
                self.doc_lengths[drug_name] = len(drug_name)
                total_length += len(drug_name)

                if self.ac:
                    self.ac.add_word(drug_name, drug_name)

                features = set(self._get_features(drug_name))
                for feat in features:
                    self.inverted_index[feat].add(drug_name)
                    self.df[feat] += 1

        if self.doc_count > 0:
            self.avg_doc_len = total_length / self.doc_count
            for feat, count in self.df.items():
                self.idf[feat] = math.log(((self.doc_count - count + 0.5) / (count + 0.5)) + 1.0)

            if self.ac:
                self.ac.make_automaton()

    def _score_candidates(self, ocr_text, candidates, ocr_features):
        best_drug = ocr_text
        best_score = -1.0
        
        global_ocr_pinyin = pypinyin.lazy_pinyin(ocr_text) if HAS_PINYIN else []

        for drug in candidates:
            bm_score = 0.0
            max_bm_score = 0.0  

            drug_features = self._get_features(drug)
            tf_dict = defaultdict(int)
            for f in drug_features:
                tf_dict[f] += 1

            doc_len = self.doc_lengths.get(drug, self.avg_doc_len)

            for feat, tf in tf_dict.items():
                idf = self.idf.get(feat, 0.0)
                max_bm_score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len)))

            for feat in set(ocr_features):
                if feat in tf_dict:
                    tf = tf_dict[feat]
                    idf = self.idf.get(feat, 0.0)
                    bm_score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len)))

            norm_bm = (bm_score / max_bm_score) if max_bm_score > 0 else 0.0

            window_size = len(drug)
            max_sim_ratio = 0.0
            search_range = max(1, len(ocr_text) - window_size + 1)
            drug_pinyin = "".join(pypinyin.lazy_pinyin(drug)) if HAS_PINYIN else drug

            for i in range(search_range):
                window_text = ocr_text[i:i+window_size]
                char_sim = difflib.SequenceMatcher(None, window_text, drug).ratio()

                if HAS_PINYIN:
                    window_pinyin = "".join(global_ocr_pinyin[i:i+window_size])
                    py_sim = difflib.SequenceMatcher(None, window_pinyin, drug_pinyin).ratio()
                    sim_ratio = max(char_sim, py_sim) * 0.7 + min(char_sim, py_sim) * 0.3
                else:
                    sim_ratio = char_sim

                if sim_ratio > max_sim_ratio:
                    max_sim_ratio = sim_ratio

            if len(drug) < 4 and len(ocr_text) > len(drug) + 2:
                len_diff = abs(len(drug) - len(ocr_text))
                length_weight = 1.0 / (1.0 + len_diff * 0.2)
            else:
                len_diff = max(0, len(drug) - len(ocr_text))
                length_weight = 1.0 / (1.0 + len_diff * 0.1) 

            combined_score = norm_bm * 0.4 + max_sim_ratio * 0.6
            score = combined_score * length_weight

            if score > best_score:
                best_score = score
                best_drug = drug

        return best_drug, best_score

    def _get_dyn_thresh(self, ocr_text, best_drug, base_threshold):
        return base_threshold

    def is_drug(self, text, threshold=0.3):
        if not text or not self.drug_names:
            return False

        clean_text = "".join(char for char in text if char.strip())
        if not clean_text:
            return False

        if self.ac:
            for match in self.ac.iter(clean_text):
                drug_str = match[1]
                if len(drug_str) < 4 and len(clean_text) > len(drug_str) + 2:
                    continue
                return True

        ocr_features = self._get_features(clean_text)
        candidates = set()
        for feat in ocr_features:
            if feat in self.inverted_index:
                candidates.update(self.inverted_index[feat])

        if not candidates:
            return False

        best_drug, best_score = self._score_candidates(clean_text, candidates, ocr_features)
        dyn_thresh = self._get_dyn_thresh(clean_text, best_drug, threshold)
        return best_score >= dyn_thresh

    def correct(self, ocr_text, threshold=0.3, depth=0):
        if depth >= 900 or not ocr_text or not self.drug_names:
            return ocr_text

        clean_text = ""
        index_map = []
        for i, char in enumerate(ocr_text):
            if char.strip():
                clean_text += char
                index_map.append(i)

        if not clean_text:
            return ocr_text

        if self.ac:
            matches = list(self.ac.iter(clean_text))
            valid_matches = []
            for m in matches:
                drug_str = m[1]
                if len(drug_str) < 4 and len(clean_text) > len(drug_str) + 2:
                    continue
                valid_matches.append(m)

            if valid_matches:
                longest_match = max(valid_matches, key=lambda x: len(x[1]))
                end_clean_idx = longest_match[0]
                best_drug = longest_match[1]
                start_clean_idx = end_clean_idx - len(best_drug) + 1

                orig_start = index_map[start_clean_idx]
                orig_end = index_map[end_clean_idx]

                orig_text_slice = ocr_text[orig_start:orig_end + 1]

                if orig_text_slice.replace(" ", "") == best_drug:
                    best_drug_to_insert = orig_text_slice
                else:
                    best_drug_to_insert = best_drug

                left_part = ocr_text[:orig_start]
                right_part = ocr_text[orig_end + 1:]

                return self.correct(left_part, threshold, depth + 1) + best_drug_to_insert + self.correct(right_part, threshold, depth + 1)

        ocr_features = self._get_features(clean_text)
        candidates = set()
        for feat in ocr_features:
            if feat in self.inverted_index:
                candidates.update(self.inverted_index[feat])

        if not candidates:
            return ocr_text

        best_drug, best_score = self._score_candidates(clean_text, candidates, ocr_features)
        dyn_thresh = self._get_dyn_thresh(clean_text, best_drug, threshold)

        if best_score >= dyn_thresh:
            best_clean_i = 0
            max_sim = -1.0
            best_w_size = len(best_drug)
            
            global_clean_pinyin = pypinyin.lazy_pinyin(clean_text) if HAS_PINYIN else []
            drug_pinyin = "".join(pypinyin.lazy_pinyin(best_drug)) if HAS_PINYIN else best_drug

            w_sizes = [len(best_drug) + offset for offset in range(-3, 4)]
            w_sizes = [w for w in w_sizes if w > 0]

            for w_size in w_sizes:
                search_range = max(1, len(clean_text) - w_size + 1)
                for i in range(search_range):
                    window_text = clean_text[i:i+w_size]
                    char_sim = difflib.SequenceMatcher(None, window_text, best_drug).ratio()
                    if HAS_PINYIN:
                        window_pinyin = "".join(global_clean_pinyin[i:i+w_size])
                        py_sim = difflib.SequenceMatcher(None, window_pinyin, drug_pinyin).ratio()
                        sim_ratio = char_sim * 0.6 + py_sim * 0.4
                    else:
                        sim_ratio = char_sim

                    if sim_ratio > max_sim:
                        max_sim = sim_ratio
                        best_clean_i = i
                        best_w_size = w_size

            len_diff = abs(len(best_drug) - best_w_size)
            length_weight = 1.0 / (1.0 + len_diff * 0.2)
            adjusted_sim = max_sim * length_weight

            required_sim = 0.50 if len(best_drug) >= 6 else 0.65
            if adjusted_sim < required_sim:
                return ocr_text

            orig_start = index_map[best_clean_i]
            orig_end = index_map[min(best_clean_i + best_w_size - 1, len(index_map) - 1)]

            orig_text_slice = ocr_text[orig_start:orig_end + 1]
            clean_orig = orig_text_slice.replace(" ", "")

            aligned_best_drug = best_drug

            if clean_orig == aligned_best_drug:
                best_drug_to_insert = orig_text_slice
            else:
                best_drug_to_insert = aligned_best_drug

            left_part = ocr_text[:orig_start]
            right_part = ocr_text[orig_end + 1:]

            return self.correct(left_part, threshold, depth + 1) + best_drug_to_insert + self.correct(right_part, threshold, depth + 1)

        return ocr_text
