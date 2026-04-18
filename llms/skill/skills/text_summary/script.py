import re
 
class TextSummarizer:
    """文本摘要处理器"""
    
    def __init__(self):
        self.min_sentence_length = 10
        self.default_max_length = 200
    
    def _split_sentences(self, text):
        """分割句子"""
        # 中文句号、英文句号、感叹号、问号
        separators = r'[。！？.!?]'
        sentences = re.split(separators, text)
        return [s.strip() for s in sentences if len(s.strip()) > self.min_sentence_length]
    
    def _score_sentence(self, sentence, all_sentences):
        """句子重要性评分"""
        score = 0
        
        length = len(sentence)
        if 20 <= length <= 100:
            score += 2
        
        if sentence in all_sentences[:3]:
            score += 3
        if sentence in all_sentences[-3:]:
            score += 2
        
        keywords = ['重要', '关键', '主要', '首先', '其次', '最后', '总结', '因此', '所以']
        for kw in keywords:
            if kw in sentence:
                score += 1
        
        return score
    
    def summarize(self, text: str, max_length: int = None):
        if max_length is None:
            max_length = self.default_max_length
        
        original_length = len(text)
        
        if original_length <= max_length:
            return {
                "summary": text,
                "original_length": original_length,
                "summary_length": original_length,
                "compression_rate": 1.0
            }
        
        sentences = self._split_sentences(text)
        
        if not sentences:
            return {
                "summary": text[:max_length],
                "original_length": original_length,
                "summary_length": max_length,
                "compression_rate": max_length / original_length
            }
        
        scored = [(s, self._score_sentence(s, sentences)) for s in sentences]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        summary_sentences = []
        current_length = 0
        for sentence, score in scored:
            if current_length + len(sentence) <= max_length:
                summary_sentences.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        summary_sentences = [
            s for s in sentences 
            if s in summary_sentences
        ]
        
        summary = '。'.join(summary_sentences)
        if summary and not summary.endswith('。'):
            summary += '。'
        
        return {
            "summary": summary,
            "original_length": original_length,
            "summary_length": len(summary),
            "compression_rate": round(len(summary) / original_length, 2)
        }
