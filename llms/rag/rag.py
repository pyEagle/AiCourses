# -*- coding:utf-8 -*-
 
import sys
import json
 
import requests
import faiss
import numpy as np
 
from sentence_transformers import SentenceTransformer
 
model = SentenceTransformer('your_bert-base-chinese_path', 
                           device='cuda', 
                           )
doc_file = sys.argv[1]
knowledge_docs = []
with open(doc_file, 'r') as fid:
    for line in fid:
        knowledge_docs.append(line.strip())
 
doc_vectors = model.encode(knowledge_docs, convert_to_numpy=True).astype(np.float32)
 
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_vectors)
 
doc_map = {i: doc for i, doc in enumerate(knowledge_docs)}
 
def retrieve_relevant_docs(query, top_k=3):
    query_vector = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vector, top_k)
    relevant_docs = [doc_map[idx] for idx in indices[0] if idx != -1]
    return relevant_docs
 
def call_deepseek_r1(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "deepseek-r1:latest",
        "prompt": prompt,
        "stream": False,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["response"]
    except requests.exceptions.RequestException as e:
        return f"调用模型失败：{str(e)}"
    except KeyError:
        return "解析模型返回结果失败，返回格式异常"
 
def build_enhanced_prompt(query, relevant_docs):
    context = "\n".join([f"- {doc}" for doc in relevant_docs])
    prompt_template = """
请基于以下提供的上下文信息回答问题，只使用上下文里的内容，不要编造信息。如果上下文没有相关信息，请明确说明“未找到相关信息”。
上下文信息：
{context}
用户问题：{query}
回答：
    """
    return prompt_template.format(context=context, query=query)
 
if __name__ == "__main__":
    user_query = "RTX 4090 D的显存和功耗是多少？"
    
    print("===== 检索相关文档 =====")
    relevant_docs = retrieve_relevant_docs(user_query, top_k=3)
    for i, doc in enumerate(relevant_docs, 1):
        print(f"{i}. {doc}")
    
    print("\n===== 构建增强Prompt =====")
    enhanced_prompt = build_enhanced_prompt(user_query, relevant_docs)
    print(enhanced_prompt)
    
    print("\n===== 调用deepseek-r1模型 =====")
    answer = call_deepseek_r1(enhanced_prompt)
    
    print("\n===== 模型回答结果 =====")
    print(answer)
