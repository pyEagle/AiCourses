# -*- coding:utf-8 -*-

import json
import numpy as np
import os

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from config.settings import KNOWLEDGE_CONFIG

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

class KnowledgeManger:
    def __init__(self):
        self.model = SentenceTransformer(KNOWLEDGE_CONFIG['model_file'])
        self.faq_data, self.question_embeddings = self._load_faq()

        self.DOC_DIR = KNOWLEDGE_CONFIG['docs_dir']
        self.doc_chunks = []
        self.doc_embeddings = None
        self.similarity_threshold = 0.92
        self._load_local_documents()

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

    def _load_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
            elif ext == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                return []
            return loader.load()
        except:
            return []

    def _load_local_documents(self):
        if not os.path.exists(self.DOC_DIR):
            os.makedirs(self.DOC_DIR)
            return

        docs = []
        for f in os.listdir(self.DOC_DIR):
            path = os.path.join(self.DOC_DIR, f)
            if os.path.isfile(path):
                docs.extend(self._load_file(path))

        if not docs:
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        self.doc_chunks = [d.page_content for d in splits]

        self.doc_embeddings = self.model.encode(self.doc_chunks)
        print(f"本地文档加载完成：共 {len(self.doc_chunks)} 个片段")

    def retrieve(self, query, topK=2):
        result = []

        if self.faq_data:
            query_emb = self.model.encode([query])[0].reshape(1, -1)
            sims = cosine_similarity(query_emb, self.question_embeddings)[0]
            valid_indices = [i for i, s in enumerate(sims) if s >= self.similarity_threshold]
            if valid_indices:
                top_idx = np.argsort(sims[valid_indices])[-topK:][::-1]
                top_idx = [valid_indices[i] for i in top_idx]
                faq_answers = [self.faq_data[i]["answer"] for i in top_idx]
                result.extend(faq_answers)

        if self.faq_data: return 'qa', "\n\n".join(result)

        if self.doc_chunks and self.doc_embeddings is not None:
            query_emb = self.model.encode([query])[0].reshape(1, -1)
            sims = cosine_similarity(query_emb, self.doc_embeddings)[0]
            valid_indices = [i for i, s in enumerate(sims) if s >= self.similarity_threshold]
            if valid_indices:
                top_idx = np.argsort(sims[valid_indices])[-topK:][::-1]
                top_idx = [valid_indices[i] for i in top_idx]
                doc_contents = [self.doc_chunks[i] for i in top_idx]
                result.extend(doc_contents)

        return 'doc', "\n\n".join(result)

