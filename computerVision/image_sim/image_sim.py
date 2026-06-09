# -*- coding:utf-8 -*-

import os
import sys
import cv2
import faiss
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

class YoloClipFaissRetriever:
    def __init__(self, yolo_weight="best007.pt", clip_model="openai/clip-vit-base-patch32"):
        print("加载 YOLOv8 模型...")
        self.yolo = YOLO(yolo_weight)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"运行设备: {self.device}")

        print("加载 CLIP 模型...")
        self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model, clean_up_tokenization_spaces=True)
        self.clip_model.eval()

        self.index = None
        self.image_paths = []

    def get_all_crops(self, image_path, conf_thresh=0.5):
        img = cv2.imread(image_path)
        if img is None:
            return []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.yolo(img, verbose=False)
        result = results[0]

        crop_images = []
        if len(result.boxes) > 0:
            for box_data in result.boxes:
                conf = float(box_data.conf.cpu().numpy()[0])
                if conf > conf_thresh:
                    x1, y1, x2, y2 = map(int, box_data.xyxy[0].cpu().numpy())
                    h, w = img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    crop = img_rgb[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_images.append(Image.fromarray(crop))

        if not crop_images:
            crop_images.append(Image.fromarray(img_rgb))

        return crop_images

    @torch.no_grad()
    def extract_feature(self, image_or_text):
        if isinstance(image_or_text, str):
            inputs = self.clip_processor(text=[image_or_text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feat = self.clip_model.get_text_features(**inputs)
        else:
            #if isinstance(image_or_text, Image.Image):
            #    image_or_text = image_or_text.convert("RGB")
            
            inputs = self.clip_processor(images=image_or_text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feat = self.clip_model.get_image_features(**inputs)

        feat = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy().astype("float32")

    def build_index(self, image_paths):
        print(f"构建 FAISS 索引，共 {len(image_paths)} 张图片...")
        embeddings = []
        valid_paths = []

        count, failed = 0, []
        for p in image_paths:
            crops = self.get_all_crops(p, conf_thresh=0.25)
            if crops:
                try:
                    emb = self.extract_feature(crops[0])
                    embeddings.append(emb)
                    valid_paths.append(p)
                except Exception as e:
                    count += 1
                    failed.append(p)
                    #print('特征提取失败: ', p)
        print(f'构建失败，共计{count}张:')
        print('failed: '+'\n'.join(failed))

        if not embeddings:
            print("未提取到有效特征！")
            return

        embeddings = np.vstack(embeddings)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.image_paths = valid_paths
        print("FAISS 索引建立完毕！")

    def search(self, query, topk=1):
        is_image_path = isinstance(query, str) and os.path.exists(query) and query.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        final_results = []
        if is_image_path:
            print(f"正在分析查询图片: {query}")
            crops = self.get_all_crops(query, conf_thresh=0.5)
            print(f"发现 {len(crops)} 个有效目标。")
            for i, crop in enumerate(crops):
                q_feat = self.extract_feature(crop)
                scores, idxs = self.index.search(q_feat, topk)
                target_matches = [(self.image_paths[idx], float(score)) for score, idx in zip(scores[0], idxs[0]) if idx != -1]
                final_results.append({"target_id": i + 1, "matches": target_matches})
        else:
            print(f"正在通过 [文本] 检索: '{query}'")
            q_feat = self.extract_feature(str(query))
            scores, idxs = self.index.search(q_feat, topk)
            target_matches = [(self.image_paths[idx], float(score)) for score, idx in zip(scores[0], idxs[0]) if idx != -1]
            final_results.append({"target_id": "Text Query", "matches": target_matches})

        return final_results

def test():
    dir_name = sys.argv[1]
    query_input = sys.argv[2]

    gallery = []
    valid_extensions = ('.jpg', '.jpeg', '.png')
    for f in os.listdir(dir_name):
        if f.lower().endswith(valid_extensions):
            gallery.append(os.path.join(dir_name, f))

    retriever = YoloClipFaissRetriever()
    retriever.build_index(gallery)
    results = retriever.search(query_input, topk=1)

    print("最终匹配结果:")
    for res in results:
        print(f"[检出目标 {res['target_id']}] 最可能对应的库内药品:")
        for path, score in res['matches']:
            print(f"库图路径: {path}  |  相似度: {score:.4f}")

if __name__ == "__main__":
    test()
