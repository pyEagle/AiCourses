import sys
import os
import cv2
import torch
import numpy as np
import faiss
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F


class YoloClipFaissRetriever:

    def __init__(
        self,
        yolo_weight="yolov8n.pt",
        clip_model="openai/clip-vit-base-patch32"
    ):
        print("加载 YOLOv8 模型")
        self.yolo = YOLO(yolo_weight)

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"-> 运行设备: {self.device}")

        print("加载 CLIP 模型")
        self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.clip_model.eval()

        self.index = None
        self.embeddings = []
        self.image_paths = []

    def get_crop(self, image_path):
        img = cv2.imread(image_path)
        results = self.yolo(img, verbose=False)
        result = results[0]

        if len(result.boxes) == 0:
            crop = img
        else:
            confs = result.boxes.conf.cpu().numpy()
            best = np.argmax(confs)
            box = result.boxes.xyxy[best].cpu().numpy()

            x1, y1, x2, y2 = map(int, box)
            h, w = img.shape[:2]

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                crop = img

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop)

    @torch.no_grad()
    def extract_image_embedding(self, image_path):
        image = self.get_crop(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        feat = self.clip_model.get_image_features(**inputs)
        feat = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy().astype("float32")

    @torch.no_grad()
    def extract_text_embedding(self, text):
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        feat = self.clip_model.get_text_features(**inputs)
        feat = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy().astype("float32")

    def build_index(self, image_paths):
        print(f"\n构建 FAISS 索引，共 {len(image_paths)} 张图片...")
        embeddings = []
        self.image_paths = image_paths

        for p in image_paths:
            emb = self.extract_image_embedding(p)
            embeddings.append(emb)

        embeddings = np.vstack(embeddings)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.embeddings = embeddings
        print("FAISS 索引建立完毕！")

    def search(self, query, topk=5):
        if isinstance(query, str) and os.path.exists(query):
            print(f"\n🔍 正在通过 [图片] 检索: {query}")
            q = self.extract_image_embedding(query)
        else:
            print(f"\n🔍 正在通过 [文本] 检索: '{query}'")
            q = self.extract_text_embedding(str(query))

        scores, idxs = self.index.search(q, topk)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.image_paths[idx], float(score)))

        return results


def main():
    dir_name = sys.argv[1]

    gallery = []
    valid_image = ['jpg', 'png']
    for f in os.listdir(dir_name):
        t = [1 for i in valid_image if i in f]
        if not t: continue
        gallery.append(os.path.join(dir_name, f))

    retriever = YoloClipFaissRetriever()
    retriever.build_index(gallery)

    test_img = sys.argv[2]
    results_img = retriever.search(test_img, topk=2)
    print("\n检索结果:")
    for path, score in results_img:
        print(f"   {path}  ->  相似度: {score:.4f}")


if __name__ == "__main__":
    main()

