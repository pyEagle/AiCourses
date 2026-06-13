import os
import cv2
import numpy as np
import yaml
import glob
from paddleocr import PaddleOCR
from ultralytics import YOLO

class TextExtractor:
    def __init__(self, lang="ch"):
        self.ocr = PaddleOCR(use_angle_cls=False, lang=lang, rec=True)

    def get_text_lines(self, img_path):
        result = self.ocr.ocr(img_path, cls=False)
        lines_info = []
        
        if result and result[0]:
            for line in result[0]:
                box = line[0]         # 文本行四个顶点的坐标
                text = line[1][0]     # 识别出的文本内容
                conf = line[1][1]     # 置信度
                lines_info.append({
                    'text': text,
                    'box': box,
                    'conf': conf
                })

        return lines_info

class CharSegmenter:
    @staticmethod
    def get_char_boxes(img_crop):
        if img_crop is None or img_crop.size == 0:
            return []
            
        img = img_crop.copy()
        pixel_values = img.reshape((-1, 3)).astype(np.float32)
        
        if len(pixel_values) < 2:
            return []
            
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        centers = np.uint8(centers)
        bg_cluster_idx = np.argmin(np.mean(centers, axis=1))
        text_cluster_idx = 1 - bg_cluster_idx
        mask = (labels == text_cluster_idx).reshape(img.shape[0], img.shape[1]).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        initial_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 2 and h > 2 and w < img.shape[1] * 0.8:
                initial_boxes.append([x, y, w, h])
                
        if not initial_boxes:
            return []

        heights = [b[3] for b in initial_boxes]
        median_h = np.median(heights)

        merged_any = True
        while merged_any:
            merged_any = False
            used = [False] * len(initial_boxes)
            
            for i in range(len(initial_boxes)):
                if used[i]: continue
                x1, y1, w1, h1 = initial_boxes[i]
                
                for j in range(i + 1, len(initial_boxes)):
                    if used[j]: continue
                    x2, y2, w2, h2 = initial_boxes[j]
                    
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    min_w = min(w1, w2)
                    
                    if min_w == 0: continue
                    if overlap_x / min_w < 0.5: continue
                    
                    gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
                    if gap_y > median_h * 0.8: continue
                    if h1 >= median_h * 0.5 and h2 >= median_h * 0.5: continue
                    
                    new_x, new_y = min(x1, x2), min(y1, y2)
                    new_w = max(x1 + w1, x2 + w2) - new_x
                    new_h = max(y1 + h1, y2 + h2) - new_y
                    
                    initial_boxes[i] = [new_x, new_y, new_w, new_h]
                    used[j] = True
                    merged_any = True
                    x1, y1, w1, h1 = initial_boxes[i] 
                    
            initial_boxes = [initial_boxes[k] for k in range(len(initial_boxes)) if not used[k]]

        return [box for box in initial_boxes if box[3] > 5]

class YOLOManager:
    def __init__(self, dataset_dir="./dataset", model_type="yolov8m.pt"):
        self.dataset_dir = dataset_dir
        self.model_type = model_type
        self.yaml_path = "char_dataset.yaml"
        self.extractor = TextExtractor()
        
    def build_dataset(self, raw_image_dir):
        images_out = os.path.join(self.dataset_dir, "images")
        labels_out = os.path.join(self.dataset_dir, "labels")
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)

        valid_images = [f for f in os.listdir(raw_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"[*] 发现 {len(valid_images)} 张原始图片，开始生成伪标签...")

        for img_name in valid_images:
            img_path = os.path.join(raw_image_dir, img_name)
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]
            
            lines_info = self.extractor.get_text_lines(img_path)
            yolo_labels = []
            
            for line in lines_info:
                box = line['box']
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                
                min_x, max_x = max(0, int(min(x_coords))), min(img_w, int(max(x_coords)))
                min_y, max_y = max(0, int(min(y_coords))), min(img_h, int(max(y_coords)))
                
                img_crop = img[min_y:max_y, min_x:max_x]
                
                char_boxes_local = CharSegmenter.get_char_boxes(img_crop)
                
                for (cx_local, cy_local, cw, ch) in char_boxes_local:
                    gx1 = min_x + cx_local
                    gy1 = min_y + cy_local
                    
                    x_center = (gx1 + cw / 2.0) / img_w
                    y_center = (gy1 + ch / 2.0) / img_h
                    norm_w = cw / img_w
                    norm_h = ch / img_h
                    
                    yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            
            if yolo_labels:
                cv2.imwrite(os.path.join(images_out, img_name), img)
                with open(os.path.join(labels_out, img_name.rsplit('.', 1)[0] + '.txt'), 'w') as f:
                    f.write('\n'.join(yolo_labels))
                print(f"  -> {img_name}: 提取文本行 {len(lines_info)} 处，字符 {len(yolo_labels)} 个。")

        yaml_content = f"""path: {os.path.abspath(self.dataset_dir)}\ntrain: images\nval: images\nnames:\n  0: character"""
        with open(self.yaml_path, "w") as f:
            f.write(yaml_content)

    def train(self, tune_epochs=10, tune_iters=10, train_epochs=50, imgsz=640):
        model = YOLO(self.model_type) 
        model.tune(data=self.yaml_path, epochs=tune_epochs, iterations=tune_iters, imgsz=imgsz, device="0")
        
        tune_dirs = sorted(glob.glob("./runs/detect/tune*"), key=os.path.getmtime)
        best_hyperparams = {}
        if tune_dirs:
            best_yaml_path = os.path.join(tune_dirs[-1], "best_hyperparameters.yaml")
            if os.path.exists(best_yaml_path):
                with open(best_yaml_path, "r") as f:
                    best_hyperparams = yaml.safe_load(f)
                    best_hyperparams.pop('fitness', None)
                print(f"[*] 加载最佳超参数成功: {best_yaml_path}")
        
        model = YOLO(self.model_type)
        model.train(data=self.yaml_path, epochs=train_epochs, imgsz=imgsz, device="0", **best_hyperparams)

    def inference(self, img_path):
        train_dirs = sorted(glob.glob("./runs/detect/train*"), key=os.path.getmtime)
        if not train_dirs:
            print("[!] 未找到训练记录。")
            return
            
        best_model_path = os.path.join(train_dirs[-1], "weights", "best.pt")
        if not os.path.exists(best_model_path):
            print(f"[!] 模型权重不存在: {best_model_path}")
            return
            
        print(f"\n[*] 阶段 3：加载模型 {best_model_path} 进行推理 ...")
        best_model = YOLO(best_model_path)
        res = best_model.predict(source=img_path, save=True, show=True, conf=0.4)
        print("[*] 推理结束，结果保存在 ./runs/detect/predict 目录中。")

def run_pipeline():
    # 参数配置
    RAW_IMAGES_DIR = "./raw_images"    # 你存放需打标签图像的目录
    DATASET_DIR = "./yolo_dataset"     # 生成的 YOLO 数据集存放目录
    TEST_IMAGE = "test_image.jpg"      # 最终推理测试用图
    
    manager = YOLOManager(dataset_dir=DATASET_DIR)
    
    print(">>> 启动自动标注流水线 <<<")
    manager.build_dataset(raw_image_dir=RAW_IMAGES_DIR)
    
    if os.path.exists(os.path.join(DATASET_DIR, "labels")):
        manager.train(tune_epochs=5, tune_iters=5, train_epochs=30)
    
    if os.path.exists(TEST_IMAGE):
        manager.inference(img_path=TEST_IMAGE)
    else:
        print(f"\n[提示] 根目录下未发现 {TEST_IMAGE}，跳过推理阶段。")

if __name__ == "__main__":
    run_pipeline()

