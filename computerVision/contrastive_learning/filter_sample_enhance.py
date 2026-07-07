import os
import sys
import shutil
import numpy as np
from PIL import Image
import onnxruntime as ort
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# ==========================================
# ONNX 版特征提取器（保持不变）
# ==========================================
class ONNXMedicineEmbedder:
    def __init__(self, onnx_model_path, target_size=320):
        self.target_size = target_size
        
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, img):
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)
        return np.expand_dims(img_np, axis=0)

    def get_embedding(self, image_data_or_path):
        if isinstance(image_data_or_path, str):
            img = Image.open(image_data_or_path).convert('RGB')
        else:
            img = image_data_or_path 
            
        input_tensor = self._preprocess(img)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        embedding = outputs[0]  # shape: [1, 256]
        
        # L2 归一化
        norm = np.linalg.norm(embedding, ord=2, axis=1, keepdims=True)
        embedding = embedding / (norm + 1e-8)
        
        return embedding

# ==========================================
# 聚类 + 簇内孤立森林数据清洗
# ==========================================
def cluster_and_filter_with_isofor(embedder, input_dir, output_dir, contamination=0.2):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    image_paths = []
    labels = set()
    embeddings = []
    
    print(f"开始扫描目录并提取特征: {input_dir} ...")
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                path = os.path.join(root, file)
                file_stem = os.path.splitext(file)[0]
                parts = file_stem.split('_')
                
                if len(parts) >= 4:
                    label = parts[2]
                    labels.add(label)
                    
                    emb = embedder.get_embedding(path).flatten()
                    image_paths.append(path)
                    embeddings.append(emb)
    
    total_samples = len(image_paths)
    num_clusters = len(labels)
    
    if total_samples == 0:
        print("未找到有效格式的图像文件。")
        return {}, np.array([])
        
    print(f"总计扫描到 {total_samples} 个样本，解析到 {num_clusters} 种唯一标签，将作为 K 值。")
    
    X = np.array(embeddings)
    
    print(f"正在使用 K={num_clusters} 进行全局 K-Means 聚类...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    retained_samples_dict = {}
    refined_centers = []
    total_retained = 0
    
    print(f"开始在每个聚类簇内部运行孤立森林 (Contamination={contamination}) 清洗边缘数据...")
    
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        cluster_X = X[cluster_indices]
        num_items = len(cluster_indices)
        
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir)
        retained_paths_for_this_cluster = []
        retained_embeddings = []
        
        if num_items <= 5:
            print(f"簇 {cluster_id}: 样本过少 ({num_items}个)，跳过清洗，全部保留。")
            inlier_mask = np.ones(num_items, dtype=bool)
        else:
            iso_forest = IsolationForest(
                contamination=contamination, 
                random_state=42, 
                n_estimators=100
            )
            preds = iso_forest.fit_predict(cluster_X)
            inlier_mask = (preds == 1)
            
        for local_idx, is_inlier in enumerate(inlier_mask):
            if is_inlier:
                global_idx = cluster_indices[local_idx]
                src_path = image_paths[global_idx]
                file_name = os.path.basename(src_path)
                dest_path = os.path.join(cluster_dir, file_name)
                
                shutil.copy2(src_path, dest_path)
                retained_paths_for_this_cluster.append(dest_path)
                retained_embeddings.append(X[global_idx])
                
        kept_count = len(retained_paths_for_this_cluster)
        retained_samples_dict[cluster_id] = retained_paths_for_this_cluster
        total_retained += kept_count
        
        print(f"簇 {cluster_id}: 原有 {num_items} 个样本，孤立森林剔除 {num_items - kept_count} 个边缘点，保留 {kept_count} 个。")
        
        if kept_count > 0:
            new_center = np.mean(retained_embeddings, axis=0)
            # L2 归一化
            new_center = new_center / (np.linalg.norm(new_center) + 1e-8)
            refined_centers.append(new_center)

    print(f"\n清洗完成！原始样本: {total_samples} -> 筛选后样本: {total_retained}")
    print(f"删减后的样本已存放在: {os.path.abspath(output_dir)}")
    
    return retained_samples_dict, np.array(refined_centers)


if __name__ == "__main__":
    ONNX_MODEL_PATH = "./saved_weights/medicine_embedder_best_320_no_norm.onnx"
    
    if len(sys.argv) < 2:
        print("用法: python script.py <输入图片目录>")
        sys.exit(1)
        
    input_directory = sys.argv[1]
    output_directory = "filtered_dataset"
    
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ 未找到ONNX模型文件 {ONNX_MODEL_PATH}，请确认路径！")
        sys.exit(1)
        
    if not os.path.exists(input_directory):
        print(f"❌ 指定的输入目录不存在: {input_directory}")
        sys.exit(1)

    embedder = ONNXMedicineEmbedder(onnx_model_path=ONNX_MODEL_PATH, target_size=320)
    
    filtered_samples, centers = cluster_and_filter_with_isofor(
        embedder=embedder, 
        input_dir=input_directory, 
        output_dir=output_directory,
        contamination=0.25
    )
    
    if len(centers) > 0:
        print(f"\n[返回值验证] 成功获取优化后的簇中心坐标 numpy 数组，Shape: {centers.shape}")

  
