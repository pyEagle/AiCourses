import os
import sys
import random
import shutil
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

# ==========================================
# ONNX 版特征提取器（保持不变）
# ==========================================
class ONNXMedicineEmbedder:
    def __init__(self, onnx_model_path, target_size=320):
        self.target_size = target_size
        
        # 加载 ONNX 模型，自动优先使用 GPU， fallback 到 CPU
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, img):
        # 尺寸对齐：双线性插值
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        # 转 float32 并归一化到 0-1
        img_np = np.array(img, dtype=np.float32) / 255.0
        # HWC -> CHW
        img_np = img_np.transpose(2, 0, 1)
        # 增加 batch 维度
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
# 聚类与数据清洗核心逻辑
# ==========================================
def cluster_and_filter_samples(embedder, input_dir, output_dir):
    """
    1. 提取输入目录所有图像特征。
    2. 根据真实 label 数量设定 K 值进行 K-Means 聚类。
    3. 每个簇按距离簇中心从小到大排序，删减后 1/4 (最远的数据)。
    4. 将保留的样本复制到 output_dir/cluster_X 目录下。
    5. 返回删减后的样本信息和簇中心点坐标 (numpy array)。
    """
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    image_paths = []
    labels = set()
    embeddings = []
    
    print(f"开始扫描目录并提取特征: {input_dir} ...")
    
    # 1. 扫描目录提取特征与 Label
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                path = os.path.join(root, file)
                file_stem = os.path.splitext(file)[0]
                parts = file_stem.split('_')
                
                # 按照命名格式: 医院id_设备编号_label_随机号.jpg
                if len(parts) >= 4:
                    label = parts[2]
                    labels.add(label)
                    
                    # 提取特征并压平为 1D 数组
                    emb = embedder.get_embedding(path).flatten()
                    
                    image_paths.append(path)
                    embeddings.append(emb)
    
    total_samples = len(image_paths)
    num_clusters = len(labels)
    
    if total_samples == 0:
        print("未找到有效格式的图像文件。")
        return {}, np.array([])
        
    print(f"总计扫描到 {total_samples} 个样本，解析到 {num_clusters} 种唯一标签。")
    
    # 转为 numpy array 以供聚类
    X = np.array(embeddings) # shape: (N, 256)
    
    # 2. 设定簇数进行 K-Means 聚类
    print(f"正在使用 {num_clusters} 个簇进行 K-Means 聚类...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_  # shape: (num_clusters, 256)
    
    # 准备输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # 每次运行前清空旧输出以防数据残留
    os.makedirs(output_dir)
    
    retained_samples_dict = {}
    total_retained = 0
    
    # 3 & 4. 遍历每个簇进行排序、过滤和文件移动
    print("开始计算距离、清洗边缘数据并转移文件...")
    for cluster_id in range(num_clusters):
        # 找到属于当前簇的所有样本索引
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        cluster_center = cluster_centers[cluster_id]
        
        # 计算该簇内所有样本到簇中心的欧几里得距离
        # (由于特征已经过L2归一化，欧式距离和余弦距离在此处有单调等价关系)
        distances = []
        for idx in cluster_indices:
            dist = np.linalg.norm(X[idx] - cluster_center)
            distances.append((idx, dist))
            
        # 按距离从小到大排序
        distances.sort(key=lambda x: x[1])
        
        # 计算需要保留的数量 (去掉后 1/4)
        # 用减法确保：如 10 个样本，去掉 10//4 = 2 个，保留 8 个
        remove_count = len(distances) // 4
        keep_count = len(distances) - remove_count
        
        retained_items = distances[:keep_count]
        
        # 创建属于该簇的子目录
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir)
        
        retained_paths_for_this_cluster = []
        
        for idx, dist in retained_items:
            src_path = image_paths[idx]
            file_name = os.path.basename(src_path)
            dest_path = os.path.join(cluster_dir, file_name)
            
            # 将保留的文件复制到指定簇的目录
            shutil.copy2(src_path, dest_path)
            retained_paths_for_this_cluster.append(dest_path)
            
        retained_samples_dict[cluster_id] = retained_paths_for_this_cluster
        total_retained += len(retained_paths_for_this_cluster)
        
        print(f"簇 {cluster_id}: 原有 {len(distances)} 个样本，去掉后1/4({remove_count}个)，保留 {keep_count} 个样本。")

    print(f"\n清洗完成！原始样本: {total_samples} -> 筛选后样本: {total_retained}")
    print(f"删减后的样本已存放在: {os.path.abspath(output_dir)}")
    
    # 5. 返回删减后的样本（字典格式）及簇中心点坐标（NumPy 格式）
    return retained_samples_dict, cluster_centers

# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    # 配置你的模型路径和目录
    ONNX_MODEL_PATH = "./saved_weights/medicine_embedder_best_320_no_norm.onnx"
    import sys
    input_directory = sys.argv[1]
    output_directory = "filtered_dataset"# 清洗并按簇分类后的输出目录
    
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ 未找到ONNX模型文件 {ONNX_MODEL_PATH}，请确认路径！")
        sys.exit(1)
        
    if not os.path.exists(input_directory):
        print(f"❌ 指定的输入目录不存在: {input_directory}")
        sys.exit(1)

    # 初始化 ONNX 特征提取器
    embedder = ONNXMedicineEmbedder(onnx_model_path=ONNX_MODEL_PATH, target_size=320)
    
    # 执行聚类与清洗
    filtered_samples, centers = cluster_and_filter_samples(
        embedder=embedder, 
        input_dir=input_directory, 
        output_dir=output_directory
    )
    
    # 打印最终输出的 NumPy 形状验证结果
    if len(centers) > 0:
        print(f"\n[返回值验证] 成功获取簇中心坐标 numpy 数组，Shape: {centers.shape}")
