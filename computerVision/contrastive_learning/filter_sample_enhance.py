import os
import sys
import shutil
import pickle
import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use('Agg')  # 兼容无GUI服务器环境，确保可正常保存图片
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 

# ==========================================
# 1. ONNX 版特征提取器
# ==========================================
class ONNXMedicineEmbedder:
    def __init__(self, onnx_model_path, target_size=320):
        self.target_size = target_size
        # 加载 ONNX 模型，优先 GPU
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
        embedding = outputs[0]  
        # cl02导出ONNX已内置F.normalize，无需额外L2归一化
        return embedding.flatten()

# ==========================================
# 2. PCA 3D 可视化方法
# ==========================================
def plt_pca_vis(all_embeddings, all_labels, all_is_retained, save_path="pca_iforest_visualization_3d.png"):
    pca = PCA(n_components=3, random_state=42)
    emb_3d = pca.fit_transform(all_embeddings)
    
    unique_labels = sorted(list(set(all_labels)))
    num_classes = len(unique_labels)
    cmap = plt.get_cmap('tab20')
    color_list = [cmap(i % 20) for i in range(num_classes)]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    labels_np = np.array(all_labels)
    retained_np = np.array(all_is_retained)
    for cls_idx, label in enumerate(unique_labels):
        cls_mask = labels_np == label
        retained_mask = cls_mask & retained_np
        removed_mask = cls_mask & ~retained_np
        
        ax.scatter(
            emb_3d[retained_mask, 0],
            emb_3d[retained_mask, 1],
            emb_3d[retained_mask, 2],
            color=color_list[cls_idx],
            label=f"Label {label} 保留",
            alpha=0.7,
            s=30
        )
        if np.sum(removed_mask) > 0:
            ax.scatter(
                emb_3d[removed_mask, 0],
                emb_3d[removed_mask, 1],
                emb_3d[removed_mask, 2],
                facecolors='none',
                edgecolors=color_list[cls_idx],
                marker='x',
                linewidths=1.2,
                label=f"Label {label} 剔除",
                alpha=0.9,
                s=50
            )
    
    total_num = len(all_embeddings)
    retained_num = sum(all_is_retained)
    removed_num = total_num - retained_num
    ax.set_title(
        f"Isolation Forest 清洗效果 PCA 3D 可视化\n"
        f"总样本: {total_num} | 保留: {retained_num} | 剔除: {removed_num}",
        fontsize=14,
        pad=15
    )
    
    ax.set_xlabel(f"PC1 方差解释率: {pca.explained_variance_ratio_[0]:.2%}", fontsize=10)
    ax.set_ylabel(f"PC2 方差解释率: {pca.explained_variance_ratio_[1]:.2%}", fontsize=10)
    ax.set_zlabel(f"PC3 方差解释率: {pca.explained_variance_ratio_[2]:.2%}", fontsize=10)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, borderaxespad=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 PCA 3D 可视化图已保存至: {os.path.abspath(save_path)}")
    plt.close()

# ==========================================
# 3. 基于 Isolation Forest 的分组清洗核心逻辑
# ==========================================
def iforest_filter_samples(
    embedder, 
    input_dir, 
    output_dir, 
    contamination=0.15,
    text_embedding_dict=None,
    alpha=0.7
):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    label_to_data = {}
    
    all_embeddings = []
    all_labels = []
    all_is_retained = []
    
    print(f"开始扫描目录并提取特征: {input_dir} ...")
    total_samples = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                path = os.path.join(root, file)
                file_stem = os.path.splitext(file)[0]
                parts = file_stem.split('_')
                
                # 解析文件名: hospitalid_deviceid_label_md5
                if len(parts) >= 4:
                    label = parts[2]
                    # 提取文本emb key: 取label字段最后7位，对齐rknn_retriever逻辑
                    imag_name = label[-7:]
                    # 图像原始特征
                    img_emb = embedder.get_embedding(path)
                    
                    # 图文加权融合（核心修改）
                    if text_embedding_dict is not None and imag_name in text_embedding_dict:
                        text_emb = np.array(text_embedding_dict[imag_name], dtype=np.float32).squeeze()
                        fuse_emb = img_emb * alpha + (1 - alpha) * text_emb
                    else:
                        fuse_emb = img_emb
                    
                    if label not in label_to_data:
                        label_to_data[label] = []
                    label_to_data[label].append((path, fuse_emb))
                    total_samples += 1
    num_classes = len(label_to_data)
    if total_samples == 0:
        print("未找到有效格式的图像文件。")
        return {}, np.array([])
        
    print(f"总计扫描到 {total_samples} 个样本，解析到 {num_classes} 个真实类别。")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    retained_samples_dict = {}
    cluster_centers = []
    total_retained = 0
    
    print(f"\n开始执行 Isolation Forest 异常清洗 (剔除率设为: {contamination*100}%)...")
    if text_embedding_dict is not None:
        print(f"已启用图文特征融合，图像权重 alpha={alpha}")
    
    for label, items in label_to_data.items():
        paths = [item[0] for item in items]
        embeddings = np.array([item[1] for item in items])
        num_items = len(paths)
        
        label_dir = os.path.join(output_dir, f"label_{label}")
        os.makedirs(label_dir)
        retained_paths_for_this_label = []
        
        if num_items < 5:
            print(f"类别 [{label}]: 样本过少({num_items}个)，跳过清洗，全部保留。")
            retained_indices = list(range(num_items))
        else:
            clf = IsolationForest(
                n_estimators=100, 
                contamination=contamination, 
                random_state=42, 
                n_jobs=-1
            )
            preds = clf.fit_predict(embeddings)
            retained_indices = np.where(preds == 1)[0]
            remove_count = num_items - len(retained_indices)
            print(f"类别 [{label}]: 原有 {num_items:3d} 个，剔除边缘点 {remove_count:3d} 个，保留 {len(retained_indices):3d} 个。")
        
        retained_mask = np.zeros(num_items, dtype=bool)
        retained_mask[retained_indices] = True
        all_embeddings.extend(embeddings.tolist())
        all_labels.extend([label] * num_items)
        all_is_retained.extend(retained_mask.tolist())
        
        retained_embeddings = []
        for idx in retained_indices:
            src_path = paths[idx]
            file_name = os.path.basename(src_path)
            dest_path = os.path.join(label_dir, file_name)
            shutil.copy2(src_path, dest_path)
            retained_paths_for_this_label.append(dest_path)
            retained_embeddings.append(embeddings[idx])
            
        retained_samples_dict[label] = retained_paths_for_this_label
        total_retained += len(retained_paths_for_this_label)
        
        if len(retained_embeddings) > 0:
            center = np.mean(retained_embeddings, axis=0)
            center = center / (np.linalg.norm(center) + 1e-8)
            cluster_centers.append(center)
    
    print(f"\n✅ 清洗完成！原始样本: {total_samples} -> 筛选后高质量样本: {total_retained}")
    print(f"📂 清洗后的样本已存放在: {os.path.abspath(output_dir)}")
    
    # 可选开启PCA可视化
    if total_samples > 0:
        plt_pca_vis(
            all_embeddings=np.array(all_embeddings),
            all_labels=all_labels,
            all_is_retained=all_is_retained,
            save_path=os.path.join(output_dir, "pca_iforest_visualization_3d.png")
        )
    
    return retained_samples_dict, np.array(cluster_centers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于IsolationForest的药品图像清洗工具，支持图文特征融合")
    parser.add_argument("input_dir", help="原始图片输入目录")
    parser.add_argument("--model", type=str, default="./saved_weights/medicine_embedder_best_rknn_opt.onnx", help="ONNX特征提取模型路径")
    parser.add_argument("--output_dir", type=str, default="filtered_dataset", help="清洗后样本输出目录")
    parser.add_argument("--contamination", type=float, default=0.15, help="每类异常样本剔除比例(0~0.3)")
    parser.add_argument("--text_emb_pkl", type=str, default=None, help="文本嵌入字典pkl文件路径，不填则仅使用图像特征")
    parser.add_argument("--alpha", type=float, default=0.7, help="图文融合时图像特征权重，文本权重=1-alpha")
    parser.add_argument("--img_size", type=int, default=320, help="模型输入分辨率")
    args = parser.parse_args()

    # 校验模型文件
    if not os.path.exists(args.model):
        print(f"❌ 未找到模型文件 {args.model}")
        sys.exit(1)
    
    # 加载文本嵌入字典（可选）
    text_emb_dict = None
    if args.text_emb_pkl is not None:
        if not os.path.exists(args.text_emb_pkl):
            print(f"❌ 文本嵌入文件不存在: {args.text_emb_pkl}")
            sys.exit(1)
        with open(args.text_emb_pkl, "rb") as f:
            text_emb_dict = pickle.load(f)
        print(f"✅ 加载文本嵌入字典成功，共 {len(text_emb_dict)} 条文本特征")

    # 初始化特征提取器
    embedder = ONNXMedicineEmbedder(onnx_model_path=args.model, target_size=args.img_size)
    
    # 执行清洗
    filtered_samples, centers = iforest_filter_samples(
        embedder=embedder,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        contamination=args.contamination,
        text_embedding_dict=text_emb_dict,
        alpha=args.alpha
    )
