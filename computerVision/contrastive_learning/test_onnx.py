import os
import sys
import random
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
import torch.nn.functional as F

# ==========================================
# ONNX 版特征提取器（接口与原 PyTorch 版完全对齐）
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
        """与训练时 test_transform 严格对齐：Resize + ToTensor (0-1归一化)"""
        # 尺寸对齐：双线性插值，与 torchvision.Resize 默认行为一致
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        # 转 float32 并归一化到 0-1，对应 ToTensor 操作
        img_np = np.array(img, dtype=np.float32) / 255.0
        # HWC -> CHW，符合 PyTorch/ONNX 输入格式
        img_np = img_np.transpose(2, 0, 1)
        # 增加 batch 维度
        return np.expand_dims(img_np, axis=0)

    def get_embedding(self, image_data_or_path):
        """对外接口与原 MedicineBoxEmbedder 完全一致，返回归一化后的特征向量"""
        if isinstance(image_data_or_path, str):
            img = Image.open(image_data_or_path).convert('RGB')
        else:
            img = image_data_or_path 
            
        input_tensor = self._preprocess(img)
        # ONNX 推理
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        embedding = outputs[0]  # shape: [1, 256]
        
        # 后处理补全 L2 归一化，与训练时 F.normalize 效果完全一致
        norm = np.linalg.norm(embedding, ord=2, axis=1, keepdims=True)
        embedding = embedding / (norm + 1e-8)
        
        return embedding

# ==========================================
# 以下函数与原脚本完全一致，未做任何修改
# ==========================================
def test_similarity(embedder, img_path1, img_path2):
    emb1 = embedder.get_embedding(img_path1)
    emb2 = embedder.get_embedding(img_path2)
    
    t1 = emb1.clone().detach() if isinstance(emb1, torch.Tensor) else torch.tensor(emb1)
    t2 = emb2.clone().detach() if isinstance(emb2, torch.Tensor) else torch.tensor(emb2)
    
    t1 = t1.view(-1)
    t2 = t2.view(-1)
    
    sim = F.cosine_similarity(t1, t2, dim=0).item()
    return sim

def evaluate_directory(embedder, directory_path):
    medicine_dict = {}
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    print(f"正在扫描目录: {directory_path} ...")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                path = os.path.join(root, file)
                
                file_stem = os.path.splitext(file)[0]
                parts = file_stem.split('_')
                
                if len(parts) >= 4:
                    medicine_name = parts[2]
                    if medicine_name not in medicine_dict:
                        medicine_dict[medicine_name] = []
                    medicine_dict[medicine_name].append(path)

    all_medicines = list(medicine_dict.keys())
    if len(all_medicines) < 2:
        print("错误: 目录中至少需要包含两种不同名称的药品才能进行对比测试。")
        return

    print("\n--- 相似度测试结果 ---")
    print(f"{'本图片药品名':<15} | {'随机异类图片名':<15} : 异类相似度(预期低) || {'同名样品相似度(预期高)':<15}")
    print("-" * 75)
    
    for med_name, paths in medicine_dict.items():
        for base_img_path in paths:
            # A. 负样本测试 (异类)
            other_meds = [m for m in all_medicines if m != med_name]
            random_diff_med = random.choice(other_meds)
            diff_med_img = random.choice(medicine_dict[random_diff_med])
            
            sim_negative = test_similarity(embedder, base_img_path, diff_med_img)
            
            # B. 正样本测试 (同类)
            other_same_med_imgs = [p for p in paths if p != base_img_path]
            
            if len(other_same_med_imgs) > 0:
                same_med_img = random.choice(other_same_med_imgs)
                sim_positive = test_similarity(embedder, base_img_path, same_med_img)
                pos_str = f"{sim_positive:.4f}"
            else:
                pos_str = "无样本-跳过"
                
            print(f"{med_name:<15} | {random_diff_med:<15} : {sim_negative:.4f}       || {pos_str}")

# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    # ========== 请修改为你的 ONNX 模型实际路径 ==========
    ONNX_MODEL_PATH = "./saved_weights/medicine_embedder_best_320_no_norm.onnx"
    
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "../../dataset/std/"
    
    if not os.path.exists(target_dir):
        print(f"指定的目录不存在: {target_dir}")
        sys.exit(1)

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ 未找到ONNX模型文件 {ONNX_MODEL_PATH}，请先运行导出脚本！")
        sys.exit(1)

    # 初始化 ONNX 特征提取器
    embedder = ONNXMedicineEmbedder(onnx_model_path=ONNX_MODEL_PATH, target_size=320)
    
    # 执行全目录评估
    evaluate_directory(embedder, target_dir)
