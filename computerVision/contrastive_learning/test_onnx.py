import os
import sys
import random
import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
import torch.nn.functional as F


# ==========================================
# ONNX 版特征提取器
# ==========================================
class ONNXMedicineEmbedder:
    def __init__(self, onnx_model_path, target_size=320):
        self.target_size = target_size

        # 加载 ONNX 模型，自动优先使用 GPU，fallback 到 CPU
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

        # 无需手动 L2 归一化
        # cl02 的 PyTorch forward 函数中已包含 F.normalize，导出的 ONNX 已自带归一化

        return embedding


# ==========================================
# 评估逻辑
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

                # 🔧 修正点 1：与 cl02 的 Dataset 解析逻辑严格对齐
                #    cl02 原始代码: med_name = parts[2][-7:]
                #    取第 3 段（index=2）的最后 7 个字符作为药品标识
                if len(parts) >= 3:
                    medicine_name = parts[2][-7:]
                    if medicine_name not in medicine_dict:
                        medicine_dict[medicine_name] = []
                    medicine_dict[medicine_name].append(path)

    all_medicines = list(medicine_dict.keys())
    if len(all_medicines) < 2:
        print("错误: 目录中至少需要包含两种不同名称的药品才能进行对比测试。")
        return

    # ---- 统计信息 ----
    total_images = sum(len(v) for v in medicine_dict.values())
    print(f"✅ 共扫描到 {total_images} 张图片，{len(all_medicines)} 个药品类别\n")

    print("--- 相似度测试结果 ---")
    print(f"{'本图片药品名':<15} | {'随机异类图片名':<15} : 异类相似度(预期低) || {'同名样品相似度(预期高)':<15}")
    print("-" * 80)

    # ---- 用于汇总统计 ----
    pos_sims = []
    neg_sims = []

    for med_name, paths in medicine_dict.items():
        for base_img_path in paths:
            # A. 负样本测试 (异类)
            other_meds = [m for m in all_medicines if m != med_name]
            random_diff_med = random.choice(other_meds)
            diff_med_img = random.choice(medicine_dict[random_diff_med])

            sim_negative = test_similarity(embedder, base_img_path, diff_med_img)
            neg_sims.append(sim_negative)

            # B. 正样本测试 (同类)
            other_same_med_imgs = [p for p in paths if p != base_img_path]

            if len(other_same_med_imgs) > 0:
                same_med_img = random.choice(other_same_med_imgs)
                sim_positive = test_similarity(embedder, base_img_path, same_med_img)
                pos_sims.append(sim_positive)
                pos_str = f"{sim_positive:.4f}"
            else:
                pos_str = "无样本-跳过"

            print(f"{med_name:<15} | {random_diff_med:<15} : {sim_negative:.4f}        || {pos_str}")

    # ---- 汇总报告 ----
    print("\n" + "=" * 80)
    print("📊 汇总统计")
    print("=" * 80)
    if neg_sims:
        print(f"  异类相似度 (越低越好)  → 均值: {np.mean(neg_sims):.4f}  "
              f"最大: {np.max(neg_sims):.4f}  最小: {np.min(neg_sims):.4f}")
    if pos_sims:
        print(f"  同类相似度 (越高越好)  → 均值: {np.mean(pos_sims):.4f}  "
              f"最大: {np.max(pos_sims):.4f}  最小: {np.min(pos_sims):.4f}")
    if pos_sims and neg_sims:
        gap = np.mean(pos_sims) - np.mean(neg_sims)
        print(f"  类间间距 (越大越好)    → {gap:.4f}")
    print("=" * 80)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX 模型相似度评估 (对齐 cl02_xiangyang)")

    # 🔧 修正点 2：ONNX 模型路径与 cl02 的 save_dir="./saved_weights" 对齐
    parser.add_argument("--onnx_model", type=str,
                        default="./saved_weights/medicine_embedder_best_rknn_opt.onnx",
                        help="ONNX 模型文件路径 (默认: ./saved_weights/medicine_embedder_best_rknn_opt.onnx)")

    parser.add_argument("--target_dir", type=str,
                        default="./train722",
                        help="待评估的图片目录 (默认: ./train722)")

    # 🔧 修正点 3：输入尺寸参数化，与 cl02 的 --img_size 对齐
    parser.add_argument("--img_size", type=int,
                        default=320,
                        help="模型输入尺寸，需与训练时一致 (默认: 320)")

    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        print(f"❌ 指定的目录不存在: {args.target_dir}")
        sys.exit(1)

    if not os.path.exists(args.onnx_model):
        print(f"❌ 未找到 ONNX 模型文件: {args.onnx_model}")
        print("💡 请先运行 cl02_train_auto_320_xiangyang.py 完成训练和导出！")
        sys.exit(1)

    print(f"📦 加载 ONNX 模型: {args.onnx_model}")
    print(f"📁 评估目录: {args.target_dir}")
    print(f"📐 输入尺寸: {args.img_size}")
    print()

    # 初始化 ONNX 特征提取器
    embedder = ONNXMedicineEmbedder(onnx_model_path=args.onnx_model, target_size=args.img_size)

    # 执行全目录评估
    evaluate_directory(embedder, args.target_dir)
