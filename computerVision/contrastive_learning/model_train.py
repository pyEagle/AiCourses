# -*- coding:utf-8 -*-

import os
import sys
import math
import random
import argparse
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import Counter

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses
from pytorch_metric_learning.samplers import MPerClassSampler


# =========================================================================
# 阻止子进程扫描硬件
# =========================================================================
if multiprocessing.current_process().name != 'MainProcess':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["ORT_LOGGING_LEVEL"] = "4"
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.modules['onnxruntime'] = None
else:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ORT_LOGGING_LEVEL"] = "4"
    os.environ["OMP_NUM_THREADS"] = "4"

# ==========================================
# 1. Dataset
# ==========================================
class MedicineClassificationDataset(Dataset):
    def __init__(self, db_dir, transform=None, m_per_class=16):
        self.db_dir = db_dir
        self.transform = transform

        all_files = [f for f in os.listdir(db_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        temp_images = []
        temp_labels = []
        temp_texts = []

        med_names = set()
        for file in all_files:
            parts = file.split('_')
            if len(parts) >= 3:
                med_names.add(parts[2][-7:])

        self.med2id = {name: idx for idx, name in enumerate(sorted(list(med_names)))}

        for file in all_files:
            parts = file.split('_')
            if len(parts) >= 3:
                med_name = parts[2][-7:]
                temp_images.append(file)
                temp_labels.append(self.med2id[med_name])
                temp_texts.append(med_name)

        label_counts = Counter(temp_labels)
        # 自适应过滤：保证留下的类别至少有 m_per_class 张图
        valid_labels = {k for k, v in label_counts.items() if v >= m_per_class}

        self.images = []
        self.labels = []
        self.texts = []
        for img, lbl, txt in zip(temp_images, temp_labels, temp_texts):
            if lbl in valid_labels:
                self.images.append(img)
                self.labels.append(lbl)
                self.texts.append(txt)

        print(f"✅ 过滤后剩余 {len(self.images)} 张有效图片，共包含 {len(valid_labels)} 个多图类别。")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label = self.labels[idx]
        text = self.texts[idx]

        img_path = os.path.join(self.db_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label, text


# ==========================================
# 2. 特征提取网络
# ==========================================
class YOLOv8NeckExtractor(nn.Module):
    def __init__(self, model_file, device, img_size=320):
        super().__init__()

        print("🛠️ 正在使用原生 PyTorch 加载 YOLO 权重，避开底层探针...")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ultralytics import YOLO
            ckpt = torch.load(model_file, map_location='cpu')

        self.backbone = nn.Sequential(*list(ckpt['model'].model.children())[:10]).float().to(device)

        for i, module in enumerate(self.backbone):
            if i < 5:
                for param in module.parameters():
                    param.requires_grad = False

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1)).to(device)

        self.backbone.eval()
        dummy_input = torch.zeros(2, 3, img_size, img_size, dtype=torch.float32, device=device).contiguous()

        with torch.no_grad():
            features = self.backbone(dummy_input)

        in_channels = features.shape[1] * 2

        self.embedding_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        ).to(device)

    def forward(self, x):
        features = self.backbone(x)
        avg_f = self.avg_pool(features)
        max_f = self.max_pool(features)
        base_emb = torch.cat([avg_f, max_f], dim=1)

        final_emb = self.embedding_head(base_emb)
        return F.normalize(final_emb, p=2, dim=1)


class TextProjector(nn.Module):
    def __init__(self, in_features=768, out_features=256):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        out = self.proj(x)
        return F.normalize(out, p=2, dim=1)


# ==========================================
# 3. 训练器
# ==========================================
class MedicineBoxEmbedder:
    def __init__(self, db_dir, model_file, img_size=320):
        self.db_dir = db_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.empty_cache()
            print("🛡️ 重新启动底层护盾：接管纯 CUDA 上下文并彻底禁用 cuDNN...")
            torch.backends.cudnn.enabled = False
            _ = nn.Conv2d(3, 3, 3).to(self.device)(torch.zeros(1, 3, 10, 10).to(self.device))
            print("✅ 护盾部署完毕：避开崩溃区，随时可以极速训练！")

        self.extractor = YOLOv8NeckExtractor(model_file, self.device, img_size).to(memory_format=torch.channels_last)
        print("✅ 图像网络构建完成，已完全移入:", self.device)

        print("📚 正在加载中文 BERT 模型...")
        try:
            from transformers import BertTokenizer, BertModel
        except ImportError:
            print("❌ 请先安装 transformers 库: pip install transformers")
            sys.exit(1)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.text_encoder = BertModel.from_pretrained('bert-base-chinese').to(self.device)
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_projector = TextProjector(in_features=768, out_features=256).to(self.device)

        resize_size = int(img_size * 1.0625)

        # 数据增强（保持原样）
        self.train_transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=12),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.1)
            ], p=0.3),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, hue=0.02),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.15),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.05, 0.15), value=0)
        ])

        self._grid_cache = {}

    def gpu_augmentations(self, imgs):
        # 此函数原样保留，提供反光、光晕和传感器噪声增强
        B, C, H, W = imgs.shape
        apply_noise = torch.rand(B, 1, 1, 1, device=self.device) < 0.15
        noise = torch.randn_like(imgs) * 0.04
        imgs = torch.where(apply_noise, torch.clamp(imgs + noise, 0., 1.), imgs)

        apply_opt = torch.rand(B, 1, 1, 1, device=self.device) < 0.2
        cache_key = (H, W)
        if cache_key not in self._grid_cache:
            y = torch.arange(0, H, device=self.device, dtype=torch.float32)
            x = torch.arange(0, W, device=self.device, dtype=torch.float32)
            y, x = torch.meshgrid(y, x, indexing='ij')
            self._grid_cache[cache_key] = (x.unsqueeze(0), y.unsqueeze(0))

        x, y = self._grid_cache[cache_key]
        center_x = (torch.rand(B, 1, 1, device=self.device) * 0.8 + 0.1) * W
        center_y = (torch.rand(B, 1, 1, device=self.device) * 0.8 + 0.1) * H
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        is_glare = torch.rand(B, 1, 1, device=self.device) < 0.5
        sigma_x = torch.rand(B, 1, 1, device=self.device) * 30 + 10
        sigma_y = torch.rand(B, 1, 1, device=self.device) * 30 + 10
        glare_mask = torch.exp(-((x - center_x)**2 / (2 * sigma_x**2) + (y - center_y)**2 / (2 * sigma_y**2)))
        glare_intensity = torch.rand(B, 1, 1, device=self.device) * 0.4 + 0.4
        sigma = torch.rand(B, 1, 1, device=self.device) * 90 + 60
        halo_mask = torch.exp(-(dist_sq) / (2 * sigma**2))
        halo_intensity = torch.rand(B, 1, 1, device=self.device) * 0.3 + 0.2
        mask = torch.where(is_glare, glare_mask, halo_mask).unsqueeze(1)
        intensity = torch.where(is_glare, glare_intensity, halo_intensity).unsqueeze(1)
        imgs = torch.where(apply_opt, torch.clamp(imgs + mask * intensity, 0., 1.), imgs)
        return imgs

    def train_model(self, save_dir='./saved_weights', epochs=1200, batch_size=128):
        # 每类采样数，保持 8 不变
        m_per_class = 8
        dataset = MedicineClassificationDataset(self.db_dir, transform=self.train_transform, m_per_class=m_per_class)

        if len(dataset.images) == 0:
            print("❌ 错误：数据集中没有足够符合要求的图片。")
            return

        print("🚀 正在预计算所有类别的 BERT 特征（只需算 1 次，解放 GPU）...")
        unique_meds = list(dataset.med2id.keys())

        max_label_id = max(dataset.med2id.values()) if dataset.med2id else 0
        self.cached_text_tensor = torch.zeros((max_label_id + 1, 768), device=self.device)

        encoded = self.tokenizer(unique_meds, padding=True, truncation=True, max_length=32, return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.text_encoder(**encoded)
            cls_feats = out.last_hidden_state[:, 0, :]

        for name, feat in zip(unique_meds, cls_feats):
            label_id = dataset.med2id[name]
            self.cached_text_tensor[label_id] = feat

        del self.text_encoder
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ BERT 特征矩阵预计算并常驻 GPU 缓存完毕！")

        num_valid_classes = len(set(dataset.labels))

        safe_batch_size = min(batch_size, num_valid_classes * m_per_class)

        if safe_batch_size < batch_size:
            print(f"\n⚠️ 数据集受限提示：当前有效类别数 {num_valid_classes} * m({m_per_class}) = {num_valid_classes * m_per_class} < 设定值 {batch_size}")
            print(f"👉 已自动将 Batch Size 动态下调至安全极限：{safe_batch_size} (未来增加图片类别数会自动提速)")

        actual_batch_size = safe_batch_size

        sampler = MPerClassSampler(dataset.labels, m=m_per_class, batch_size=actual_batch_size, length_before_new_iter=len(dataset))

        num_workers = 16
        dataloader = DataLoader(
            dataset, batch_size=actual_batch_size, sampler=sampler,
            drop_last=True, num_workers=num_workers, pin_memory=True,
            prefetch_factor=4, persistent_workers=True
        )

        # ========== 参数调整 1：温度从 0.07 提高到 0.12 ==========
        criterion = losses.SupConLoss(temperature=0.12)

        # ========== 参数调整 2：学习率从 5e-5 提高到 8e-5 ==========
        lr = 8e-5
        weight_decay = 1e-4   # 保持原样

        optim_params = list(filter(lambda p: p.requires_grad, self.extractor.parameters())) + \
                       list(self.text_projector.parameters())

        optimizer = torch.optim.AdamW(optim_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        self.extractor.train()
        self.text_projector.train()

        scaler = torch.amp.GradScaler('cuda')

        best_loss = float('inf')
        os.makedirs(save_dir, exist_ok=True)

        # ========== 参数调整 3：语义差分对齐损失权重从 0.5 增加到 1.0 ==========
        diff_loss_weight = 1.0

        print(f"🔥 开始图文对齐精细化训练 | 实际 Batch Size: {actual_batch_size} | Temp: 0.12 | LR: {lr} | WD: {weight_decay} | DiffLoss: {diff_loss_weight}")
        for epoch in range(epochs):
            total_loss = 0.0
            valid_batches = 0

            for i, (imgs, labels, _) in enumerate(dataloader):
                imgs = imgs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(self.device, non_blocking=True)
                
                # =========================================================
                # 【关键修正】：启用 GPU 数据增强，针对反光、光晕及微小晃动
                # =========================================================
                imgs = self.gpu_augmentations(imgs)
                
                text_cls = self.cached_text_tensor[labels]
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    img_embeddings = self.extractor(imgs)
                    text_embeddings = self.text_projector(text_cls)

                    # [基础] 监督对比损失
                    joint_embeddings = torch.cat([img_embeddings, text_embeddings], dim=0)
                    joint_labels = torch.cat([labels, labels], dim=0)
                    loss_supcon = criterion(joint_embeddings.float(), joint_labels)

                    # ============================================================
                    # [改动] 语义差分对齐：使用类内平均嵌入作为代表，构造异类对
                    # ============================================================
                    unique_labels = torch.unique(labels)
                    num_unique = len(unique_labels)
                    if num_unique >= 2:
                        rep_img_list = []
                        rep_txt_list = []
                        for ul in unique_labels:
                            mask = (labels == ul)
                            # 改为类内平均，更稳定
                            rep_img = img_embeddings[mask].mean(dim=0)   # [256]
                            rep_txt = text_embeddings[mask].mean(dim=0)  # [256]
                            rep_img_list.append(rep_img)
                            rep_txt_list.append(rep_txt)
                        rep_img = torch.stack(rep_img_list)   # [num_unique, 256]
                        rep_txt = torch.stack(rep_txt_list)

                        # 生成所有异类对 (i, j), i < j
                        pairs_i, pairs_j = [], []
                        for ii in range(num_unique):
                            for jj in range(ii + 1, num_unique):
                                pairs_i.append(ii)
                                pairs_j.append(jj)
                        if len(pairs_i) > 0:
                            i_idx = torch.tensor(pairs_i, device=labels.device)
                            j_idx = torch.tensor(pairs_j, device=labels.device)
                            img_diff = rep_img[i_idx] - rep_img[j_idx]   # 类别i代表 - 类别j代表
                            txt_diff = rep_txt[i_idx] - rep_txt[j_idx]   # 对应文本差
                            loss_diff = 1.0 - F.cosine_similarity(img_diff.float(), txt_diff.float(), dim=-1).mean()
                            loss = loss_supcon + diff_loss_weight * loss_diff
                        else:
                            loss = loss_supcon
                    else:
                        loss = loss_supcon

                loss_val = loss.item()
                if loss_val == 0 or loss_val != loss_val:
                    continue

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss_val
                valid_batches += 1

            avg_loss = (total_loss / valid_batches) if valid_batches > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | Multi-Modal Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

            if avg_loss < best_loss and valid_batches > 0:
                best_loss = avg_loss
                best_path = os.path.join(save_dir, "medicine_embedder_best_rknn_opt.pth")
                onnx_path = os.path.join(save_dir, "medicine_embedder_best_rknn_opt.onnx")
                self.save_model(best_path, silent=True)
                self.export_onnx(onnx_path, silent=True)

            scheduler.step()

        print(f"\n🎉 训练完成！最佳抗量化多模态模型已保存至 {save_dir}")

    def save_model(self, save_path, silent=False):
        checkpoint = {
            "extractor": self.extractor.state_dict(),
            "text_projector": self.text_projector.state_dict()
        }
        torch.save(checkpoint, save_path)
        if not silent:
            print(f"✅ 图像提取器与文本投影器权重已保存至: {save_path}")

    def load_model(self, weight_path):
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"找不到权重文件: {weight_path}")

        checkpoint = torch.load(weight_path, map_location=self.device)
        if "extractor" in checkpoint:
            self.extractor.load_state_dict(checkpoint["extractor"])
            if "text_projector" in checkpoint:
                self.text_projector.load_state_dict(checkpoint["text_projector"])
                self.text_projector.eval()
        else:
            self.extractor.load_state_dict(checkpoint)

        self.extractor.eval()
        print(f"✅ 成功加载精调模型权重: {weight_path}")

    def get_embedding(self, image_source):
        self.extractor.eval()

        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"图像路径不存在: {image_source}")
            img = Image.open(image_source).convert('RGB')
        elif isinstance(image_source, Image.Image):
            img = image_source.convert('RGB')
        elif isinstance(image_source, np.ndarray):
            img = Image.fromarray(image_source).convert('RGB')
        else:
            raise TypeError("不支持的图像格式，请传入图片路径(str)、PIL.Image 或 numpy.ndarray")

        infer_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        img_tensor = infer_transform(img).unsqueeze(0).to(self.device, non_blocking=True)
        img_tensor = img_tensor.to(memory_format=torch.channels_last)

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    embedding = self.extractor(img_tensor)
            else:
                embedding = self.extractor(img_tensor)

        return embedding.cpu().numpy()[0]

    def export_onnx(self, onnx_path, silent=False):
        self.extractor.eval()
        dummy_input = torch.randn(1, 3, self.img_size, self.img_size, device=self.device).to(memory_format=torch.channels_last)
        import warnings
        try:
            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    torch.onnx.export(
                        self.extractor,
                        dummy_input,
                        onnx_path,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['images'],
                        output_names=['embeddings']
                    )
            if not silent:
                print(f"✅ 自动同步导出 ONNX 模型至: {onnx_path}")
        except Exception as e:
            print(f"⚠️ ONNX 导出失败，但不影响训练继续进行: {e}")
        self.extractor.train()

    def export_rknn(self, onnx_path, rknn_path, target_platform='rk3588'):
        try:
            from rknn.api import RKNN
        except ImportError:
            print("\n❌ 错误: 未检测到 rknn-toolkit2 环境。")
            return

        print(f"\n🔄 正在构建 RKNN 模型 (目标平台: {target_platform})...")
        rknn = RKNN(verbose=False)
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=target_platform)
        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            print("❌ RKNN 加载 ONNX 模型失败！")
            return
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            print("❌ RKNN 构建模型失败！")
            return
        ret = rknn.export_rknn(rknn_path)
        if ret == 0:
            print(f"✅ 恭喜！RKNN 模型成功导出至: {rknn_path}")
        else:
            print("❌ RKNN 导出文件失败！")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Medicine Box Embedder Training / Inference")
    parser.add_argument("db_dir", type=str, nargs="?", default="train729", help="Path to database directory")
    parser.add_argument("--model_file", type=str, default="/usr/rfzn/xiangyang/model/pt/best_n_722_600_320_16.pt")
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--infer_img", type=str, default="")
    parser.add_argument("--infer_weight", type=str, default="./saved_weights/medicine_embedder_best_rknn_opt.pth")
    args = parser.parse_args()

    try:
        embedder = MedicineBoxEmbedder(db_dir=args.db_dir, model_file=args.model_file, img_size=args.img_size)
        if args.infer_img:
            print("\n" + "="*50)
            print("🔍 进入纯推理测试模式...")
            embedder.load_model(args.infer_weight)
            emb = embedder.get_embedding(args.infer_img)
            print(f"✅ 获取到 Embedding, 维度形状: {emb.shape}")
            print(f"👀 前 10 个特征值预览: \n{emb[:10]}")
            print("="*50 + "\n")
        else:
            # ========== 参数调整 4：训练轮数从 500 增加到 800 ==========
            embedder.train_model(save_dir="./saved_weights", epochs=1500, batch_size=256)
            onnx_file = "./saved_weights/medicine_embedder_best_rknn_opt.onnx"
            if os.path.exists(onnx_file):
                print("\n" + "="*60)
                print("已跳过 RKNN 转换流程。你可以随后自行转换。")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"运行出错: {e}")
