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
import torchvision.transforms.functional as TF  

from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler


if multiprocessing.current_process().name != 'MainProcess':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["ORT_LOGGING_LEVEL"] = "4"
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.modules['onnxruntime'] = None
else:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ORT_LOGGING_LEVEL"] = "4"
    os.environ["OMP_NUM_THREADS"] = "4"


class RandomRightAngleRotation:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            return img
        return img.rotate(angle, expand=True)

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return TF.pad(image, padding, 0, 'constant')


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


class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)


class MultiScaleAttention(nn.Module):
    def __init__(self, channels):
        super(MultiScaleAttention, self).__init__()
        reduction = max(8, channels // 8)

        self.gem_pool = GeMPooling()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.gem_pool(x))
        max_out = self.fc(self.max_pool(x))
        c_attn = self.sigmoid_channel(avg_out + max_out)
        x_ca = x * c_attn

        avg_map = torch.mean(x_ca, dim=1, keepdim=True)
        max_map, _ = torch.max(x_ca, dim=1, keepdim=True)
        s_map = torch.cat([avg_map, max_map], dim=1)
        s_attn = self.sigmoid_spatial(self.conv_spatial(s_map))
        
        return x_ca * s_attn


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

        self.backbone.eval()
        dummy_input = torch.zeros(2, 3, img_size, img_size, dtype=torch.float32, device=device).contiguous()

        with torch.no_grad():
            feats_shallow = None
            feat = dummy_input
            for i, layer in enumerate(self.backbone):
                feat = layer(feat)
                if i == 4:
                    feats_shallow = feat 
            feats_deep = feat            

        c_shallow = feats_shallow.shape[1]
        c_deep = feats_deep.shape[1]
        c_in = c_deep
        
        self.shallow_proj = nn.Sequential(
            nn.Conv2d(c_shallow, c_deep, 1, bias=False),
            nn.BatchNorm2d(c_deep),
            nn.ReLU(inplace=True)
        ).to(device)

        self.gem_pool = GeMPooling().to(device)
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1)).to(device)
        self.ms_attention = MultiScaleAttention(c_in).to(device)
        in_channels = c_in * 2

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
        feat_shallow = None
        feat = x
        for i, layer in enumerate(self.backbone):
            feat = layer(feat)
            if i == 4:
                feat_shallow = feat

        shallow_aligned = self.shallow_proj(feat_shallow)
        shallow_aligned = F.adaptive_max_pool2d(shallow_aligned, output_size=feat.shape[-2:])
        feat = feat + shallow_aligned
        features = self.ms_attention(feat)
        
        avg_f = self.gem_pool(features)
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


class MedicineBoxEmbedder:
    def __init__(self, db_dir, model_file, img_size=640):
        self.db_dir = db_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.empty_cache()
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            print("🚀 已启用 cuDNN 加速与 TF32 支持，极致训练性能准备就绪！")
            _ = nn.Conv2d(3, 3, 3).to(self.device)(torch.zeros(1, 3, 10, 10).to(self.device))
            print("✅ CUDA 上下文初始化完成。")

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

        # ==========================================
        # [调参重点 1] 保护细粒度特征的数据增强
        # ==========================================
        # 针对 1ml、5ml 这种微弱的文字特征，过度裁剪和擦除会导致毁灭性灾难。
        self.train_transform = transforms.Compose([
            RandomRightAngleRotation(), 
            SquarePad(),  
            transforms.RandomRotation(degrees=180, fill=0), 
            # 1. 大幅收紧裁剪下限 (0.75 -> 0.95)，几乎只做微小抖动，确保“5ml”等字样绝不被切出画面外
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.95, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # 2. 降低透视形变，防止文字扭曲导致不可读
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), 
            transforms.RandomGrayscale(p=0.10), 
            transforms.RandomAutocontrast(p=0.2), 
            # 3. 极度克制高斯模糊，文字一旦模糊，1 和 5 就会混淆
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)), 
            transforms.ToTensor(),
            # 4. 取消大面积遮挡，仅使用极小概率的微小遮挡，防止“命门”被盖住
            transforms.RandomErasing(p=0.05, scale=(0.01, 0.02), value=0), 
        ])

    def gpu_augmentations(self, imgs):
        B, C, H, W = imgs.shape
        apply_noise = torch.rand(B, 1, 1, 1, device=self.device) < 0.2
        noise = torch.randn_like(imgs) * 0.01
        imgs = torch.where(apply_noise, torch.clamp(imgs + noise, 0., 1.), imgs)
        return imgs

    def train_model(self, save_dir='./saved_weights', epochs=1200, batch_size=64):
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

        num_valid_classes = len(set(dataset.labels))
        safe_batch_size = min(batch_size, num_valid_classes * m_per_class)
        actual_batch_size = safe_batch_size
        
        sampler = MPerClassSampler(dataset.labels, m=m_per_class, batch_size=actual_batch_size, length_before_new_iter=len(dataset))

        num_workers = 16
        dataloader = DataLoader(
            dataset, batch_size=actual_batch_size, sampler=sampler,
            drop_last=True, num_workers=num_workers, pin_memory=True,
            prefetch_factor=4, persistent_workers=True
        )

        lr = 2.0e-4  
        weight_decay = 5e-4 

        optim_params = list(filter(lambda p: p.requires_grad, self.extractor.parameters())) + \
                       list(self.text_projector.parameters())

        optimizer = torch.optim.AdamW(optim_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        self.extractor.train()
        self.text_projector.train()
        scaler = torch.amp.GradScaler('cuda')
        best_loss = float('inf')
        os.makedirs(save_dir, exist_ok=True)

        print(f"🔥 开始细粒度极限训练 | 超大 Scale + CosFace Margin + GeM")
        for epoch in range(epochs):
            total_loss = 0.0
            valid_batches = 0

            for i, (imgs, labels, _) in enumerate(dataloader):
                imgs = imgs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(self.device, non_blocking=True)
                
                imgs_clean = imgs.clone()
                imgs = self.gpu_augmentations(imgs)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    img_embeddings = self.extractor(imgs)
                    img_embeddings_clean = self.extractor(imgs_clean)
                    
                    k_rot = random.choice([1, 2, 3]) 
                    imgs_rot = torch.rot90(imgs_clean, k=k_rot, dims=[2, 3])
                    img_embeddings_rot = self.extractor(imgs_rot)
                    loss_rot = 1.0 - F.cosine_similarity(img_embeddings_clean, img_embeddings_rot, dim=-1).mean()
                    
                    unique_labels, inverse_indices = torch.unique(labels, return_inverse=True)
                    unique_text_cls = self.cached_text_tensor[unique_labels]
                    unique_text_emb = self.text_projector(unique_text_cls)
                    
                    # ==========================================
                    # [调参重点 2] 极限细粒度区分：大幅拉升尺度和间距
                    # ==========================================
                    # 强行要求不同型号的注射器之间留出极宽的护城河
                    m_cos = 0.50        # (原0.35) 极高标准，强制产生清晰界限
                    logit_scale = 64.0  # (原30.0) 工业级 ArcFace/CosFace 标准值，让微小像素差异导致梯度的巨幅变化
                    
                    # 图像推向对应文本
                    sim_i2t = torch.matmul(img_embeddings, unique_text_emb.T)
                    one_hot_i2t = torch.zeros_like(sim_i2t).scatter_(1, inverse_indices.unsqueeze(1), 1.0)
                    sim_i2t_margin = sim_i2t - one_hot_i2t * m_cos
                    logits_i2t = sim_i2t_margin * logit_scale
                    loss_i2t = F.cross_entropy(logits_i2t, inverse_indices)
                    
                    # 文本中心绝对排斥
                    sim_t2t = torch.matmul(unique_text_emb, unique_text_emb.T)
                    one_hot_t2t = torch.eye(len(unique_labels), device=self.device)
                    sim_t2t_margin = sim_t2t - one_hot_t2t * m_cos
                    logits_t2t = sim_t2t_margin * logit_scale
                    labels_txt = torch.arange(len(unique_labels), device=self.device)
                    loss_t2t = F.cross_entropy(logits_t2t, labels_txt)
                    
                    # 绝对死区 Hinge Loss
                    sim_mat_img = torch.matmul(img_embeddings, img_embeddings.T)
                    
                    # A. 负样本（异类）处理：形成距离黑洞
                    mask_neg = labels.unsqueeze(0) != labels.unsqueeze(1)
                    if mask_neg.any():
                        sim_mat_neg = sim_mat_img.clone()
                        sim_mat_neg[~mask_neg] = -1.0 
                        hardest_neg_sims, _ = sim_mat_neg.max(dim=1)
                        # 【下压异类界限】要求异类相似度必须跌破 -0.20，确保负数
                        loss_hard_neg = F.relu(hardest_neg_sims - (-0.20)).mean()
                    else:
                        loss_hard_neg = torch.tensor(0.0, device=self.device)
                        
                    # B. 正样本（同类）处理：形成高密度星系
                    mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
                    mask_pos.fill_diagonal_(False) 
                    
                    if mask_pos.any():
                        sim_mat_pos = sim_mat_img.clone()
                        sim_mat_pos[~mask_pos] = 1.0 
                        hardest_pos_sims = sim_mat_pos.min(dim=1)[0]
                        # 【上抬同类界限】要求同类星系聚集密度必须高达 0.90，远远保障测试时大于0.65
                        loss_hard_pos = F.relu(0.90 - hardest_pos_sims).mean()
                    else:
                        loss_hard_pos = torch.tensor(0.0, device=self.device)

                    loss_inv = 1.0 - F.cosine_similarity(img_embeddings, img_embeddings_clean, dim=-1).mean()
                    
                    w_i2t = 4.0
                    w_t2t = 1.0
                    w_neg = 5.0
                    w_pos = 5.0
                    w_inv = 0.5 
                    w_rot = 0.5 

                    loss = (w_i2t * loss_i2t + 
                            w_t2t * loss_t2t + 
                            w_neg * loss_hard_neg + 
                            w_pos * loss_hard_pos + 
                            w_inv * loss_inv + 
                            w_rot * loss_rot) 

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
                print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

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
            self.extractor.load_state_dict(checkpoint["extractor"], strict=False)
            if "text_projector" in checkpoint:
                self.text_projector.load_state_dict(checkpoint["text_projector"], strict=False)
                self.text_projector.eval()
        else:
            self.extractor.load_state_dict(checkpoint, strict=False)

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
            SquarePad(),  
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
    parser.add_argument("db_dir", type=str, nargs="?", default="train_0902", help="Path to database directory")
    parser.add_argument("--model_file", type=str, default="best_n_0901_3000_16.pt")
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--infer_img", type=str, default="")
    parser.add_argument("--infer_weight", type=str, default="./saved_weights/embedder_best_rknn_opt.pth")
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
            embedder.train_model(save_dir="./saved_weights", epochs=2000, batch_size=128)
            onnx_file = "./saved_weights/embedder_best_rknn_opt.onnx"
            if os.path.exists(onnx_file):
                print("\n" + "="*60)
                print("已跳过 RKNN 转换流程。你可以随后转换。")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"运行出错: {e}")
