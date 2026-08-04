# -*- coding:utf-8 -*-

import os
import sys
import math
import random
import argparse
import multiprocessing
import json

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
# 2. 特征提取网络 (保持结构绝对不变)
# ==========================================
class MultiScaleAttention(nn.Module):
    def __init__(self, channels):
        super(MultiScaleAttention, self).__init__()
        reduction = max(8, channels // 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
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

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1)).to(device)

        self.backbone.eval()
        dummy_input = torch.zeros(2, 3, img_size, img_size, dtype=torch.float32, device=device).contiguous()

        with torch.no_grad():
            features = self.backbone(dummy_input)

        c_in = features.shape[1]
        
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
        features = self.backbone(x)
        features = self.ms_attention(features)
        
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

        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(degrees=1, translate=(0.01, 0.01), scale=(0.98, 1.02)),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.0),
            transforms.ToTensor(),
        ])

    def gpu_augmentations(self, imgs):
        B, C, H, W = imgs.shape
        apply_noise = torch.rand(B, 1, 1, 1, device=self.device) < 0.1
        noise = torch.randn_like(imgs) * 0.005
        imgs = torch.where(apply_noise, torch.clamp(imgs + noise, 0., 1.), imgs)
        return imgs

    def _generate_mllm_prompt(self, med_name):
        """
        [提示词工程注入模块]
        直接读取离线配置好的专家级 MLLM 提示词 JSON 文件。
        """
        json_path = "mllm_prompts.json"
        
        if not hasattr(self, 'mllm_dict'):
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.mllm_dict = json.load(f)
                print(f"✅ 成功加载外部提示词库，包含 {len(self.mllm_dict)} 个类别的富语义增强。")
            else:
                print(f"⚠️ 未找到 {json_path}，将使用动态降级模板...")
                self.mllm_dict = {}

        if med_name in self.mllm_dict:
            return self.mllm_dict[med_name]
        else:
            prompt = (
                f"这是一个标号为【{med_name}】的临床药品包装盒。作为精确识别对象，它具有独特的制药厂品牌视觉设计。"
                f"该药盒表面的颜色分布、特定位置的排版格式、防伪标记、以及核心的规格容量文字"
                f"是区分它与其它近似甚至同品牌不同规格药盒的本质特征。模型需要重点关注盒面的这些边缘轮廓与文字区域的高频细节。"
            )
            return prompt

    # =========================================================================================
    # [集成添加]: MCMC Metropolis-Hastings 负样本采样器 (来自 _0805.py)
    # 作用：依据当前距离分布建立马尔可夫链，概率化捕获最具信息量的 Hard Negatives
    # =========================================================================================
    def _mcmc_sample_negatives(self, sim_mat, mask_neg, num_chains=3, steps=3, temperature=0.1):
        """
        利用张量化 Metropolis-Hastings 进行 MCMC 采样。
        返回一个新的 Mask，仅保留通过 MCMC 筛选出的“优质且难分”的负样本。
        """
        N = sim_mat.size(0)
        mcmc_mask = torch.zeros_like(mask_neg)
        
        # 仅让存在负样本的行参与 MCMC，防止类别单一导致概率崩溃
        valid_rows = mask_neg.sum(dim=1) > 0
        if not valid_rows.any():
            return mask_neg  # 容错降级
            
        # P(x) \propto exp(sim / T)：相似度越高，越难区分，采样的目标概率越大
        log_prob = sim_mat.clone().detach() / temperature
        log_prob[~mask_neg] = -1e4  # 屏蔽正样本和自身
        
        valid_mask = mask_neg[valid_rows]
        valid_log_prob = log_prob[valid_rows]
        
        # 初始状态：依据均匀概率随机挑选负样本 (num_chains代表每张图采样几个负样本)
        current_state = torch.multinomial(valid_mask.float(), num_chains, replacement=True)
        
        for _ in range(steps):
            # Propose (提议)：随机产生新的游走目标
            proposal_state = torch.multinomial(valid_mask.float(), num_chains, replacement=True)
            
            # 计算当前状态与提议状态的对数概率
            current_lp = torch.gather(valid_log_prob, 1, current_state)
            proposal_lp = torch.gather(valid_log_prob, 1, proposal_state)
            
            # 接受概率 (Acceptance Probability)
            accept_prob = torch.exp(proposal_lp - current_lp)
            rand_u = torch.rand_like(accept_prob)
            
            # 依照 MH 准则更新马尔可夫链
            accept = rand_u < accept_prob
            current_state = torch.where(accept, proposal_state, current_state)
            
        # 建立全新的 MCMC 过滤层
        valid_mcmc_mask = torch.zeros_like(valid_mask)
        valid_mcmc_mask.scatter_(1, current_state, True)
        
        mcmc_mask[valid_rows] = valid_mcmc_mask
        
        # 确保选出的样本严格属于原生负样本集合
        return mcmc_mask & mask_neg


    def train_model(self, save_dir='./company_saved_weights', epochs=1200, batch_size=128):
        m_per_class = 8
        dataset = MedicineClassificationDataset(self.db_dir, transform=self.train_transform, m_per_class=m_per_class)

        if len(dataset.images) == 0:
            print("❌ 错误：数据集中没有足够符合要求的图片。")
            return

        print("🚀 [多模态提示词引擎已启动] 正在生成富语义 Prompt 并预计算所有类别的 BERT 特征...")
        unique_meds = list(dataset.med2id.keys())

        # 核心注入点：短文本映射为富语义长文本
        rich_prompts = [self._generate_mllm_prompt(name) for name in unique_meds]

        max_label_id = max(dataset.med2id.values()) if dataset.med2id else 0
        self.cached_text_tensor = torch.zeros((max_label_id + 1, 768), device=self.device)

        # max_length 提升至 128，保障 MLLM 先验描述不被截断
        encoded = self.tokenizer(rich_prompts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
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

        criterion = losses.SupConLoss(temperature=0.01)
        lr = 3e-4
        weight_decay = 1e-4

        optim_params = list(filter(lambda p: p.requires_grad, self.extractor.parameters())) + \
                       list(self.text_projector.parameters())

        optimizer = torch.optim.AdamW(optim_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        self.extractor.train()
        self.text_projector.train()

        scaler = torch.amp.GradScaler('cuda')

        best_loss = float('inf')
        os.makedirs(save_dir, exist_ok=True)

        print(f"🔥 开始极限指标训练 | 核心算法：VLM + MCMC (Metropolis-Hastings) + LogSumExp Margin Constraint")
        for epoch in range(epochs):
            total_loss = 0.0
            valid_batches = 0

            for i, (imgs, labels, _) in enumerate(dataloader):
                imgs = imgs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(self.device, non_blocking=True)
                
                imgs_clean = imgs.clone()
                imgs = self.gpu_augmentations(imgs)
                
                text_cls = self.cached_text_tensor[labels]
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    img_embeddings = self.extractor(imgs)
                    text_embeddings = self.text_projector(text_cls)
                    img_embeddings_clean = self.extractor(imgs_clean)

                    joint_embeddings = torch.cat([img_embeddings, text_embeddings], dim=0)
                    joint_labels = torch.cat([labels, labels], dim=0)
                    loss_supcon = criterion(joint_embeddings.float(), joint_labels)

                    loss_inv = 1.0 - F.cosine_similarity(img_embeddings.float(), img_embeddings_clean.float(), dim=-1).mean()

                    sim_mat_img = torch.matmul(img_embeddings.float(), img_embeddings.float().T)
                    sim_mat_txt = torch.matmul(text_embeddings.float(), text_embeddings.float().T)
                    
                    mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
                    mask_neg = ~mask_pos
                    mask_pos.fill_diagonal_(False)

                    gamma = 32.0         
                    pos_margin = 0.95    
                    neg_margin = -0.05   
                    zeros = torch.zeros(1, device=self.device).float()

                    if mask_pos.any():
                        pos_sims = sim_mat_img[mask_pos]
                        pos_args = torch.cat([zeros, -gamma * (pos_sims.float() - pos_margin)])
                        loss_pos_img = (1.0 / gamma) * torch.logsumexp(pos_args, dim=0)
                    else:
                        loss_pos_img = torch.tensor(0.0, device=self.device)

                    # ====================================================================
                    # [修改点]: 应用 MCMC 动态采样替换原有的全局硬惩罚
                    # ====================================================================
                    if mask_neg.any():
                        # 使用 MCMC 从海量异类组合中猎取高质量目标
                        mcmc_mask_neg_img = self._mcmc_sample_negatives(sim_mat_img, mask_neg, num_chains=3)
                        mcmc_mask_neg_txt = self._mcmc_sample_negatives(sim_mat_txt, mask_neg, num_chains=3)
                        
                        neg_sims = sim_mat_img[mcmc_mask_neg_img]
                        neg_args = torch.cat([zeros, gamma * (neg_sims.float() - neg_margin)])
                        loss_neg_img = (1.0 / gamma) * torch.logsumexp(neg_args, dim=0)
                        
                        txt_neg_sims = sim_mat_txt[mcmc_mask_neg_txt]
                        txt_neg_args = torch.cat([zeros, gamma * (txt_neg_sims.float() - neg_margin)])
                        loss_neg_txt = (1.0 / gamma) * torch.logsumexp(txt_neg_args, dim=0)
                    else:
                        loss_neg_img = torch.tensor(0.0, device=self.device)
                        loss_neg_txt = torch.tensor(0.0, device=self.device)

                    loss_strict = 15.0 * loss_pos_img + 25.0 * loss_neg_img + 25.0 * loss_neg_txt

                    unique_labels = torch.unique(labels)
                    num_unique = len(unique_labels)
                    
                    if num_unique >= 2:
                        rep_img_list, rep_txt_list = [], []
                        for ul in unique_labels:
                            mask = (labels == ul)
                            rep_img_list.append(img_embeddings[mask].mean(dim=0))
                            rep_txt_list.append(text_embeddings[mask].mean(dim=0))
                        rep_img = torch.stack(rep_img_list)
                        rep_txt = torch.stack(rep_txt_list)

                        pairs_i, pairs_j = [], []
                        for ii in range(num_unique):
                            for jj in range(ii + 1, num_unique):
                                pairs_i.append(ii)
                                pairs_j.append(jj)
                                
                        if len(pairs_i) > 0:
                            i_idx = torch.tensor(pairs_i, device=labels.device)
                            j_idx = torch.tensor(pairs_j, device=labels.device)
                            img_diff = rep_img[i_idx] - rep_img[j_idx]
                            txt_diff = rep_txt[i_idx] - rep_txt[j_idx]
                            loss_diff = 1.0 - F.cosine_similarity(img_diff.float(), txt_diff.float(), dim=-1).mean()
                            
                            loss = loss_supcon + 0.1 * loss_diff + 5.0 * loss_inv + loss_strict
                        else:
                            loss = loss_supcon + 5.0 * loss_inv + loss_strict
                    else:
                        loss = loss_supcon + 5.0 * loss_inv + loss_strict

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
    parser.add_argument("db_dir", type=str, nargs="?", default="company_train_0804", help="Path to database directory")
    parser.add_argument("--model_file", type=str, default="/usr/rfzn/xiangyang/model/pt/best_n_722_600_320_16.pt")
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--infer_img", type=str, default="")
    parser.add_argument("--infer_weight", type=str, default="./company_saved_weights/medicine_embedder_best_rknn_opt.pth")
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
            embedder.train_model(save_dir="./company_saved_weights", epochs=1500, batch_size=256)
            onnx_file = "./company_saved_weights/medicine_embedder_best_rknn_opt.onnx"
            if os.path.exists(onnx_file):
                print("\n" + "="*60)
                print("已跳过 RKNN 转换流程。你可以随后自行转换。")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"运行出错: {e}")
