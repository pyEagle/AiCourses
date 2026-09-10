# -*- coding: utf-8 -*-

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorch_metric_learning.samplers import MPerClassSampler

@dataclass
class CosmoConfig:
    embedding_dim = 192
    hidden_dim = 256
    subcenter_k = 3
    arc_s = 64.0
    arc_m = 0.35
    
    init_neg_margin = 0.50
    init_pos_margin = 0.50
    target_neg_margin = 0.10
    target_pos_margin = 0.70
    
    density_offset = 0.3
    density_scale = 0.1
    density_margin_penalty = 0.15
    
    w_arc = 1.0
    w_inv = 1.0
    w_pos = 5.0
    w_neg = 5.0
    w_uniform = 2.0
    
    tau = 0.1
    k_nn = 8
    
    lr_base = 1e-5
    lr_head = 3e-4

    weight_decay = 5e-4
    eta_min = 1e-6
    epochs = 10
    m_per_class = 4
    
    train_batch_size = 32
    val_batch_size = 16
    num_train_samples = 256
    num_val_samples = 64
    num_train_classes = 8
    num_val_classes = 4
    
    eval_abs_thresh = 0.65
    eval_margin_thresh = 0.10
    eval_chunk_size = 2048
    eval_bins = 10000
    fusion_topk = 3
    fusion_weight = 0.5


class SphericalMeanPooling(nn.Module):
    def __init__(self):
        super(SphericalMeanPooling, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.mean(dim=0), p=2, dim=-1)


class SubCenterArcFace(nn.Module):
    def __init__(self, in_features: int, out_features: int, K: int = 3, s: float = 64.0, m: float = 0.35):
        super(SubCenterArcFace, self).__init__()
        self.K = K
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features * K, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        input_f, weight_f = input.float(), self.weight.float()
        cosine_all = F.linear(F.normalize(input_f), F.normalize(weight_f))
        cosine_all = cosine_all.view(-1, self.out_features, self.K)
        cosine, _ = cosine_all.max(dim=2)
        
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class CosmologicalStructureLoss(nn.Module):
    def __init__(self, config: CosmoConfig):
        super(CosmologicalStructureLoss, self).__init__()
        self.cfg = config

    def forward(self, emb_anchor: torch.Tensor, emb_perturbed: torch.Tensor, 
                labels: torch.Tensor, centers: torch.Tensor, progress: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        e_a, e_p = emb_anchor.float(), emb_perturbed.float()
        device = e_a.device
        
        g_t = (1.0 - math.cos(math.pi * progress)) / 2.0
        c_neg_margin = self.cfg.init_neg_margin - (self.cfg.init_neg_margin - self.cfg.target_neg_margin) * g_t
        c_pos_margin = self.cfg.init_pos_margin + (self.cfg.target_pos_margin - self.cfg.init_pos_margin) * g_t

        loss_inv = 1.0 - F.cosine_similarity(e_a, e_p, dim=-1).mean()

        norm_centers = F.normalize(centers.float(), dim=-1).view(-1, self.cfg.subcenter_k, self.cfg.embedding_dim)
        batch_centers = norm_centers[labels]
        sim_to_sub = torch.bmm(e_a.unsqueeze(1), batch_centers.transpose(1, 2)).squeeze(1)
        s_i_star, _ = sim_to_sub.max(dim=1)
        loss_pos = F.relu(c_pos_margin - s_i_star).mean()

        sim_mat = torch.matmul(e_a, e_p.T)
        mask_neg = labels.unsqueeze(0) != labels.unsqueeze(1)
        has_neg = mask_neg.sum(dim=1) > 0
        
        if has_neg.any():
            sim_mat_neg = sim_mat.masked_fill(~mask_neg, float('-inf'))
            k_eff = min(self.cfg.k_nn, sim_mat_neg.size(1))
            knn_sim, _ = torch.topk(sim_mat_neg, k=k_eff, dim=1)
            valid_knn = torch.isfinite(knn_sim)
            
            knn_sum = knn_sim.masked_fill(~valid_knn, 0.0).sum(dim=1)
            knn_count = valid_knn.sum(dim=1).clamp_min(1)
            local_density = torch.sigmoid((knn_sum / knn_count - self.cfg.density_offset) / self.cfg.density_scale)
            adaptive_neg_margin = c_neg_margin - self.cfg.density_margin_penalty * local_density.unsqueeze(1)

            neg_weights = torch.zeros_like(sim_mat_neg)
            neg_weights[has_neg] = F.softmax(sim_mat_neg[has_neg] / self.cfg.tau, dim=1) 
            
            repulsion_forces = F.relu(sim_mat_neg - adaptive_neg_margin)
            loss_neg = torch.sum(neg_weights * repulsion_forces, dim=1)[has_neg].mean()
        else:
            loss_neg = torch.tensor(0.0, device=device)

        sq_dist = 2.0 - 2.0 * torch.matmul(e_a, e_a.T)
        neg_sq_dist = sq_dist[mask_neg]
        if neg_sq_dist.numel() > 0:
            loss_uniform = torch.logsumexp(-2.0 * neg_sq_dist, dim=0) - math.log(max(1, neg_sq_dist.numel()))
        else:
            loss_uniform = torch.tensor(0.0, device=device)

        total_loss = (self.cfg.w_inv * loss_inv + 
                      self.cfg.w_pos * loss_pos + 
                      self.cfg.w_neg * loss_neg + 
                      self.cfg.w_uniform * loss_uniform)

        metrics = {"L_inv": loss_inv.item(), "L_pos": loss_pos.item(), 
                   "L_neg": loss_neg.item(), "L_uni": loss_uniform.item()}
        return total_loss, metrics


class CosmoEvaluator:
    @staticmethod
    def compute_eer_hist(embeddings: torch.Tensor, labels: torch.Tensor, 
                         device: torch.device, chunk_size: int = 2048, bins: int = 10000) -> Tuple[float, float]:
        N = embeddings.size(0)
        if N < 2: return 1.0, 0.0

        embeddings = F.normalize(embeddings, p=2, dim=-1).float().to(device)
        labels = labels.to(device)
        hist_pos = torch.zeros(bins, device=device)
        hist_neg = torch.zeros(bins, device=device)
        
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            emb_chunk = embeddings[i:end]
            sim_chunk = torch.matmul(emb_chunk, embeddings.T) 
            
            label_chunk = labels[i:end]
            same = label_chunk.unsqueeze(1) == labels.unsqueeze(0)
            eye_mask = torch.zeros_like(same, dtype=torch.bool)
            eye_mask[:, i:end] = torch.eye(end - i, device=device, dtype=torch.bool)
            
            mask_pos = same & ~eye_mask
            mask_neg = ~same
            
            p_sims = sim_chunk[mask_pos]
            n_sims = sim_chunk[mask_neg]
            
            if p_sims.numel() > 0:
                p_idx = torch.clamp(((p_sims + 1.0) / 2.0 * (bins - 1)).long(), 0, bins - 1)
                hist_pos.scatter_add_(0, p_idx, torch.ones_like(p_idx, dtype=torch.float))
            if n_sims.numel() > 0:
                n_idx = torch.clamp(((n_sims + 1.0) / 2.0 * (bins - 1)).long(), 0, bins - 1)
                hist_neg.scatter_add_(0, n_idx, torch.ones_like(n_idx, dtype=torch.float))

        total_pos, total_neg = hist_pos.sum(), hist_neg.sum()
        if total_pos == 0 or total_neg == 0: return 1.0, 0.0
            
        cum_pos = torch.cumsum(hist_pos, dim=0)
        cum_neg = torch.cumsum(hist_neg, dim=0)
        
        frr = cum_pos / total_pos
        far = 1.0 - (cum_neg / total_neg)
        
        diff = torch.abs(far - frr)
        best_idx = torch.argmin(diff)
        eer = ((far[best_idx] + frr[best_idx]) / 2.0).item()
        best_th = -1.0 + 2.0 * (best_idx.item() / (bins - 1))
        
        return eer, best_th

    @staticmethod
    def retrieve_1_to_n(query_emb: Union[np.ndarray, torch.Tensor], gallery_embs: Union[np.ndarray, torch.Tensor],
                        gallery_ids: List[str], abs_thresh: float = 0.65, margin_thresh: float = 0.10,
                        topk: int = 3, fusion_weight: float = 0.5) -> Dict:
        if isinstance(query_emb, np.ndarray): query_emb = torch.from_numpy(query_emb)
        if isinstance(gallery_embs, np.ndarray): gallery_embs = torch.from_numpy(gallery_embs)

        query_emb = F.normalize(query_emb.view(1, -1), p=2, dim=-1).float()
        gallery_embs = F.normalize(gallery_embs, p=2, dim=-1).float()
        scores = torch.matmul(query_emb, gallery_embs.T).squeeze(0)

        spk_scores_list: Dict[str, List[float]] = {}
        for score, spk_id in zip(scores, gallery_ids):
            spk_scores_list.setdefault(spk_id, []).append(score.item())

        id_scores: Dict[str, float] = {}
        for spk_id, s_list in spk_scores_list.items():
            s_list = sorted(s_list, reverse=True)
            k = min(topk, len(s_list))
            topk_mean = sum(s_list[:k]) / k
            id_scores[spk_id] = fusion_weight * s_list[0] + (1.0 - fusion_weight) * topk_mean

        if not id_scores: return {"match_id": "UNKNOWN", "accepted": False}

        sorted_cands = sorted(id_scores.items(), key=lambda x: x[1], reverse=True)
        top1_id, top1_score = sorted_cands[0]
        
        if len(sorted_cands) == 1:
            accepted = top1_score >= abs_thresh
            return {"match_id": top1_id if accepted else "UNKNOWN", "score": top1_score, "margin": 1.0, "accepted": accepted}

        top2_id, top2_score = sorted_cands[1]
        margin = top1_score - top2_score
        accepted = (top1_score >= abs_thresh) and (margin >= margin_thresh)
        return {
            "match_id": top1_id if accepted else "UNKNOWN",
            "score": round(top1_score, 4),
            "margin": round(margin, 4),
            "accepted": accepted,
            "runner_up_id": top2_id,
        }


class CosmoTrainer:
    def __init__(self, model: nn.Module, num_classes: int, config: Optional[CosmoConfig] = None, device: Optional[torch.device] = None):
        self.cfg = config or CosmoConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)

        if not hasattr(self.model, 'get_embedding'):
            self.model.get_embedding = self.model.forward
        
        self.classifier = SubCenterArcFace(
            in_features=self.cfg.embedding_dim, out_features=num_classes, 
            K=self.cfg.subcenter_k, s=self.cfg.arc_s, m=self.cfg.arc_m
        ).to(self.device)
        
        self.structure_loss = CosmologicalStructureLoss(self.cfg)
        self.arc_criterion = nn.CrossEntropyLoss()
        
        optim_params = [
            {'params': self.model.parameters(), 'lr': self.cfg.lr_base},
            {'params': self.classifier.parameters(), 'lr': self.cfg.lr_head}
        ]
        self.optimizer = torch.optim.AdamW(optim_params, weight_decay=self.cfg.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.epochs, eta_min=self.cfg.eta_min
        )
        self.scaler = torch.amp.GradScaler('cuda' if self.device.type == 'cuda' else 'cpu', enabled=(self.device.type == 'cuda'))

    def train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int, use_forward: bool = True) -> Dict[str, float]:
        self.model.train()
        self.classifier.train()
        total_loss = 0.0
        progress = epoch / max(1, total_epochs - 1)
        
        for batch in dataloader:
            x_anchor, x_perturbed, labels = batch
            x_anchor, x_perturbed, labels = x_anchor.to(self.device), x_perturbed.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(self.device.type):
                if use_forward:
                    raw_anchor = self.model(x_anchor)
                    raw_perturbed = self.model(x_perturbed)
                else:
                    raw_anchor = self.model.get_embedding(x_anchor)
                    raw_perturbed = self.model.get_embedding(x_perturbed)
                
                emb_anchor = F.normalize(raw_anchor, p=2, dim=-1)
                emb_perturbed = F.normalize(raw_perturbed, p=2, dim=-1)
                
                loss_struct, _ = self.structure_loss(emb_anchor, emb_perturbed, labels, self.classifier.weight, progress)
                logits_arc = self.classifier(emb_perturbed, labels)
                loss_arc = self.arc_criterion(logits_arc, labels)
                
                loss = self.cfg.w_arc * loss_arc + loss_struct

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        self.scheduler.step()
        current_lr = self.optimizer.param_groups[1]['lr']
        return {"train_loss": total_loss / max(1, len(dataloader)), "lr": current_lr}

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, use_forward: bool = True) -> Tuple[float, float]:
        self.model.eval()
        embs, labels_list = [], []
        pooler = SphericalMeanPooling()
        
        for batch in val_loader:
            x_val, labels = batch[0].to(self.device), batch[1].to(self.device)
            
            if x_val.dim() > 2: 
                x_in = x_val.view(-1, *x_val.shape[2:])
                raw = self.model(x_in) if use_forward else self.model.get_embedding(x_in)
                raw = raw.view(x_val.size(0), x_val.size(1), -1)
                emb = pooler(raw.permute(1, 0, 2)) 
            else:
                raw = self.model(x_val) if use_forward else self.model.get_embedding(x_val)
                emb = F.normalize(raw, dim=-1)
                
            embs.append(emb)
            labels_list.append(labels)
            
        return CosmoEvaluator.compute_eer_hist(
            torch.cat(embs, dim=0), torch.cat(labels_list, dim=0), self.device,
            chunk_size=self.cfg.eval_chunk_size, bins=self.cfg.eval_bins
        )


class HypersphericalMockDataset(Dataset):
    def __init__(self, num_samples: int = 256, dim: int = 192, num_classes: int = 8, label_offset: int = 0):
        self.num_samples = num_samples
        self.dim = dim
        self.labels = [(i % num_classes) + label_offset for i in range(num_samples)]
        
        torch.manual_seed(42)
        centers = torch.randn(num_classes + label_offset, dim)
        self.centers = F.normalize(centers, p=2, dim=-1).numpy()

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        label = self.labels[idx]
        center = self.centers[label]
        x_clean = center + np.random.randn(self.dim).astype(np.float32) * 0.15
        x_perturbed = x_clean + np.random.randn(self.dim).astype(np.float32) * 0.35
        return torch.from_numpy(x_clean), torch.from_numpy(x_perturbed), label


class MockBackbone(nn.Module):
    def __init__(self, in_dim: int = 192, hidden_dim: int = 256, out_dim: int = 192):
        super(MockBackbone, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class SpeakerDirDataset(Dataset):
    def __init__(self, data_dir: str, dim: int = 192):
        self.data_dir = data_dir
        self.dim = dim
        
        self.files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        
        # 提取文件名结构并映射为ID类别
        self.spk_ids = [f.split('_')[2] if len(f.split('_')) >= 3 else "unknown" for f in self.files]
        self.unique_spks = sorted(list(set(self.spk_ids)))
        self.spk2idx = {spk: i for i, spk in enumerate(self.unique_spks)}
        self.labels = [self.spk2idx[spk] for spk in self.spk_ids]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        # 模拟特征（实际项目中替换为真实声音特征的提取）
        torch.manual_seed(hash(self.files[idx]) % 1000000)
        base_feat = torch.randn(self.dim)
        
        x_anchor = base_feat + torch.randn(self.dim) * 0.05
        x_perturbed = base_feat + torch.randn(self.dim) * 0.20
        
        return x_anchor, x_perturbed, label


def main():
    parser = argparse.ArgumentParser(description="Speaker Voice Embeddings Training")
    parser.add_argument("--data_dir", type=str, default="data/audio", help="Path to the training dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save the best model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--max_batch_size", type=int, default=32, help="Maximum training batch size allowable")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        return

    dataset = SpeakerDirDataset(args.data_dir, dim=CosmoConfig.embedding_dim)
    num_classes = len(dataset.unique_spks)
    total_samples = len(dataset)
    
    print(f"Loaded dataset from: {args.data_dir}")
    print(f"Found {total_samples} files, {num_classes} speakers: {dataset.unique_spks}")
    
    if total_samples == 0:
        print("Error: No files found for training.")
        return

    min_samples_per_class = min([dataset.labels.count(c) for c in set(dataset.labels)])
    m_per_class = min(2, min_samples_per_class)
    
    adaptive_batch = min(args.max_batch_size, total_samples)
    should_drop_last = (total_samples > adaptive_batch)
    
    if m_per_class > 1:
        max_sampler_batch = m_per_class * num_classes
        adaptive_batch = min(adaptive_batch, max_sampler_batch)
        
        adaptive_batch = max(m_per_class, (adaptive_batch // m_per_class) * m_per_class)
        
        sampler = MPerClassSampler(dataset.labels, m_per_class, batch_size=adaptive_batch, length_before_new_iter=total_samples)
        dataloader = DataLoader(dataset, batch_size=adaptive_batch, sampler=sampler, drop_last=should_drop_last)
        print(f"[*] Auto Config: MPerClassSampler active (m={m_per_class}). Adaptive Batch Size = {adaptive_batch}")
    else:
        print("Warning: Some classes have only 1 sample. Falling back to Random Shuffling.")
        dataloader = DataLoader(dataset, batch_size=adaptive_batch, shuffle=True, drop_last=should_drop_last)
        print(f"[*] Auto Config: Random Shuffle active. Adaptive Batch Size = {adaptive_batch}")

    cfg = CosmoConfig()
    cfg.epochs = args.epochs
    
    model = MockBackbone(in_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim, out_dim=cfg.embedding_dim)
    trainer = CosmoTrainer(model=model, num_classes=num_classes, config=cfg)

    print("\n--- Starting Training ---")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(dataloader, epoch, args.epochs, use_forward=True)
        print(f"Epoch [{epoch:02d}/{args.epochs}] | Loss: {metrics['train_loss']:.4f} | LR: {metrics['lr']:.6f}")
        
        if metrics['train_loss'] < best_loss:
            best_loss = metrics['train_loss']
            best_model_path = os.path.join(args.output_dir, "best_speaker_model.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'speakers_map': dataset.spk2idx  
            }, best_model_path)
            
            print(f"  [+] Saved new best model to -> {best_model_path}")
            
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()

