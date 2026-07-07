import os
import random
import multiprocessing
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import sys

from pytorch_metric_learning import losses
from pytorch_metric_learning.samplers import MPerClassSampler

# ==========================================
# 0. 自定义高斯噪声 (降低了强度和概率)
# ==========================================
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.02, p=0.2): # std 从 0.05 降到 0.02，概率降到 0.2
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return torch.clamp(tensor + noise, 0., 1.)
        return tensor

# ==========================================
# 1. Dataset
# ==========================================
class MedicineClassificationDataset(Dataset):
    def __init__(self, db_dir, transform=None):
        self.db_dir = db_dir
        self.transform = transform
        
        all_files = [f for f in os.listdir(db_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        temp_images = []
        temp_labels = []
        
        med_names = set()
        for file in all_files:
            parts = file.split('_')
            if len(parts) >= 3:
                med_names.add(parts[2])
                
        self.med2id = {name: idx for idx, name in enumerate(sorted(list(med_names)))}
        
        for file in all_files:
            parts = file.split('_')
            if len(parts) >= 3:
                med_name = parts[2]
                temp_images.append(file)
                temp_labels.append(self.med2id[med_name])

        label_counts = Counter(temp_labels)
        valid_labels = {k for k, v in label_counts.items() if v >= 2}
        
        self.images = []
        self.labels = []
        for img, lbl in zip(temp_images, temp_labels):
            if lbl in valid_labels:
                self.images.append(img)
                self.labels.append(lbl)
                
        print(f"过滤后剩余 {len(self.images)} 张有效图片，共包含 {len(valid_labels)} 个多图类别。")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label = self.labels[idx]
        img_path = os.path.join(self.db_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# ==========================================
# 2. 特征提取网络
# ==========================================
class YOLOv8NeckExtractor(nn.Module):
    def __init__(self, model_file):
        super().__init__()
        yolo = YOLO(model_file)
        
        self.core_model = yolo.model 
        self.backbone_neck = self.core_model.model 
        
        for i, module in enumerate(self.backbone_neck):
            if i < 5: 
                for param in module.parameters():
                    param.requires_grad = False
        
        self.features = None
        def hook_fn(module, input, output):
            self.features = output
            
        target_layer_idx = 9 
        self.hook_handle = self.backbone_neck[target_layer_idx].register_forward_hook(hook_fn)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.embedding_head = nn.Sequential(
            nn.Flatten(1),
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256) 
        )

    def forward(self, x):
        _ = self.core_model(x)  
        avg_f = self.avg_pool(self.features)
        max_f = self.max_pool(self.features)
        base_emb = torch.cat([avg_f, max_f], dim=1) 
        
        final_emb = self.embedding_head(base_emb)
        return F.normalize(final_emb, p=2, dim=1)

# ==========================================
# 3. 终极训练器 (修正版数据增强)
# ==========================================
class MedicineBoxEmbedder:
    def __init__(self, db_dir, model_file):
        self.db_dir = db_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = YOLOv8NeckExtractor(model_file).to(self.device)
        
        self.train_transform = transforms.Compose([
            transforms.Resize((340, 340)),                 
            transforms.RandomCrop((320, 320)),             
            transforms.RandomApply([
                transforms.RandomRotation(degrees=10)      # 角度再收敛到 10 度
            ], p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2               # 移除 Saturation，绝对保护色彩区分度
            ),                                             
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)) # 降低模糊核大小
            ], p=0.2),                                     # 概率降为 20%
            transforms.ToTensor(),
            AddGaussianNoise(mean=0.0, std=0.02, p=0.2),   # 概率降为 20%
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), value=0) # 遮挡概率降为 30%
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((320, 320)), 
            transforms.ToTensor()
        ])

    def get_embedding(self, image_data_or_path):
        self.extractor.eval()
        if isinstance(image_data_or_path, str):
            img = Image.open(image_data_or_path).convert('RGB')
        else:
            img = image_data_or_path 
            
        img_tensor = self.test_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.extractor(img_tensor)
        return embedding.cpu().numpy()

    def train_model(self, save_dir='./saved_weights', epochs=500, batch_size=32):
        dataset = MedicineClassificationDataset(self.db_dir, transform=self.train_transform)
        
        if len(dataset.images) == 0:
            print("错误：数据集中没有足够符合要求的图片(每个类别至少2张)。")
            return

        sampler = MPerClassSampler(dataset.labels, m=2, batch_size=batch_size, length_before_new_iter=len(dataset))
        
        num_workers = min(16, multiprocessing.cpu_count()) 
        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            drop_last=True, num_workers=num_workers, pin_memory=True 
        )

        criterion = losses.SupConLoss(temperature=0.07)
        
        lr = 1e-4
        optim_params = filter(lambda p: p.requires_grad, self.extractor.parameters())
        optimizer = torch.optim.AdamW(optim_params, lr=lr, weight_decay=1e-4)
        # 延长 Cosine 周期到 500
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        scaler = torch.amp.GradScaler('cuda', enabled=self.device.type == 'cuda')
        self.extractor.train()

        best_loss = float('inf')
        os.makedirs(save_dir, exist_ok=True)

        print(f"🔥 开始精细化长程训练 | Epochs: {epochs} | Batch Size: {batch_size}")
        
        dummy_input = torch.zeros(2, 3, 320, 320).to(self.device)
        _ = self.extractor(dummy_input)
        
        for epoch in range(epochs):
            total_loss = 0.0
            valid_batches = 0
            
            for i, (imgs, labels) in enumerate(dataloader):
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                    embeddings = self.extractor(imgs)
                    loss = criterion(embeddings, labels)
                
                if torch.isnan(loss) or loss.item() == 0:
                    continue
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                valid_batches += 1
                
            avg_loss = (total_loss / valid_batches) if valid_batches > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']
            
            # 每 10 轮打印一次，防止日志刷屏
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | SupCon Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
            
            if avg_loss < best_loss and valid_batches > 0:
                best_loss = avg_loss
                best_path = os.path.join(save_dir, "medicine_embedder_best.pth")
                self.save_model(best_path, silent=True)
                
            scheduler.step()
            
        print(f"\n🎉 训练完成！最佳模型已保存至 {save_dir}")

    def save_model(self, save_path, silent=False):
        torch.save(self.extractor.state_dict(), save_path)
        if not silent:
            print(f"✅ 模型已保存至: {save_path}")

    def load_model(self, load_path):
        self.extractor.load_state_dict(torch.load(load_path, map_location=self.device))
        self.extractor.eval()

if __name__ == "__main__":
    DB_DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else "../../dataset/std/"
    #YOLO_MODEL_PATH = "best_final.pt" 
    #YOLO_MODEL_PATH = "best_n_7.1.pt" 
    YOLO_MODEL_PATH = "best_n_7.4_800_320_8.pt" 
    
    try:
        embedder = MedicineBoxEmbedder(db_dir=DB_DIRECTORY, model_file=YOLO_MODEL_PATH)
        embedder.train_model(save_dir="./saved_weights", epochs=2000, batch_size=16)
    except Exception as e:
        import traceback;traceback.print_exc()
        print(f"运行出错: {e}")
