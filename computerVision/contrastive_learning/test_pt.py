import os
import sys
import random
import torch
import torch.nn.functional as F

from cl02_train import MedicineBoxEmbedder 

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

if __name__ == "__main__":
    # 请确保路径与你的环境一致
    YOLO_MODEL = "best_n_7.1.pt"
    WEIGHT_PATH = "./saved_weights/medicine_embedder_best.pth"
    
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "../../dataset/std/"
    
    if not os.path.exists(target_dir):
        print(f"指定的目录不存在: {target_dir}")
        sys.exit(1)

    embedder = MedicineBoxEmbedder(db_dir=target_dir, model_file=YOLO_MODEL)
    
    if os.path.exists(WEIGHT_PATH):
        embedder.load_model(WEIGHT_PATH)
        evaluate_directory(embedder, target_dir)
    else:
        print(f"❌ 未找到权重文件 {WEIGHT_PATH}，请先运行训练脚本！")
