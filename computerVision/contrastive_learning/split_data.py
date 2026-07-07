import os
import random
import shutil
import argparse

def split_dataset(src_dir: str, train_dir: str = "./run_train", test_dir: str = "./run_test", 
                  train_ratio: float = 0.8, seed: int = 42, move: bool = False):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    
    all_files = []
    for fname in os.listdir(src_dir):
        fpath = os.path.join(src_dir, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(exts):
            all_files.append(fname)
    
    if not all_files:
        raise ValueError(f"源目录 {src_dir} 中未找到任何图片文件")
    
    total = len(all_files)
    print(f"[数据集划分] 找到图片总数: {total} 张")
    print(f"[数据集划分] 训练集占比: {train_ratio*100:.0f}%, 测试集占比: {(1-train_ratio)*100:.0f}%")
    
    random.seed(seed)
    random.shuffle(all_files)
    
    split_idx = int(total * train_ratio)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    
    print(f"[数据集划分] 训练集: {len(train_files)} 张, 测试集: {len(test_files)} 张")
    
    # 创建输出目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    file_op = shutil.move if move else shutil.copy2
    
    print("[数据集划分] 正在处理训练集...")
    for fname in train_files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(train_dir, fname)
        file_op(src_path, dst_path)
    
    # 复制/移动测试集
    print("[数据集划分] 正在处理测试集...")
    for fname in test_files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(test_dir, fname)
        file_op(src_path, dst_path)
    
    print("=" * 50)
    print("[数据集划分] 完成！")
    print(f"  训练集目录: {os.path.abspath(train_dir)} ({len(train_files)} 张)")
    print(f"  测试集目录: {os.path.abspath(test_dir)} ({len(test_files)} 张)")
    print(f"  操作模式: {'移动' if move else '复制'}")


# ===================== 命令行入口 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图片数据集随机划分工具（80%训练 / 20%测试）")
    parser.add_argument("--src", type=str, default="dataset", help="源图片目录")
    parser.add_argument("--train_dir", type=str, default="./run_train", help="训练集输出目录")
    parser.add_argument("--test_dir", type=str, default="./run_test", help="测试集输出目录")
    parser.add_argument("--ratio", type=float, default=0.8, help="训练集占比，默认 0.8")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--move", action="store_true", help="移动文件而非复制（默认复制，保留原文件）")
    
    args = parser.parse_args()
    
    try:
        split_dataset(
            src_dir=args.src,
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            train_ratio=args.ratio,
            seed=args.seed,
            move=args.move
        )
    except Exception as e:
        print(f"\n运行出错: {str(e)}")
        exit(1)
