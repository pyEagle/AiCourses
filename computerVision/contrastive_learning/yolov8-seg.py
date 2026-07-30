# -*- coding:utf-8 -*-

import os
import sys

# ============================================
# 关键修复：必须在所有 import 之前设置环境变量
# ============================================

# 1. 强制使用确定性 cuDNN 算法
os.environ['CUDNN_DETERMINISTIC'] = '1'

# 2. 关闭 Ultralytics 在线检查
os.environ['ULTRALYTICS_OFFLINE'] = 'True'

# 3. 限制 CUDA 内存分配策略（避免碎片化）
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 4. 设置 CUDA 启动为阻塞模式（获取更详细错误）
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 训练时不需要阻塞

import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


def setup_gpu():
    """配置 GPU 环境"""
    print("=" * 50)
    print("GPU 环境检查:")
    
    # 清理可能残留的 GPU 缓存
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 关键配置
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False  
        torch.backends.cudnn.deterministic = True
        
        # 测试 cuDNN 是否正常
        try:
            x = torch.randn(1, 3, 64, 64).cuda()
            conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
            y = conv(x)
            print("  ✓ cuDNN 初始化成功")
        except Exception as e:
            print(f"  ✗ cuDNN 测试失败: {e}")
            print("  尝试降级方案: 禁用 cuDNN benchmark")
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            # 再次测试
            try:
                y = conv(x)
                print("  ✓ cuDNN 降级方案成功")
            except:
                print("  ⚠ cuDNN 完全禁用，使用 PyTorch 原生实现")
                torch.backends.cudnn.enabled = False
    
    print("=" * 50)
    return True


def print_class_names(model, source="模型"):
    """打印模型类别映射"""
    if model.names:
        print(f"\n{source} 的类别映射:")
        for class_id, class_name in model.names.items():
            print(f"  {class_id}: {class_name}")
    else:
        print(f"\n⚠ {source} 未找到类别名称")


def train():
    # 1. 配置 GPU
    setup_gpu()
    
    # 2. 加载预训练分割模型
    print("\n正在加载模型...")
    model_path = '/home/rfzn/rfzn/smart_car_camer/xinxiang/Desktop/new_model/runs/segment/output/yolov8_seg_retrain/weights/best_m_7.10_3000_16.pt'
    model = YOLO(model_path)
    
    # 3. 读取数据集配置文件
    data_yaml_path = 'coco_seg.yaml'
    print(f"数据集配置: {data_yaml_path}")
    
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    
    # 打印类别信息
    print_class_names(model, source="预训练模型")
    
    # 4. 开始训练
    print("\n开始训练...")
    results = model.train(
        data=data_yaml_path,
        epochs=3000,
        batch=16,            
        imgsz=640,          
        device='0',
        workers=2,          
        project='output',
        name='yolov8_seg_retrain_robust_v2', # 换个名字避免覆盖
        exist_ok=True,
        
        # =======================================================
        # 核心修改区：针对“抽屉外漏检”的高级数据增强策略 (做最小调整)
        # =======================================================
        copy_paste=0.4,     
        mixup=0.2,          
        mosaic=1.0,         
        scale=0.7,          # 【微调】0.6 -> 0.7 适应抽屉外手持时距离镜头的尺度变化
        translate=0.3,      # 【微调】0.2 -> 0.3 允许物体更多地出现在画面边缘（离开抽屉中心）
        degrees=30.0,       # 【微调】15.0 -> 30.0 手拿着物体倾斜角度比平放更大
        perspective=0.0005, # 【新增】加入极微量的透视形变，抵抗抽屉内外视角差异
        erasing=0.3,        # 【新增】30%概率随机擦除部分区域，强制模型克服手部遮挡和塑料袋反光带来的特征缺失
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, 
        
        # 训练策略优化
        mask_ratio=2,
        overlap_mask=True,
        cos_lr=True,
        close_mosaic=20,    
        patience=200,       
        weight_decay=0.0005 
    )
    
    # 5. 验证并输出指标
    print("\n正在验证...")
    metrics = model.val()
    print("\n【验证指标】")
    print(f"  mAP50 (box): {metrics.box.map50:.4f}")
    print(f"  mAP50-95 (box): {metrics.box.map:.4f}")
    print(f"  seg mAP50: {metrics.seg.map50:.4f}")
    print(f"  seg mAP50-95: {metrics.seg.map:.4f}")


def predict():
    # 配置 GPU
    setup_gpu()
    
    # 1. 加载训练好的最佳模型
    model_path = 'output/yolov8_seg_retrain_robust_v2/weights/best.pt'
    if not Path(model_path).exists():
        print(f"✗ 模型文件不存在: {model_path}")
        print("请先完成训练，或修改 model_path 为实际路径")
        return
    
    model = YOLO(model_path)
    
    # 2. 打印模型的类别映射
    print_class_names(model, source=f"模型({model_path})")
    
    # 3. 执行推理
    results = model.predict(
        source='test_images',
        imgsz=640,
        conf=0.25,     # 【关键微调】0.4 -> 0.25。抽屉外物体由于失去背景加持，置信度通常在0.2~0.35之间，降低阈值可直接打捞漏检
        iou=0.6,       # 【微调】配合降低的conf，稍微提高NMS阈值防止同一物体出现多个重复框
        save=True,
        project='output',
        name='predict_seg_output',
        exist_ok=True,
    )
    
    # 4. 打印每张图片的检测结果
    print("\n【推理结果】")
    for r in results:
        if r.masks is not None:
            names = [model.names[int(c)] for c in r.boxes.cls]
            print(f"  {Path(r.path).name}: {len(r.masks)} 个目标, 类别: {names}")
        else:
            print(f"  {Path(r.path).name}: 未检测到目标")


def export():
    # 配置 GPU
    setup_gpu()
    
    # 1. 加载训练好的最佳模型
    model_path = 'output/yolov8_seg_retrain_robust_v2/weights/best.pt'
    if not Path(model_path).exists():
        print(f"✗ 模型文件不存在: {model_path}")
        return
    
    model = YOLO(model_path)
    
    # 2. 打印当前模型类别
    print_class_names(model, source=f"导出前的模型({model_path})")
    
    # 3. 导出 ONNX
    model.export(
        format='onnx',
        simplify=True,
        imgsz=640,
        opset=12
    )
    print(f"\n✅ ONNX 导出完成，文件位于: {model_path.replace('.pt', '.onnx')}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'export'],
                        help='选择运行模式: train/predict/export')
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            train()
        elif args.mode == 'predict':
            predict()
        elif args.mode == 'export':
            export()
    except RuntimeError as e:
        if "cuDNN" in str(e) or "CUDNN_STATUS" in str(e):
            print("\n" + "=" * 50)
            print("⚠ cuDNN 错误再次出现！执行终极修复方案：")
            print("=" * 50)
            print("\n请在命令行执行以下命令找到 cuDNN 库路径：")
            print("find ~/anaconda3/envs/yolov8 -name 'libcudnn*'")
            print("\n然后设置 LD_PRELOAD 重新运行训练：")
            print("LD_PRELOAD=/path/to/libcudnn_ops_infer.so python train_seg.py --mode train")
        else:
            raise e
