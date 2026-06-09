“”“
my_dataset/
├── data.yaml
├── images/
│   ├── train/    # 存放训练集图片
│   └── val/      # 存放验证集图片
└── labels/
    ├── train/    # 存放训练集 txt 标签
    └── val/      # 存放验证集 txt 标签

data.yaml
path: ./my_dataset
train: images/train
val: images/val

names:
  0: defect_type_A
  1: defect_type_B
”“”

import os
from ultralytics import YOLO

# 基础配置
DATA_YAML = "./my_dataset/data.yaml"
TEST_IMAGES_DIR = "./my_dataset/images/test_batch" 
PROJECT_NAME = "yolov8_final_pipeline"
MODEL_NAME = "yolov8n-seg.pt"

def tune_model():
    print("\n" + "=" * 60)
    print("超参数寻优 (Hyperparameter Tuning)")
    print("=" * 60)
    model = YOLO(MODEL_NAME)
    
    # 寻优：寻找最优超参数
    model.tune(
        data=DATA_YAML,
        epochs=30,
        iterations=50,
        optimizer='AdamW',
        plots=True,
        save=True,
        cache=True,
        project=PROJECT_NAME,
        name="tuning_exp"
    )
    
    best_weights = os.path.join(PROJECT_NAME, "tuning_exp", "weights", "best.pt")
    return best_weights

def evaluate_model(best_model_path):
    print("\n" + "=" * 60)
    print("开始整体验证 (Evaluation on full test set)")
    print("=" * 60)
    model = YOLO(best_model_path)
    
    metrics = model.val(data=DATA_YAML, save_json=True)
    
    print(f"\n验证指标:")
    print(f"Mask mAP50    : {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95 : {metrics.seg.map:.4f}")
    return model

def inference_model(model, source_path):
    print("\n" + "=" * 60)
    print(f"开始批量推理 (Inference): {source_path}")
    print("=" * 60)
    
    results = model.predict(
        source=source_path,
        conf=0.25,
        save=True,
        project=PROJECT_NAME,
        name="inference_results"
    )
    
    print(f"成功处理 {len(results)} 张图片。")

def export_onnx(model):
    print("\n" + "=" * 60)
    print("导出 ONNX 模型")
    print("=" * 60)
    model.export(format="onnx", simplify=True)

def main():
    best_weights = tune_model()
    best_model = evaluate_model(best_weights)
    inference_model(best_model, TEST_IMAGES_DIR)
    export_onnx(best_model)

    print("结果已保存在目录: " + PROJECT_NAME)

if __name__ == "__main__":
    main()

  
