import os
from ultralytics import YOLO


DATA_YAML = "./my_dataset/data.yaml"
TEST_IMAGE = "./my_dataset/images/val/test.jpg"  # 请替换为实际存在的测试图片
PROJECT_NAME = "yolov8_few_shot"
EXP_NAME = "seg_experiment"
MODEL_NAME = "yolov8n-seg.pt"  # 少量样本优先使用 Nano 模型


def train_model():
    print("\n" + "=" * 60)
    print("🚀 开始训练 (Training)")
    print("=" * 60)

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=4,
        workers=2,
        cache=True, 
        amp=True,
        optimizer="AdamW",
        lr0=1e-4, 
        lrf=0.01,
        weight_decay=5e-4,
        patience=20,
        freeze=None,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.2,
        copy_paste=0.3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        save=True,
        project=PROJECT_NAME,
        name=EXP_NAME
    )
    return results

def evaluate_model(best_model_path):
    print("\n" + "=" * 60)
    print("📊 开始验证 (Evaluation)")
    print("=" * 60)
    model = YOLO(best_model_path)

    metrics = model.val(data=DATA_YAML)

    print("\n" + "-" * 30)
    print(f"Box mAP50      : {metrics.box.map50:.4f}")
    print(f"Box mAP50-95   : {metrics.box.map:.4f}")
    print(f"Mask mAP50     : {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95  : {metrics.seg.map:.4f}")
    print("-" * 30)

    return model

def inference_model(model, image_path):
    print("\n" + "=" * 60)
    print("👁️ 开始推理 (Inference)")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"⚠️ 测试图片不存在: {image_path}，已跳过推理环节。")
        return

    results = model.predict(
        source=image_path,
        conf=0.25,
        save=True,
        save_txt=True,
        save_conf=True,
        project=PROJECT_NAME,
        name="seg_outputs"
    )

    for r in results:
        num_boxes = len(r.boxes)
        num_masks = len(r.masks.data) if r.masks is not None else 0
        print(f"🎯 检测目标: {num_boxes} | Mask数量: {num_masks}")

    save_dir = os.path.join(PROJECT_NAME, "seg_outputs")
    print(f"📂 预测结果及坐标文件已保存至: {save_dir}")

def export_onnx(model):
    print("\n" + "=" * 60)
    print("📦 导出 ONNX 模型 (Export)")
    print("=" * 60)

    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,        # 简化 ONNX 算子图，方便后端推理引擎(如 TensorRT/OpenVINO)解析
        opset=12              # 兼容性较好的算子集版本
    )
    print("✅ ONNX 模型导出完成，可用于生产环境部署！")

def main():
    train_results = train_model()

    best_model_path = os.path.join(train_results.save_dir, "weights", "best.pt")
    print(f"\n🏆 最佳模型权重位于: {best_model_path}")

    best_model = evaluate_model(best_model_path)

    inference_model(best_model, TEST_IMAGE)
    export_onnx(best_model)

    print("\n" + "=" * 60)
    print("🎉 实例分割流水线全部执行完毕！")
    print("=" * 60)

if __name__ == "__main__":
    main()
  
