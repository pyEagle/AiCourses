# -*- coding:utf-8 -*-
import os
from ultralytics import YOLO
from setting import parse_arguments

# 设置为离线模式
os.environ['ULTRALYTICS_OFFLINE'] = 'True'

def train():
    args = parse_arguments()

    yolo_pt = args.yolo_pt
    coco_yaml = args.coco_yaml
    train_output = args.train_output

    model = YOLO(yolo_pt)
    results = model.tune(
        data=coco_yaml,
        epochs=30,           # 每次迭代的训练轮次 
        iterations=50,       # 总共执行的实验次数
        optimizer='AdamW',
        device='0',          # 指定 GPU
        project=train_output,
        name='yolov8_tuning',
        plots=True,          # 自动生成寻优过程可视化图表
        save=True,           # 保存过程中的模型检查点
        exist_ok=False
    )
    
    # 寻优完成后，model.tune 会自动保存表现最好的模型到 project/name/weights/best.pt
    best_model_path = os.path.join(train_output, 'yolov8_tuning', 'weights', 'best.pt')
    best_model = YOLO(best_model_path)
    
    # 验证最佳模型
    print(f"\n🏆 寻优完成，开始验证最佳模型: {best_model_path}")
    metrics = best_model.val(project=train_output, name='yolov8_best_val')

if __name__ == "__main__":
    train()
