# -*- coding:utf-8 -*-

import os

import ultralytics

from ultralytics import YOLO

from setting import parse_arguments

os.environ['ULTRALYTICS_OFFLINE'] = 'True'

def train():
    args = parse_arguments()

    yolo_pt = args.yolo_pt
    coco_yaml= args.coco_yaml
    train_output = args.train_output

    model = YOLO(yolo_pt)
    results = model.train(data=coco_yaml,
                          epochs=200,
                          batch=4,
                          device='0',
                          project=train_output,
                          name='yolov8_retrain',
                          exist_ok=False, # True,
                          amp=False,
                          )
    
    
    # 训练完成后，手动指定 val 的保存路径
    metrics = model.val(project=train_output, name='yolov8_retrain_val')

if __name__ == "__main__":
    train()

