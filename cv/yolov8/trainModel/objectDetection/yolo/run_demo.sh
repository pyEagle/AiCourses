#! /bin/bash

# --train_output 需指定绝对路径
python reTrain.py --yolo_pt /usr/songzs/model/yolov8n.pt --coco_yaml ./config/coco.yaml --train_output /usr/songzs/model/objectDetection/yolo/train/

