# -*- coding:utf-8 -*-

import os
import shutil

from ultralytics import YOLO

from setting import parse_arguments

def pt2onnx():
    args = parse_arguments()

    yolo_pt = args.yolo_pt
    onnx_dir = args.onnx_dir

    model = YOLO(yolo_pt)
    exported_path = model.export(format="onnx", 
                 project=onnx_dir, # 不起作用
                 name='',
                 simplify=True, 
                 exist_ok=False, #True,
                 batch=1,
                 imgsz=640, # TODO
                 opset=12)

    # 手动迁移onnx模型
    os.makedirs(onnx_dir, exist_ok=True)
    file_path = os.path.join(onnx_dir, os.path.basename(exported_path))
    if os.path.exists(file_path):
        os.remove(file_path)
    shutil.move(exported_path, onnx_dir)
    print(f'onnx模型已迁移：{onnx_dir}')


if __name__ == "__main__":
    pt2onnx()

