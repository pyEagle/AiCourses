# -*- coding:utf-8 -*-

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo_pt",
        required=True, 
        help="yolo模型文件")

    parser.add_argument(
        "--onnx_dir",
        required=True,
        help="onnx模型目录")

    parser.add_argument(
        "--train_output",
        required=False,
        help="训练输目录")

    return parser.parse_args()

