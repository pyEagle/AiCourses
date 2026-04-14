# -*- coding:utf-8 -*-

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo_pt",
        required=True, 
        help="yolo模型文件")

    parser.add_argument(
        "--coco_yaml",
        required=True,
        help="coco.yaml文件")

    parser.add_argument(
        "--train_output",
        required=True,
        help="训练输目录")

    return parser.parse_args()

