# -*- coding:utf-8 -*-

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_model",
        required=True, 
        help="onnx模型文件")

    parser.add_argument(
        "--platform",
        required=True,
        help="边缘计算开发板型号")

    parser.add_argument(
        "--rknn_model",
        required=True,
        help="rknn模型文件")

    parser.add_argument(
        "--test_image_dir",
        required=True,
        help="测试图片目录")

    parser.add_argument(
        "--do_quant",
        required=False,
        default=True,
        help="设置quant")

    return parser.parse_args()

