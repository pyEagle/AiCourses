#! /bin/bash

python onnx2rknn3588.py \
           --onnx_model ../model/yolov8n.onnx \
           --platform rk3588 \
           --rknn_model ./best.rknn \
           --test_image_dir /usr/songzs/data/sample/images/test/ \

