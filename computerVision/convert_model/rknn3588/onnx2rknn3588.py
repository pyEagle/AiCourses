# -*- coding:utf-8 -*-

import os
import sys
import glob

from rknn.api import RKNN

from setting import parse_arguments

# 准备测试图像
def prepareDataSet(test_image_dir):
    txt_name  = './temp.txt'
    image_list = glob.glob(os.path.join(test_image_dir, '*.png'))
    with open(txt_name, 'w') as fid:
        images = '\n'.join(image_list)
        fid.write(images)

    return txt_name


def run():
    args = parse_arguments()
    onnx_model = args.onnx_model
    platform = args.platform
    rknn_model = args.rknn_model
    do_quant = args.do_quant
    test_image_dir = args.test_image_dir

    test_image_path = prepareDataSet(test_image_dir)

    rknn = RKNN(verbose=False)
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[
                    [255, 255, 255]], target_platform=platform)
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=onnx_model)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=test_image_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(rknn_model)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()

    os.remove(test_image_path)

if __name__ == '__main__':
    run()

