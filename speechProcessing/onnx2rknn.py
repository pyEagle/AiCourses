"""
pip install rknn-toolkit2
"""

from rknn.api import RKNN

ONNX_MODEL = "./onnx_output/encoder_model.onnx"
RKNN_MODEL = "./whisper_encoder.rknn"

rknn = RKNN(verbose=True)

print("Config")

rknn.config(
    target_platform="rk3588"
)

print("加载 ONNX")

ret = rknn.load_onnx(
    model=ONNX_MODEL,
    inputs=["input_features"],
    input_size_list=[
        [1,128,3000]
    ]
)

if ret != 0:
    raise RuntimeError("加载onnx失败")

print("Build")

ret = rknn.build(
    do_quantization=False
)

if ret != 0:
    raise RuntimeError("构建失败")

print("Export")

ret = rknn.export_rknn(RKNN_MODEL)

if ret != 0:
    raise RuntimeError("输出失败")

rknn.release()

print("成功")
