from rknn.api import RKNN

ONNX_MODEL = "./onnx_output/model.onnx"
RKNN_MODEL = "./whisper_rk3588.rknn"

rknn = RKNN()

print("配置 RKNN 环境...")
rknn.config(
    mean_values=[[0]],
    std_values=[[255]],
    target_platform='rk3588',
    optimization_level=3
)

print("加载 ONNX 模型...")
ret = rknn.load_onnx(model=ONNX_MODEL)
if ret != 0:
    raise RuntimeError("ONNX加载失败")

print("构建 RKNN 模型...")
ret = rknn.build(
    do_quantization=True,
    dataset="./dataset.txt"   # 校准数据（必须）
)
if ret != 0:
    raise RuntimeError("build失败")

print("导出 RKNN 模型...")
rknn.export_rknn(RKNN_MODEL)

rknn.release()

print("完成！RK3588模型已生成:", RKNN_MODEL)
