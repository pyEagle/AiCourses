from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import WhisperProcessor


model_path = "./merged_whisper"
onnx_output_dir = "./onnx_output"

print("正在导出 ONNX 模型...")
model = ORTModelForSpeechSeq2Seq.from_pretrained(
    model_path, 
    export=True
)

model.save_pretrained(onnx_output_dir)
processor = WhisperProcessor.from_pretrained(model_path)
processor.save_pretrained(onnx_output_dir)

print(f"ONNX 模型已导出至: {onnx_output_dir}")
