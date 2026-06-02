from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import WhisperProcessor

merged_model_path = "./merged_whisper"    # 合并好的模型路径
onnx_output_dir = "./onnx_output"        # ONNX 输出路径
base_model_id = "openai/whisper-large-v3" 
print(f"正在从 {merged_model_path} 导出 ONNX 模型...")

model = ORTModelForSpeechSeq2Seq.from_pretrained(
    merged_model_path, 
    export=True
)

model.save_pretrained(onnx_output_dir)

print(f"正在从 {base_model_id} 获取 Processor 配置...")
processor = WhisperProcessor.from_pretrained(base_model_id)

processor.save_pretrained(onnx_output_dir)
print("="*30)
print(f"转换成功！")
print(f"ONNX 模型和配置文件已保存至: {onnx_output_dir}")
print("="*30)
