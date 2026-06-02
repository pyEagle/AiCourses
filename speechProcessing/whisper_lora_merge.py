from peft import PeftModel
from transformers import WhisperForConditionalGeneration
import shutil

# 路径配置
base_model_id = "openai/whisper-large-v3" 
adapter_path = "./whisper-lora-output/checkpoint-500" # 你的检查点路径
output_dir = "./merged_whisper"

print("正在加载基础模型与 LoRA 适配器...")
base_model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, adapter_path)

print("正在合并权重 (Merge & Unload)...")
# merge_and_unload 会将 LoRA 权重合并入 base model
merged_model = model.merge_and_unload()

print(f"正在保存合并后的模型到: {output_dir}")
merged_model.save_pretrained(output_dir)

# 同时保存 tokenizer 和 preprocessor 配置，确保推理时可以使用
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained(base_model_id)
processor.save_pretrained(output_dir)

print("合并完成！")
