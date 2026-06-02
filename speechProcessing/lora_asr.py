# -*- coding:utf-8 -*-

import torch

from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from peft import PeftModel
from opencc import OpenCC

base_model_id = "openai/whisper-large-v3"
adapter_path = "./whisper-lora-output/checkpoint-500"
audio_path = "./wav/zh.wav"

cc = OpenCC('t2s')

print("正在加载模型...")
model = WhisperForConditionalGeneration.from_pretrained(base_model_id)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

processor = WhisperProcessor.from_pretrained(base_model_id, language="Chinese", task="transcribe")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device
)

print(f"正在识别音频: {audio_path}...")
try:
    result = pipe(
        audio_path,
        generate_kwargs={"language": "chinese", "task": "transcribe"}
    )
    
    simplified_text = cc.convert(result["text"])
    print("\n" + "="*30)
    print("微调模型识别结果 (简体中文):")
    print(simplified_text)
    print("="*30)
except Exception as e:
    print(f"推理过程中出现错误: {e}")
