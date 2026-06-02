# -*- coding:utf-8 -*-

import os
import torch

from transformers import pipeline
from opencc import OpenCC  # 引入简繁转换库

# OpenCC转换器(t2s 表示繁体转简体)
cc = OpenCC('t2s')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"正在使用设备: {device.upper()}")

model_id = "openai/whisper-small" 
pipe = pipeline(
    "automatic-speech-recognition", 
    model=model_id, 
    device=device
)

audio_path = "./wav/zh.wav"
print(f"正在识别音频: {audio_path} ...")

try:
    result = pipe(
        audio_path,
        generate_kwargs={"language": "chinese", "task": "transcribe"}
    )
    
    raw_text = result["text"]
    simplified_text = cc.convert(raw_text)
    
    print("\n" + "="*30)
    print("识别结果 (强制简体):")
    print(simplified_text)
    print("="*30)

except Exception as e:
    print(f"识别过程中出现错误: {e}")
