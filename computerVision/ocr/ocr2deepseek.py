 # -*- coding:utf-8 -*-
import os
import sys
import warnings

import torch

from transformers import AutoModel, AutoTokenizer

# 过滤transformers相关的DeprecationWarning、FutureWarning、UserWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_name = '/usr/songzs/model/LLms/DeepSeek-OCR2/'
torch_dtype = torch.bfloat16  # Fix: flash Attention警告

image_file = sys.argv[1]
output_path = './result'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 用eos_token替代pad_token

model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    _attn_implementation='flash_attention_2',
    torch_dtype=torch_dtype,                    # 显式指定精度
    device_map='cuda'                           # 直接加载到GPU
)
model = model.eval()

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
if len(sys.argv) < 2:
    print("使用方式: python deepseek-ocr2.py <图片路径>")
    sys.exit(1)

res = model.infer(
    tokenizer=tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=768,
    crop_mode=True,
    save_results=True
)

print(f"✅ 模型是否运行在GPU上: {next(model.parameters()).is_cuda}")
print(f"✅ OCR完成，结果已保存至: {output_path}")
print(f"✅ 推理结果: {res}")

