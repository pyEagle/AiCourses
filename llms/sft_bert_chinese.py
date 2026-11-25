# -*- coding: utf-8 -*-

import os
from typing import Dict, List

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertLMHeadModel, BertTokenizer


# ==============================================================================
# 1. 构造指令微调数据集
# ==============================================================================
RAW_DATA: List[Dict[str, str]] = [
    {"instruction": "判断以下评论的情感倾向，只输出正面或负面。", "input": "这家店的牛肉面汤鲜肉嫩，强烈推荐！", "output": "正面"},
    {"instruction": "判断以下评论的情感倾向，只输出正面或负面。", "input": "物流太慢了，包装还破损了，非常失望。", "output": "负面"},
    {"instruction": "判断以下评论的情感倾向，只输出正面或负面。", "input": "屏幕清晰流畅，续航也很给力。", "output": "正面"},
    {"instruction": "判断以下评论的情感倾向，只输出正面或负面。", "input": "客服态度极差，问题完全没有解决。", "output": "负面"},
    {"instruction": "提取文本中的核心城市名称。", "input": "明天我将从北京出发前往郑州出差。", "output": "北京、郑州"},
    {"instruction": "提取文本中的核心城市名称。", "input": "上海和深圳是国内重要的科技创新中心。", "output": "上海、深圳"},
    {"instruction": "用一句话解释人工智能中的过拟合。", "input": "", "output": "模型在训练集上表现很好但在测试集上泛化能力差的现象。"},
    {"instruction": "用一句话解释什么是指令微调。", "input": "", "output": "使用指令和响应配对数据训练模型以增强其指令遵循能力的方法。"},
]


def format_prompt(instruction: str, user_input: str = "") -> str:
    if user_input.strip():
        return f"指令：{instruction}\n输入：{user_input}\n回答："
    return f"指令：{instruction}\n回答："


class InstructionDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer: BertTokenizer, max_len: int = 128):
        self.samples = []
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id

        for item in data:
            prompt_text = format_prompt(item["instruction"], item.get("input", ""))
            response_text = item["output"].strip()

            prompt_ids = [cls_id] + tokenizer.encode(prompt_text, add_special_tokens=False)
            response_ids = tokenizer.encode(response_text, add_special_tokens=False) + [sep_id]

            input_ids = (prompt_ids + response_ids)[:max_len]
            labels = ([-100] * len(prompt_ids) + response_ids)[:max_len]

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.samples[idx]


def collate_fn(batch: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []

    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ==============================================================================
# 2. 自回归推理函数
# ==============================================================================
@torch.no_grad()
def generate_response(
    model: BertLMHeadModel,
    tokenizer: BertTokenizer,
    instruction: str,
    user_input: str = "",
    max_new_tokens: int = 32,
    device: torch.device = torch.device("cpu"),
) -> str:
    model.eval()
    prompt_text = format_prompt(instruction, user_input)
    prompt_ids = [tokenizer.cls_token_id] + tokenizer.encode(prompt_text, add_special_tokens=False)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_ids=input_tensor,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,  # 贪心解码，保证教学演示结果稳定可复现
    )

    new_tokens = output_ids[0][len(prompt_ids):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).replace(" ", "")


# ==============================================================================
# 3. 主流程：加载模型 -> 指令微调 -> 效果验证
# ==============================================================================
def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-chinese"
    print(f"[设备] {device} | [基座模型] {model_name}")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name, is_decoder=True)
    model = BertLMHeadModel.from_pretrained(model_name, config=config).to(device)

    dataset = InstructionDataset(RAW_DATA, tokenizer, max_len=128)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    epochs = 25

    print("\n=== 开始指令微调 ===")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")

    print("\n=== 指令微调推理验证 ===")
    test_cases = [
        ("判断以下评论的情感倾向，只输出正面或负面。", "这件衣服质量很好，穿着很舒服。"),
        ("判断以下评论的情感倾向，只输出正面或负面。", "耳机刚用两天就坏了，太差劲。"),
        ("提取文本中的核心城市名称。", "明天我将从北京出发前往郑州出差。"),
        ("用一句话解释什么是指令微调。", ""),
    ]

    for inst, inp in test_cases:
        pred = generate_response(model, tokenizer, inst, inp, device=device)
        print(f"指令: {inst}")
        if inp:
            print(f"输入: {inp}")
        print(f"模型输出: {pred}\n" + "-" * 45)


if __name__ == "__main__":
    main()
