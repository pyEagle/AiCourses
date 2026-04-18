# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def _set_submodule(self, target, module):
    if '.' not in target:
        setattr(self, target, module)
        return
    parent_name, child_name = target.rsplit('.', 1)
    parent = self.get_submodule(parent_name)
    setattr(parent, child_name, module)

torch.nn.Module.set_submodule = _set_submodule

DATA_PATH = "train.json"
MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "./qwen3_lora_dpo"

BATCH_SIZE = 1
EPOCHS = 10
LR = 2e-5
BETA = 0.1
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def preprocess(example):
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]
    
    prompt_enc = tokenizer(prompt, truncation=True, max_length=MAX_LEN)
    chosen_text = prompt + "\n" + chosen
    rejected_text = prompt + "\n" + rejected
    
    chosen_enc = tokenizer(
        chosen_text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )
    rejected_enc = tokenizer(
        rejected_text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )
    
    prompt_len = len(prompt_enc["input_ids"])
    chosen_prompt_mask = [0] * prompt_len + [1] * (MAX_LEN - prompt_len)
    rejected_prompt_mask = [0] * prompt_len + [1] * (MAX_LEN - prompt_len)
    
    chosen_prompt_mask = chosen_prompt_mask[:MAX_LEN]
    rejected_prompt_mask = rejected_prompt_mask[:MAX_LEN]
    
    return {
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "chosen_prompt_mask": chosen_prompt_mask,
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
        "rejected_prompt_mask": rejected_prompt_mask,
    }

dataset = load_dataset("json", data_files=DATA_PATH)["train"]
dataset = dataset.map(preprocess)
dataset.set_format(type="torch", columns=[
    "chosen_input_ids", "chosen_attention_mask", "chosen_prompt_mask",
    "rejected_input_ids", "rejected_attention_mask", "rejected_prompt_mask"
])

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 正常加载模型
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
)

reference_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
)

reference_model.eval()
for p in reference_model.parameters():
    p.requires_grad = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)

model = get_peft_model(base_model, lora_config)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.print_trainable_parameters()
model.train()

def compute_log_probs(model, input_ids, attention_mask, prompt_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )
    
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    per_token_logps = torch.gather(
        log_probs,
        dim=2,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)
    
    mask = attention_mask[:, 1:] * prompt_mask[:, 1:]
    return (per_token_logps * mask).sum(dim=1)

def dpo_loss(policy_chosen_logps, policy_rejected_logps, 
             reference_chosen_logps, reference_rejected_logps, 
             beta=0.1):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    # ===================== 🔥 修复拼写错误 =====================
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    # ==========================================================
    
    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)
    
    loss = losses.mean()
    
    return loss, {
        "loss": loss.item(),
        "chosen_logps": policy_chosen_logps.mean().item(),
        "rejected_logps": policy_rejected_logps.mean().item(),
        "logits": logits.mean().item()
    }

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LR,
    betas=(0.9, 0.95),
    weight_decay=0.01
)
# 修复新版PyTorch警告
scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()
    
    for step, batch in enumerate(dataloader):
        chosen_input_ids = batch["chosen_input_ids"].to(DEVICE)
        chosen_attention_mask = batch["chosen_attention_mask"].to(DEVICE)
        chosen_prompt_mask = batch["chosen_prompt_mask"].to(DEVICE)
        
        rejected_input_ids = batch["rejected_input_ids"].to(DEVICE)
        rejected_attention_mask = batch["rejected_attention_mask"].to(DEVICE)
        rejected_prompt_mask = batch["rejected_prompt_mask"].to(DEVICE)
        
        # 修复新版PyTorch警告
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            policy_c = compute_log_probs(model, chosen_input_ids, chosen_attention_mask, chosen_prompt_mask)
            policy_r = compute_log_probs(model, rejected_input_ids, rejected_attention_mask, rejected_prompt_mask)
            
            with torch.no_grad():
                ref_c = compute_log_probs(reference_model, chosen_input_ids, chosen_attention_mask, chosen_prompt_mask)
                ref_r = compute_log_probs(reference_model, rejected_input_ids, rejected_attention_mask, rejected_prompt_mask)
            
            loss, loss_info = dpo_loss(policy_c, policy_r, ref_c, ref_r, BETA)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        if step % 1 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {step+1}/{len(dataloader)} | "
                  f"Loss: {loss_info['loss']:.4f} | "
                  f"Chosen LogP: {loss_info['chosen_logps']:.4f} | "
                  f"Rejected LogP: {loss_info['rejected_logps']:.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"\nEpoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}\n")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("训练完成！模型已保存至:", OUTPUT_DIR)

