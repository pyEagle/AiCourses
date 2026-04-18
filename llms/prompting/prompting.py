# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW # Import from torch instead
 
# 1. 配置参数
MODEL_NAME = './model/LLms/bert-base-chinese'
NUM_VIRTUAL_TOKENS = 10  # 虚拟提示词的数量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
class BertPromptTuning(nn.Module):
    def __init__(self, model_name, num_virtual_tokens, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.num_virtual_tokens = num_virtual_tokens
        hidden_size = self.bert.config.hidden_size
        
        self.soft_prompt = nn.Parameter(torch.randn(num_virtual_tokens, hidden_size))
        self.classifier = nn.Linear(hidden_size, num_labels)
 
    def forward(self, input_ids, attention_mask):
        raw_embeddings = self.bert.embeddings.word_embeddings(input_ids) # [batch, seq_len, hidden]
        batch_size = input_ids.shape[0]
        learned_prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        embeddings = torch.cat([learned_prompt, raw_embeddings], dim=1)
        prompt_mask = torch.ones(batch_size, self.num_virtual_tokens).to(DEVICE)
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        outputs = self.bert(inputs_embeds=embeddings, attention_mask=extended_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
 
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
texts = ["这个电影真好看！", "实在是太难看了。"]
labels = torch.tensor([1, 0]).to(DEVICE)
 
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
 
model = BertPromptTuning(MODEL_NAME, NUM_VIRTUAL_TOKENS).to(DEVICE)
 
# 观察参数量：只有 soft_prompt 和 classifier 的参数是可训练的
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=1e-3)
criterion = nn.CrossEntropyLoss()
 
model.train()
optimizer.zero_grad()
outputs = model(inputs['input_ids'], inputs['attention_mask'])
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
 
print(f"Loss: {loss.item():.4f}")
print("训练成功！仅训练了 Prompt 和分类头。")
