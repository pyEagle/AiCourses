# -*- coding:utf-8 -*-
 
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
 
# 1. 定义 P-Tuning 模型
class BertPTuningModel(nn.Module):
    def __init__(self, model_name, num_labels, prompt_len=10):
        super(BertPTuningModel, self).__init__()
        self.prompt_len = prompt_len
        
        self.bert = BertModel.from_pretrained(model_name)
        
        # 冻结 BERT 参数
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.bert.config.hidden_size
        
        # P-Tuning 核心：使用更稳定的连续向量编码
        self.prompt_embeddings = nn.Embedding(prompt_len, self.hidden_size)
        self.prompt_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        self.classifier = nn.Linear(self.hidden_size, num_labels)
 
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        device = input_ids.device # 动态获取设备，确保多GPU兼容
        
        raw_embedding = self.bert.embeddings.word_embeddings(input_ids) 
        
        # 生成 Prompt
        prompt_indices = torch.arange(self.prompt_len).to(device)
        prompt_indices = prompt_indices.unsqueeze(0).expand(batch_size, -1)
        prompts = self.prompt_embeddings(prompt_indices) 
        prompts = self.prompt_encoder(prompts)
        
        # 拼接向量与 Mask
        combined_embedding = torch.cat((prompts, raw_embedding), dim=1)
        prompt_mask = torch.ones(batch_size, self.prompt_len).to(device)
        combined_mask = torch.cat((prompt_mask, attention_mask), dim=1)
        
        outputs = self.bert(inputs_embeds=combined_embedding, attention_mask=combined_mask)
        
        # P-Tuning 通常使用 [CLS] 位的输出
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        return logits

class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=32):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

def main():
    MODEL_NAME = 'your_bert-base-chinese/'
    PROMPT_LEN = 8
    NUM_LABELS = 2
    BATCH_SIZE = 4
    EPOCHS = 10 # 增加轮数以看到拟合效果
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {DEVICE}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_texts = ["这个电影真好看", "太难看了，浪费时间", "演员演技在线", "剧情非常拉跨"]
    train_labels = [1, 0, 1, 0] # 1:正面, 0:负面

    dataset = SimpleDataset(train_texts, train_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BertPTuningModel(MODEL_NAME, NUM_LABELS, PROMPT_LEN).to(DEVICE)

    # 仅优化 Prompt 相关参数和分类头
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3) # 适当调高学习率
    criterion = nn.CrossEntropyLoss()

    # --- 训练阶段 ---
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

    # --- 测试效果展示 ---
    print("\n--- 测试效果 ---")
    model.eval()
    test_texts = ["电影还行", "不推荐，很差劲"]
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=32).to(DEVICE)
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            pred = torch.argmax(logits, dim=1).item()
            label_map = {0: "负面", 1: "正面"}
            print(f"输入: {text} -> 预测结果: {label_map[pred]}")

if __name__ == "__main__":
    main()

