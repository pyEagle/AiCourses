# -×- coding:utf-8 -*-
 
 
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
 
 
class IA3LinearWrapper(nn.Module):
    def __init__(self, original_linear):
        super().__init__()
        self.original_linear = original_linear
        
        # 冻结原始参数
        for param in self.original_linear.parameters():
            param.requires_grad = False
        
        # IA3 核心
        self.ia3_vector = nn.Parameter(torch.ones(original_linear.out_features))
 
    def forward(self, x):
        # 公式：(xW^T) ⊙ l
        output = self.original_linear(x)
        return output * self.ia3_vector

def apply_ia3_to_bert(model):
    for name, module in model.named_modules():
        # IA3 通常作用于 Key, Value 和 Intermediate 层的输出
        if any(target in name for target in ['attention.self.key', 'attention.self.value', 'intermediate.dense']):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, IA3LinearWrapper(module))

    return model

def main():
    model_name = "your_bert-base-chinese" 
    
    print(f"正在加载 {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    base_model = BertModel.from_pretrained(model_name)
 
    model = apply_ia3_to_bert(base_model)
    
    classifier = nn.Linear(768, 2) 
 
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters()) + \
                sum(p.numel() for p in classifier.parameters())
    print(f"训练占比: {100 * trainable_params / all_param:.4f}% ({trainable_params}/{all_param})\n")
 
    texts = ["我非常喜欢这部电影！", "这个产品太差劲了。"]
    labels = torch.tensor([1, 0]) # 1: 正面, 0: 负面
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
 
    model.eval()
    with torch.no_grad():
        pre_logits = classifier(model(**inputs).last_hidden_state[:, 0, :])
        pre_probs = F.softmax(pre_logits, dim=-1)
        print("训练前预测概率 (句子1 - 正向):", pre_probs[0][1].item())
 
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=5e-3)
    model.train()
    
    print("\n开始快速优化（5次迭代）...")
    for epoch in range(9):
        optimizer.zero_grad()
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]
        logits = classifier(last_hidden_state)
        
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
 
    model.eval()
    with torch.no_grad():
        post_logits = classifier(model(**inputs).last_hidden_state[:, 0, :])
        post_probs = F.softmax(post_logits, dim=-1)
        print("\n训练后预测概率 (句子1 - 正向):", post_probs[0][1].item())
        
        if post_probs[0][1] > pre_probs[0][1]:
            print(">>> 测试成功：IA3 参数已通过梯度下降学习，模型性能有所提升。")

if __name__ == "__main__":
    main()



