# -*- coding:utf-8 -*-
 
import sys
import torch
sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
 
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
 
class EnterpriseAgent:
    def __init__(self):
        #self.model_path = 'your_Qwen3-8B_path'
        self.model_path = '/usr/rfzn/LLms/Qwen3-8B/'
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16
        )
        self.model.eval()
        
    def _build_prompt(self, user_input):
        
        messages = [
            {"role": "user", "content": user_input}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True ,
        )
        return prompt
 
    def generate_response(self, user_input):
        prompt = self._build_prompt(user_input)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
 
if __name__ == "__main__":
    agent = EnterpriseAgent()
    while 1:
        user_input = input("请输入问题: ")
        if user_input in ('q', 'Q'):
            break
        response = agent.generate_response(user_input)
        print(response)
