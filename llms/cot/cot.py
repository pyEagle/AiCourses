# -×- coding:utf-8 -*-
 
import json
 
import ollama
 
def cot_demo(model_name: str = "deepseek-r1:latest"):
    cot_prompt = """
    请你用思维链（Chain of Thought）的方式分步解决这个数学问题：
    问题：小明有15个苹果，他先送给小红4个，又从超市买了8个，然后分给弟弟一半，请问小明最后还剩多少个苹果？
    
    要求：
    1. 第一步：明确初始条件和每一步操作
    2. 第二步：逐步计算每一步后的苹果数量
    3. 第三步：总结最终结果并给出答案
    """

    print("===== deepseek-r1推理过程 =====\n")
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": cot_prompt
            }
        ],
        stream=True  # 流式输出更贴近真实的"思考"过程
    )
 
    full_response = ""
    for chunk in response:
        content = chunk["message"]["content"]
        full_response += content
        print(content, end="", flush=True)  # 实时打印，不缓存
 
    print("\n===== 推理完成 =====")
 
 
if __name__ == "__main__":
    cot_demo(model_name="deepseek-r1:latest")
