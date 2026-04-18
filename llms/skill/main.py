# -*- coding:utf-8 -*-
 
import json
import ollama
from datetime import datetime
from pathlib import Path
 
from config import config
from tools import ToolRegistry
from skill_loader import SkillLoader
 
 
class IntelligentSummarizer:
    
    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.skill_loader = SkillLoader()
        self.conversation_history = []
    
    def _build_system_prompt(self):
        tools_info = json.dumps(ToolRegistry.get_tools_schema(), ensure_ascii=False, indent=2)
        
        skills_info = "\n".join([
            f"- {skill['name']}: {skill['description']}"
            for skill in self.skill_loader.list_skills()
        ])
        
        return f"""你是一个智能摘要助手，运行在本地环境中。
## 可用工具
{tools_info}
## 可用技能
{skills_info}
## 决策规则
1. 分析用户输入，判断是否需要摘要
2. 如果需要摘要，自主决定调用 summarize_text 工具
3. 如果需要关键词，自主决定调用 extract_keywords 工具
4. 如果不需要工具，直接回答
## 输出格式
当需要调用工具时，返回 JSON 格式：
{{
    "need_tool": true,
    "tool_name": "工具名称",
    "tool_args": {{ "参数": "值" }},
    "reason": "调用原因"
}}
当不需要工具时，返回 JSON 格式：
{{
    "need_tool": false,
    "response": "直接回复内容"
}}
请始终以 JSON 格式回复。"""
    
    def _parse_model_response(self, response):
        try:
            # 清理可能的 markdown 标记
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            return {
                "need_tool": False,
                "response": f"解析响应失败：{str(e)}\n原始响应：{response}"
            }
    
    def _call_tool(self, tool_name, **kwargs):
        print(f"🔧 调用工具：{tool_name}")
        result = ToolRegistry.execute(tool_name, **kwargs)
        print(f"✅ 工具执行完成")
        return result
    
    def _save_result(self, content, filename=None):
        """保存结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{timestamp}.txt"
        
        output_path = config.OUTPUT_DIR / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"📁 结果已保存：{output_path}")
        return output_path
    
    def process(self, user_input, save_result=True):
        print(f"\n{'='*60}")
        print(f"📝 用户输入：{user_input[:100]}{'...' if len(user_input) > 100 else ''}")
        print(f"{'='*60}\n")
        
        # 构建消息
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_input}
        ]
        
        # 调用本地模型
        print(f"🤖 调用模型：{self.model}")
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": config.TEMPERATURE,
                "num_predict": config.MAX_TOKENS
            }
        )
        
        model_response = response["message"]["content"]
        print(f"📤 模型响应：{model_response}")
        
        # 解析响应
        decision = self._parse_model_response(model_response)
        
        # 执行决策
        if decision.get("need_tool", False):
            tool_name = decision.get("tool_name")
            tool_args = decision.get("tool_args", {})
            
            if tool_name:
                tool_result = self._call_tool(tool_name, **tool_args)
                
                # 生成最终回复
                final_response = {
                    "tool_called": tool_name,
                    "tool_result": tool_result,
                    "user_input": user_input[:500]
                }
                
                if save_result:
                    self._save_result(json.dumps(final_response, ensure_ascii=False, indent=2))
                
                return final_response
        else:
            final_response = {
                "tool_called": None,
                "response": decision.get("response", "无响应")
            }
            
            if save_result:
                self._save_result(final_response["response"])
            
            return final_response
 
 
def main():
    """主函数"""
    summarizer = IntelligentSummarizer()
    
    # 示例 1：长文本摘要
    long_text = """
    人工智能是当今科技领域最热门的话题之一。随着深度学习技术的快速发展，
    人工智能在图像识别、自然语言处理、语音识别等领域取得了突破性进展。
    特别是在大语言模型方面，GPT、BERT、LLaMA 等模型的出现，使得机器能够
    更好地理解和生成人类语言。这些技术正在改变我们的工作方式、生活方式，
    甚至思维方式。然而，人工智能的发展也带来了一些挑战，包括就业问题、
    隐私保护、算法偏见等。我们需要在推动技术进步的同时，也要关注这些
    社会问题，确保人工智能的发展能够造福全人类。未来，人工智能将与
    各行各业深度融合，创造更多的价值和可能性。
    """ * 3  # 重复以增加长度
    
    print("\n" + "🚀" * 30)
    print("开始智能摘要处理")
    print("🚀" * 30 + "\n")
    
    result = summarizer.process(
        user_input=f"请对以下文本进行摘要：{long_text}",
        save_result=True
    )
    
    print("\n" + "="*60)
    print("📊 最终结果")
    print("="*60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
 
 
if __name__ == "__main__":
    main()
