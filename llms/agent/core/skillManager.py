import os
import importlib.util
import re
import inspect
from typing import Dict, List, Any

from skills.baseSkill import BaseSkill

class SkillManager:
    def __init__(self, skills_dir="./skills"):
        self.skills_dir = skills_dir
        self.skills = {}
        self.tools_definition = []
        self.load_skills()

    def parse_skill_md(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        name_match = re.search(r'name:\s*(.+)', content)
        if not name_match:
            return None
        name = name_match.group(1).strip()
        
        desc_match = re.search(r'description:\s*(.+)', content)
        description = desc_match.group(1).strip() if desc_match else "无描述"
        
        params = {}
        params_section = re.search(r'##\s*参数\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        if params_section:
            for line in params_section.group(1).strip().split('\n'):
                if ':' in line:
                    param_name, param_desc = line.split(':', 1)
                    params[param_name.strip()] = param_desc.strip()
        
        return {
            "name": name,
            "description": description,
            "parameters": params
        }

    def load_skills(self):
        print("正在扫描 Skills...")
        for root, _, files in os.walk(self.skills_dir):
            if '__pycache__' in root:
                continue
                
            md_file = os.path.join(root, "skill.md")
            if os.path.exists(md_file):
                meta = self.parse_skill_md(md_file)
                if meta:
                    skill_name = meta['name']
                    py_file = os.path.join(root, f"{skill_name}.py")
                    
                    if os.path.exists(py_file):
                        module_name = f"{root.replace(os.sep, '.')}.{skill_name}"
                        print(f"发现技能模块: {root} -> {module_name} ({py_file})")
                        
                        try:
                            spec = importlib.util.spec_from_file_location(module_name, py_file)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            self.scan_and_load_classes(module, meta, skill_name, root)
                        except Exception as e:
                            print(f"加载模块 {module_name} 时出错: {str(e)}")
    
        print(f"共加载 {len(self.skills)} 个技能。")

    def scan_and_load_classes(self, module, meta, skill_name, root):
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BaseSkill) and obj != BaseSkill:
                try:
                    skill_instance = obj()
                    
                    if not hasattr(skill_instance, 'name'):
                        skill_instance.name = skill_name
                    
                    self.skills[skill_name] = skill_instance
                    
                    parameters = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                    
                    for param_name, param_desc in meta['parameters'].items():
                        parameters["properties"][param_name] = {
                            "type": "string",
                            "description": param_desc
                        }
                        parameters["required"].append(param_name)
                    
                    self.tools_definition.append({
                        "type": "function",
                        "function": {
                            "name": skill_name,
                            "description": meta['description'],
                            "parameters": parameters
                        }
                    })
                    print(f"✓ 已加载技能: {skill_name} (类: {name})")
                except Exception as e:
                    print(f"实例化类 {name} 时出错: {str(e)}")

    def get_all_skills(self):
        """获取所有技能信息"""
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "parameters": skill.parameters
            }
            for skill in self.skills.values()
        ]

    def execute_skill(self, skill_name, **kwargs):
        """执行指定技能"""
        if skill_name not in self.skills:
            return f"错误：未找到技能 {skill_name}"
        
        try:
            skill = self.skills[skill_name]
            missing_params = [param for param in skill.parameters if param not in kwargs]
            if missing_params:
                return f"错误：缺少必需参数 {', '.join(missing_params)}"

            return skill.execute(**kwargs)
        except Exception as e:
            return f"执行出错: {str(e)}"

    def get_skill_prompt(self):
        """生成技能调用提示词"""
        skills_info = self.get_all_skills()
        prompt = "可用skill列表：\n"
        for skill in skills_info:
            prompt += f"- **{skill['name']}**: {skill['description']}\n"
            prompt += f"  **参数**：{', '.join(skill['parameters'])}\n"
        return prompt

    def get_tools_for_llm(self):
        """返回给大模型的工具定义列表"""
        return self.tools_definition

