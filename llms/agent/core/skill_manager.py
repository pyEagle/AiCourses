from skills.base_skill import BaseSkill
from skills.calculator_skill import CalculatorSkill

class SkillManager:
    def __init__(self):
        self.skills = {
            CalculatorSkill().name: CalculatorSkill()
        }

    def get_all_skills(self) :
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "parameters": skill.parameters
            }
            for skill in self.skills.values()
        ]

    def execute_skill(self, skill_name, **kwargs):
        if skill_name not in self.skills:
            return f"未知技能：{skill_name}"
        
        skill = self.skills[skill_name]
        return skill.execute(**kwargs)

    def get_skill_prompt(self):
        skills_info = self.get_all_skills()
        prompt = "可用技能列表：\n"
        for skill in skills_info:
            prompt += f"- {skill['name']}: {skill['description']}\n"
            prompt += f"  参数：{skill['parameters']}\n"
        return prompt

