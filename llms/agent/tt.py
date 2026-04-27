from core.skillManager import SkillManager


sm = SkillManager()
print('--'*50)
print(sm.get_skill_prompt())
print('--'*50)
print(sm.get_all_skills())
print('--'*50)
print(sm.skills)
