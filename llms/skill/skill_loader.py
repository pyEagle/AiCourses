import json
import os
from pathlib import Path
from config import config
 
class SkillLoader:
    def __init__(self, skills_dir = None):
        self.skills_dir = skills_dir or config.PROJECT_ROOT / "skills"
        self.skills = {}
        self._load_all_skills()
    
    def _load_all_skills(self):
        """加载所有可用的 Skills"""
        if not self.skills_dir.exists():
            return
        
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_config = skill_dir / "skill.json"
                if skill_config.exists():
                    self._load_skill(skill_dir, skill_config)
    
    def _load_skill(self, skill_dir, config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            skill_data = json.load(f)
        
        skill_name = skill_dir.name
        self.skills[skill_name] = {
            "name": skill_data.get("name", skill_name),
            "description": skill_data.get("description", ""),
            "version": skill_data.get("version", "1.0.0"),
            "tools": skill_data.get("tools", []),
            "script_path": skill_dir / "script.py",
            "doc_path": skill_dir / "skill.md"
        }
    
    def get_skill(self, name):
        return self.skills.get(name)
    
    def list_skills(self):
        return list(self.skills.values())
    
    def get_available_tools(self):
        tools = []
        for skill in self.skills.values():
            tools.extend(skill.get("tools", []))
        return tools
