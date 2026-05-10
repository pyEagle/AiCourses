# -*- coding:utf-8 -*-

from abc import ABC, abstractmethod

class BaseSkill(ABC):
    """技能基类"""
    @property
    @abstractmethod
    def name(self):
        """技能名称"""
        pass

    @property
    @abstractmethod
    def description(self):
        """技能描述"""
        pass

    @property
    @abstractmethod
    def parameters(self):
        """技能参数说明"""
        pass

    @abstractmethod
    def execute(self, **kwargs):
        """执行技能"""
        pass

    def check_parameters(self, **kwargs):
        """检查参数是否完整"""
        required_params = self.parameters.keys()
        return all(param in kwargs for param in required_params)

