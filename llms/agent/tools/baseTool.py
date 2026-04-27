from abc import ABC, abstractmethod

class BaseTool(ABC):
    """工具基类"""
    @property
    @abstractmethod
    def name(self):
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self):
        """工具描述"""
        pass

    @property
    @abstractmethod
    def parameters(self):
        """工具参数说明"""
        pass

    @abstractmethod
    def call(self, **kwargs):
        """调用工具"""
        pass

    def check_parameters(self, **kwargs):
        """检查参数是否完整"""
        required_params = self.parameters.keys()
        return all(param in kwargs for param in required_params)

