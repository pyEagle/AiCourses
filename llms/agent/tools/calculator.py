import math

from tools.baseTool import BaseTool

class Calculator(BaseTool):
    @property
    def name(self):
        return "calculator"

    @property
    def description(self):
        return "执行数学计算，支持加减乘除、平方、开方等基本运算"

    @property
    def parameters(self):
        return {
            "expression": "数学表达式（如：1+2*3、sqrt(16)）"
        }

    def call(self, **kwargs):
        if not self.check_parameters(**kwargs):
            return "参数错误，缺少数学表达式"
        
        expression = kwargs.get("expression")
        try:
            # 安全执行数学表达式
            allowed_names = {
                "math": math,
                "sqrt": math.sqrt,
                "pow": math.pow,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan
            }
            result = eval(expression, {"__builtins__": None}, allowed_names)
            return f"计算结果：{expression} = {result}"
        except Exception as e:
            return f"计算失败: {str(e)}"

