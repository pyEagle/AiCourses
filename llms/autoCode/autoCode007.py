import json
import os
import subprocess
import requests
import re
import ast

class CodeGenerator:
    def __init__(self, user_description):
        self.user_description = user_description
        self.ollama_url = "http://localhost:11434/api/generate" 
        
        self.dangerous_operations = [
            r'os\.remove\(.+\)',
            r'os\.unlink\(.+\)',
            r'shutil\.rmtree\(.+\)',
            r'subprocess\.run\(["\']rm ["\'].*\)',
            r'os\.system\(.+\)'
        ]

        self.memory = []
        self.max_memory = 5

        self.safe_dir = os.path.abspath("./sandbox")
        os.makedirs(self.safe_dir, exist_ok=True)

    def update_memory(self, prompt, response, error=None):
        entry = {
            "prompt": prompt,
            "response": response,
            "error": error
        }
        self.memory.append(entry)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def build_context(self):
        context = ""
        for m in self.memory:
            context += f"\n历史Prompt: {m['prompt']}\n历史错误: {m['error']}\n"
        return context

    def generate_code(self, prompt):
        context = self.build_context()
        full_prompt = context + "\n当前任务:\n" + prompt

        payload = {
            "model": "deepseek-r1:latest",
            "prompt": full_prompt,
            "format": "json",
            "stream": False
        }
        response = requests.post(self.ollama_url, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"Ollama 调用失败: {response.text}")

    def is_dangerous_code(self, code):
        for pattern in self.dangerous_operations:
            if re.search(pattern, code):
                return True
        return False

    def is_dangerous_code_ast(self, code):
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if hasattr(node.func, 'attr'):
                        if node.func.attr in ['remove', 'unlink', 'rmtree', 'system']:
                            return True
            return False
        except:
            return True

    def save_code(self, output):
        path = self.safe_dir
        for file in output["files"]:
            code = file["code"]

            if self.is_dangerous_code(code) or self.is_dangerous_code_ast(code):
                raise Exception("❌ 检测到危险操作")

            filename = os.path.join(path, file["pyfile"])
            with open(filename, "w") as f:
                f.write(code)

        return output["main"]

    def run_code(self, main_command):
        try:
            result = subprocess.run(
                main_command.split(),   # ❗ 禁止 shell=True
                capture_output=True,
                text=True,
                timeout=5               # ❗ 防止死循环
            )
            if result.returncode == 0:
                print("✅ 代码执行成功")
                return True, result.stdout
            else:
                print(f"❌ 执行失败:\n{result.stderr}")
                return False, result.stderr
        except Exception as e:
            print(f"❌ 执行异常: {e}")
            return False, str(e)

    def validate_output(self, output):
        required_fields = ["path", "files", "main"]
        for field in required_fields:
            if field not in output:
                raise ValueError(f"缺少字段: {field}")
        return True

    def fix_and_retry(self, prompt, max_retries=5):
        for i in range(max_retries):
            print(f"尝试第 {i + 1} 次生成...")

            code_output = self.generate_code(prompt)
            self.update_memory(prompt, code_output)

            try:
                output = json.loads(code_output)

                print(output)
                print('--'*20)

                self.validate_output(output)

                main_command = self.save_code(output)

                success, log = self.run_code(main_command)

                if success:
                    return output, log
                else:
                    self.update_memory(prompt, code_output, log)

                    correct_example = '{"path": "./", "files": [{"code": "def add(a, b):\\n    return a + b\\n", "pyfile": "test.py"}], "main": "python test.py"}'

                    prompt = f"""
用户需求: {self.user_description}
错误信息: {log}
请修复代码并重新输出结构化 JSON
参考示例: {correct_example}
"""

            except json.JSONDecodeError:
                print("❌ JSON错误")

                correct_example = '{"path": "./", "files": [{"code": "def add(a, b):\\n    return a + b\\n", "pyfile": "test.py"}], "main": "python test.py"}'

                prompt = f"""
用户需求: {self.user_description}
错误: JSON格式无效
请输出合法JSON
参考示例: {correct_example}
"""

            except Exception as e:
                print(f"❌ 异常: {e}")

                prompt = f"""
用户需求: {self.user_description}
错误: {str(e)}
请修复问题并重新输出
"""

        raise Exception("❌ 达到最大重试次数")

def main():
    input_str = "创建一个函数，计算两个数的和，并在test.py中调用它。"

    correct_example = '{"path": "./", "files": [{"code": "def add(a, b):\\n    return a + b\\n", "pyfile": "test.py"}], "main": "python test.py"}'

    prompt = f"""
角色：资深开发
任务：{input_str}

要求：
1. Python代码
2. JSON结构输出
3. 可执行
参考：{correct_example}
"""

    generator = CodeGenerator(input_str)

    final_output, result_log = generator.fix_and_retry(prompt)

    print("最终输出:")
    print(json.dumps(final_output, indent=2))

    print("执行结果:")
    print(result_log)


if __name__ == "__main__":
    main()
