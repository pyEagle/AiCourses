import json
import os
import subprocess
import requests
import re

correct_example = '{"path": "./", "files": [{"code": "def add(a, b):\\n    return a + b\\n", "pyfile": "test.py"}], "main": "python test.py"}'

fix_prompt = f"用户需求:{}\n上一次生成代码:{}\n错误信息:{}\n请输出包含必需字段的 JSON。正确示例:{}"
class CodeGenerator:
    def __init__(self, user_description):
        self.user_description = user_description
        self.ollama_url = "http://localhost:11434/api/generate" 
        self.dangerous_operations = [
            r'os\.remove\(.+\)',
            r'os\.unlink\(.+\)',
            r'shutil\.rmtree\(.+\)',
            r'subprocess\.run\(["\']rm ["\'].*\)',
            r'subprocess\.run\["rmdir ["\'].*\)',
            r'os\.system\["rmdir ["\'].*\)'
        ]

    def generate_code(self, prompt):
        payload = {
            "model": "deepseek-r1:latest",
            "prompt": prompt,
            "format": "json",
            "stream": False
        }
        response = requests.post(self.ollama_url, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"Ollama 调用失败: {response.text}")

    def is_dangerous_code(self, code):
        """检查代码是否包含危险操作"""
        for pattern in self.dangerous_operations:
            if re.search(pattern, code):
                return True
        return False

    def save_code(self, output):
        path = output["path"]
        for file in output["files"]:
            if self.is_dangerous_code(file["code"]):
                raise Exception("❌ 检测到危险操作：代码尝试删除系统文件")
            
            filename = os.path.join(path, file["pyfile"])
            with open(filename, "w") as f:
                f.write(file["code"])
        return output["main"]

    def run_code(self, main_command):
        try:
            result = subprocess.run(
                main_command, shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                print("✅ 代码执行成功")
                return True, result.stdout
            else:
                print(f"❌ 代码执行失败:\n{result.stderr}")
                return False, result.stderr
        except Exception as e:
            print(f"❌ 执行异常: {e}")
            return False, str(e)

    def validate_output(self, output):
        """验证输出是否包含必需的字段"""
        required_fields = ["path", "files", "main"]
        for field in required_fields:
            if field not in output:
                raise ValueError(f"生成的 JSON 缺少必需字段: '{field}'")
        return True

    def fix_and_retry(self, prompt, max_retries=5):
        for i in range(max_retries):
            print(f"尝试第 {i + 1} 次生成...")
            code_output = self.generate_code(prompt)
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
                    prompt = fix_prompt.format(self.user_description, code_output, log, correct_example)
            except json.JSONDecodeError:
                prompt = fix_prompt.format(self.user_description, code_output, "", correct_example)
            except Exception as e:
                prompt = fix_prompt.format(self.user_description, code_output, e, correct_example)

        raise Exception("❌ 达到最大重试次数，无法生成有效代码")

def main():
    input_str = "创建一个函数，计算两个数的和，并在test.py中调用它。"
    prompt = f"""
角色：你是一位资深软件开发工程师，在这个领域深耕20年了。
背景：开发一个根据用户需求，自动生成完整可执行代码
功能：{input_str}
要求：1.所有代码必须是python写的，且完整可执行
      2.结构化输出，参考用例 {correct_example}
      3.执行python test.py，输出测试效果
      4.如果报错，将错误信息给deepseek-r1:latest， deepseek-r1:latest再次按照2.结构化输出，直到代码运行成功，且实现了业务逻辑。
输出：
    1.实现功能所需要的代码
    2.结构化输出，参考用例 {correct_example}
    """
    generator = CodeGenerator(input_str)
    final_output, result_log = generator.fix_and_retry(prompt)
    print("最终输出:", json.dumps(final_output, indent=2))
    print("执行结果:", result_log)

if __name__ == "__main__":
    main()
