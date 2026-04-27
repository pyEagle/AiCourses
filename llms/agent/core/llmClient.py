import ollama

from config.settings import OLLAMA_CONFIG

class OllamaClient:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_CONFIG["base_url"])
        self.model = OLLAMA_CONFIG["model"]
        self.temperature = OLLAMA_CONFIG["temperature"]
        self.max_tokens = OLLAMA_CONFIG["max_tokens"]

    def generate(self, prompt, history=None):
        try:
            messages = []
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.max_tokens
                }
            )
            return response["message"]["content"]
        except Exception as e:
            raise Exception(f"调用Ollama失败: {str(e)}")

    def get_model_info(self):
        return self.client.show(model=self.model)

