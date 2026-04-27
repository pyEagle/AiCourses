# -*- coding: utf-8 -*-

import torch
import requests
import time
import json
from typing import List, Dict

class DeviceManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OllamaClient:
    def __init__(self, model="deepseek:latest", host="http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, prompt):
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"[Ollama 错误] {e}")
            return ""


class Task:
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset

    def build_prompt(self, item):
        return f"请直接给出结果。问题: {item['input']}\n答案:"


class Evaluator:
    def __init__(self):
        pass

    def exact_match(self, pred: str, target: str) -> int:
        return int(pred.strip().lower() == target.strip().lower())

    def evaluate(self, preds, targets):
        if not preds: return 0.0
        scores = [self.exact_match(p, t) for p, t in zip(preds, targets)]
        return sum(scores) / len(scores)


class Metrics:
    def __init__(self):
        self.records = []

    def add(self, latency):
        self.records.append(latency)

    def summary(self):
        if not self.records:
            return {"avg_latency": 0, "num_samples": 0}
        avg = sum(self.records) / len(self.records)
        return {
            "avg_latency": avg,
            "num_samples": len(self.records)
        }


class Runner:
    def __init__(self, model_client, evaluator):
        self.model = model_client
        self.evaluator = evaluator

    def run(self, task: Task):
        print(f"\n[执行器] 正在运行任务: {task.name}")

        preds = []
        targets = []
        metrics = Metrics()

        for idx, item in enumerate(task.dataset):
            prompt = task.build_prompt(item)

            start = time.time()
            pred = self.model.generate(prompt)
            latency = time.time() - start

            metrics.add(latency)

            preds.append(pred)
            targets.append(item["target"])

            print(f"\n--- 样本 {idx} ---")
            print(f"提示词: {prompt.replace('\n', ' ')}")
            print(f"预测结果: {pred}")
            print(f"标准答案: {item['target']}")
            print(f"响应耗时: {latency:.2f}秒")

        acc = self.evaluator.evaluate(preds, targets)
        stat = metrics.summary()

        print("\n" + "="*30)
        print(f"[任务评测完成: {task.name}]")
        print(f"准确率: {acc:.4f}")
        print(f"平均耗时: {stat['avg_latency']:.2f}秒")
        print("="*30)

        return {
            "任务名": task.name,
            "准确率": acc,
            "平均耗时": stat['avg_latency'],
            "样本数": stat['num_samples']
        }


def build_demo_task():
    dataset = [
        {"input": "1+1=?", "target": "2"},
        {"input": "2+2=?", "target": "4"},
        {"input": "3+5=?", "target": "8"},
        {"input": "10-3=?", "target": "7"},
    ]
    return Task("基础算术评测", dataset)


def main():
    device_manager = DeviceManager()
    model = OllamaClient() 
    evaluator = Evaluator()

    runner = Runner(model, evaluator)

    task = build_demo_task()

    result = runner.run(task)

    print("\n[最终 JSON 统计结果]")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

