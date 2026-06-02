import torch
import json
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"训练开始，当前设备: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

model_id = "openai/whisper-small" # 或 whisper-large-v3
processor = WhisperProcessor.from_pretrained(model_id, language="Chinese", task="transcribe")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def prepare_dataset(example):
    audio_path = example["audio"]
    array, sr = librosa.load(audio_path, sr=16000)
    
    example["input_features"] = processor.feature_extractor(array, sampling_rate=16000).input_features[0]
    example["labels"] = processor.tokenizer(example["sentence"]).input_ids
    return example

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = WhisperForConditionalGeneration.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=32, lora_alpha=64, 
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"], 
    lora_dropout=0.05, bias="none"
)
model = get_peft_model(model, config)

dataset = load_dataset("json", data_files="metadata.json", split="train")

dataset = dataset.map(prepare_dataset, remove_columns=["audio", "sentence"])

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-lora-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-3,
    warmup_steps=50,
    max_steps=500,
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    predict_with_generate=True,
    generation_max_length=225,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("开始训练...")
trainer.train()
print("训练完成！")
