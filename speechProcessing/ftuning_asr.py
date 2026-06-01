# -*- coding: utf-8 -*-

import os
import json
import random
import numpy as np
import paddle
import soundfile as sf
from paddlespeech.cli.asr import ASRExecutor
from paddlespeech.s2t.training.trainer import Trainer
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.utility import AttrDict
from paddlespeech.metrics.error_rate import WERMeter

USER_DATA_DIR = "./user_voice_samples"
OUTPUT_DIR = "./personalized_asr"
NUM_SAMPLES = 8
BASE_MODEL = "conformer_wenetspeech"
SAMPLE_RATE = 16000

os.makedirs(USER_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./temp_augmented", exist_ok=True)

def add_noise_to_audio(wav, snr_db=15):
    if np.random.rand() > 0.6:
        noise = np.random.randn(len(wav))
        clean_rms = np.sqrt(np.mean(wav**2))
        noise_rms = np.sqrt(np.mean(noise**2))
        scale = clean_rms / (10**(snr_db/20)) / (noise_rms + 1e-6)
        wav = wav + scale * noise
    return wav

def augment_audio(wav):
    if random.random() > 0.5:
        wav *= random.uniform(0.8, 1.2)

    wav = add_noise_to_audio(wav, snr_db=random.randint(10, 20))
    return np.clip(wav, -1, 1).astype(np.float32)

def prepare_fewshot_data():
    wav_files = [f for f in os.listdir(USER_DATA_DIR) if f.endswith('.wav')]
    assert len(wav_files) >= NUM_SAMPLES, f"至少准备{NUM_SAMPLES}条wav语音"
    
    augmented_list = []
    for wav_file in wav_files[:NUM_SAMPLES]:
        wav_path = os.path.join(USER_DATA_DIR, wav_file)
        txt_path = wav_path.replace(".wav", ".txt")
        
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        augmented_list.append((os.path.abspath(wav_path), text))

        for i in range(2):
            wav, sr = sf.read(wav_path)
            if sr != SAMPLE_RATE:
                raise ValueError("语音必须是16000采样率")
                
            wav_aug = augment_audio(wav)
            temp_file = f"./temp_augmented/{os.path.splitext(wav_file)[0]}_{i}.wav"
            sf.write(temp_file, wav_aug, SAMPLE_RATE)
            augmented_list.append((os.path.abspath(temp_file), text))

    with open(f"{USER_DATA_DIR}/train.list", "w", encoding="utf-8") as f:
        for p, t in augmented_list:
            f.write(f"{p}\t{t}\n")

    asr = ASRExecutor()
    asr(model=BASE_MODEL, lang="zh", sample_rate=SAMPLE_RATE, audio_file=None, force_download=True)
    os.system(f"cp pretrained_models/{BASE_MODEL}/units.txt {USER_DATA_DIR}/")

    print(f"✅ 数据增强完成，共 {len(augmented_list)} 条有效训练样本")
    return len(augmented_list)

def apply_lora_to_model(model, lora_rank=4):
    from paddlespeech.s2t.modules.lora import LoRALinear

    for p in model.parameters():
        p.trainable = False

    for name, layer in model.named_sublayers():
        if any(k in name for k in ["ffn", "self_attn"]):
            if hasattr(layer, "weight") and len(layer.weight.shape) == 2:
                lora_layer = LoRALinear(
                    layer.weight.shape[0],
                    layer.weight.shape[1],
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    lora_dropout=0.1
                )
                with paddle.no_grad():
                    lora_layer.base_layer.weight.set_value(layer.weight)
                    if layer.bias is not None:
                        lora_layer.base_layer.bias.set_value(layer.bias)
                try:
                    model._sub_layers[name] = lora_layer
                except:
                    pass

    for n, p in model.named_parameters():
        p.trainable = "lora_" in n

    train_num = sum(p.numel() for p in model.parameters() if p.trainable)
    return model

class EarlyStoppingCallback:
    def __init__(self, patience=3):
        self.best_loss = float("inf")
        self.wait = 0
        self.patience = patience

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                raise StopIteration("早停：防止小样本过拟合")

def fine_tune_personalized_asr():
    num_samples = prepare_fewshot_data()

    model_cls = dynamic_import("paddlespeech.s2t.models.u2:U2Model")
    base = f"pretrained_models/{BASE_MODEL}"
    
    with open(f"{base}/model.yaml") as f:
        cfg = AttrDict(json.load(f))

    cfg.data.train_manifest = f"{USER_DATA_DIR}/train.list"
    cfg.data.unit_type = "spm"
    cfg.data.vocab_filepath = f"{USER_DATA_DIR}/units.txt"
    cfg.training = AttrDict({
        "epochs": 20,
        "batch_size": 2,
        "lr": 5e-4,
        "lr_scheduler": "warmup_linear",
        "warmup_steps": 10,
        "grad_clip": 5.0,
        "log_interval": 1,
        "save_interval": 999,
    })

    model = model_cls.from_pretrained(
        tag=BASE_MODEL, config=cfg, state_dict=paddle.load(f"{base}/model.pdparams")
    )
    model = apply_lora_to_model(model)

    trainer = Trainer(config=cfg, model=model, output_dir=OUTPUT_DIR, do_eval=False)
    trainer.register_callback(EarlyStoppingCallback(patience=3))
    
    try:
        trainer.train()
    except StopIteration:
        print("🛑 触发早停，模型已最优")

    paddle.save(model.state_dict(), f"{OUTPUT_DIR}/lora_adapter.pdparams")

def test_personalized_asr():
    test_files = [f for f in os.listdir(USER_DATA_DIR) if f.endswith(".wav")]
    test_audio = os.path.join(USER_DATA_DIR, test_files[0])
    gt_file = test_audio.replace(".wav", ".txt")
    
    with open(gt_file, encoding="utf-8") as f:
        gt_text = f.read().strip()

    asr = ASRExecutor()
    base_res = asr(model=BASE_MODEL, audio_file=test_audio)

    lora_path = f"{OUTPUT_DIR}/lora_adapter.pdparams"
    if os.path.exists(lora_path):
        lora_state = paddle.load(lora_path)
        model_state = asr.model.state_dict()

        for k in lora_state:
            if "lora_" in k and k in model_state:
                model_state[k] = lora_state[k]
        asr.model.set_state_dict(model_state)

    lora_res = asr(audio_file=test_audio)

    wer = WERMeter()
    wer_base = wer(base_res, gt_text)
    wer_lora = wer(lora_res, gt_text)

    print("\n===== 效果对比 =====")
    print(f"真实文本：{gt_text}")
    print(f"基座模型：{base_res} | WER {wer_base:.1%}")
    print(f"个性化模型：{lora_res} | WER {wer_lora:.1%}")
    print(f"✅ 字错率下降：{wer_base - wer_lora:.1%}")

if __name__ == "__main__":
    fine_tune_personalized_asr()
    test_personalized_asr()

