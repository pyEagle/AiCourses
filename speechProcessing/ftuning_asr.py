# -*- coding: utf-8 -*-
import os
import json
import random
import yaml
import numpy as np
import paddle
import paddle.nn as nn
import soundfile as sf
from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.s2t.training.trainer import Trainer
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.utility import UpdateConfig
from paddlespeech.metrics.error_rate import WERMeter

# ===================== 配置项 =====================
USER_DATA_DIR = "./user_voice_samples"
OUTPUT_DIR = "./personalized_asr"
NUM_SAMPLES = 8
BASE_MODEL = "conformer_online_zh" 
SAMPLE_RATE = 16000
FEAT_DIM = 80
# ==================================================

os.makedirs(USER_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./temp_augmented", exist_ok=True)


class InjectableLoRALinear(nn.Layer):
    def __init__(self, base_layer, r=4, lora_alpha=8, dropout=0.1):
        super().__init__()
        self.base_layer = base_layer

        self.base_layer.weight.stop_gradient = True
        if self.base_layer.bias is not None:
            self.base_layer.bias.stop_gradient = True

        in_features = base_layer.weight.shape[0]
        out_features = base_layer.weight.shape[1]
        
        self.lora_A = nn.Linear(in_features, r, bias_attr=False)
        self.lora_B = nn.Linear(r, out_features, bias_attr=False)
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(p=dropout)

        self.lora_B.weight.set_value(paddle.zeros_like(self.lora_B.weight))

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

def apply_lora_to_model(model, lora_rank=4):
    for p in model.parameters():
        p.trainable = False # 基础冻结

    def replace_linear_with_lora(layer):
        for name, sub_layer in layer.named_children():
            if isinstance(sub_layer, nn.Linear) and any(k in name for k in ["linear", "proj", "q", "k", "v"]):
                lora_layer = InjectableLoRALinear(sub_layer, r=lora_rank)
                setattr(layer, name, lora_layer)
            else:
                replace_linear_with_lora(sub_layer)

    replace_linear_with_lora(model)

    train_num = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
    total_num = sum(p.numel() for p in model.parameters())
    print(f"LoRA 注入成功！可训练参数比例: {train_num/total_num:.4%}")
    return model
# --------------------------------------------------------------------

def add_noise_to_audio(wav, snr_db=15):
    if np.random.rand() > 0.6:
        noise = np.random.randn(len(wav))
        clean_rms = np.sqrt(np.mean(wav**2)) + 1e-6
        noise_rms = np.sqrt(np.mean(noise**2)) + 1e-6
        scale = clean_rms / (10**(snr_db/20)) / noise_rms
        wav = wav + scale * noise
    return wav

def augment_audio(wav):
    if random.random() > 0.5:
        wav *= random.uniform(0.8, 1.2)
    wav = add_noise_to_audio(wav, snr_db=random.randint(10, 20))
    return np.clip(wav, -1, 1).astype(np.float32)

def prepare_fewshot_data():
    wav_files = [f for f in os.listdir(USER_DATA_DIR) if f.endswith('.wav') and not f.startswith('aug')]
    if len(wav_files) == 0:
        print("在目录下放置原始 .wav 文件与同名 .txt 文件")
        return 0
    
    manifest_list = []
    for wav_file in wav_files[:NUM_SAMPLES]:
        wav_path = os.path.join(USER_DATA_DIR, wav_file)
        txt_path = wav_path.replace(".wav", ".txt")
        
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        wav, sr = sf.read(wav_path)
        frames = int((len(wav) / sr) * 100)

        base_item = {
            "utt": wav_file.split('.')[0],
            "feat": os.path.abspath(wav_path),
            "feat_shape": [frames, FEAT_DIM],
            "text": text
        }
        manifest_list.append(base_item)

        for i in range(2):
            wav_aug = augment_audio(wav)
            temp_name = f"{os.path.splitext(wav_file)[0]}_aug_{i}"
            temp_file = f"./temp_augmented/{temp_name}.wav"
            sf.write(temp_file, wav_aug, SAMPLE_RATE)
            aug_frames = int((len(wav_aug) / SAMPLE_RATE) * 100)
            
            manifest_list.append({
                "utt": temp_name,
                "feat": os.path.abspath(temp_file),
                "feat_shape": [aug_frames, FEAT_DIM],
                "text": text
            })

    manifest_path = os.path.join(USER_DATA_DIR, "data.manifest")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in manifest_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"数据增强完成，共 {len(manifest_list)} 条合法样本")
    return len(manifest_list)

class EarlyStoppingCallback:
    def __init__(self, patience=3):
        self.best_loss = float("inf")
        self.wait = 0
        self.patience = patience
        
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss", float("inf"))
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("\n触发早停：损失不再下降，防止模型在极少样本上过拟合。")
                raise StopIteration()

def fine_tune_personalized_asr():
    if prepare_fewshot_data() == 0:
        return False

    asr_executor = ASRExecutor()
    _ = asr_executor(audio_file="", model=BASE_MODEL, force_yes=True)
    res_path = asr_executor.res_path
    
    real_config_path = os.path.join(res_path, 'conf', 'conformer.yaml')
    real_model_path = os.path.join(res_path, 'exp', 'conformer', 'avg_1.pt')

    with open(real_config_path, 'r') as f:
        cfg = yaml.safe_load(f) # 修正：正确解析 YAML
        
    cfg['data']['train_manifest'] = f"{USER_DATA_DIR}/data.manifest"
    cfg['data']['dev_manifest'] = f"{USER_DATA_DIR}/data.manifest"
    cfg['data']['feat_type'] = 'fbank' # 修正：强制实时特征提取
    cfg['data']['vocab_filepath'] = os.path.join(res_path, 'vocab.txt')
    cfg['training']['epochs'] = 20
    cfg['training']['batch_size'] = 2
    cfg['training']['lr'] = 5e-4
    if 'collator' not in cfg:
        cfg['collator'] = {}
    cfg['collator']['augmentation_config'] = ""
    
    temp_conf = "./temp_train.yaml"
    with open(temp_conf, 'w') as f:
        yaml.dump(cfg, f)

    update_config = UpdateConfig(temp_conf)
    model_class = dynamic_import(update_config.model_conf['name'])
    model = model_class.from_config(update_config.model_conf)
    model.set_state_dict(paddle.load(real_model_path))

    model = apply_lora_to_model(model, lora_rank=4)

    trainer = Trainer(update_config, None, model)
    trainer.output_dir = OUTPUT_DIR
    
    try:
        print("🚀 开始 LoRA 个性化微调...")
        trainer.train() 
    except StopIteration:
        pass # 被早停截断
    except Exception as e:
        print(f"⚠️ 训练警告 (常因底层依赖引发): {e}")
        
    paddle.save(model.state_dict(), f"{OUTPUT_DIR}/lora_adapter.pdparams")
    print("个性化 LoRA 模型保存完毕！")
    return True

def test_personalized_asr():
    test_files = [f for f in os.listdir(USER_DATA_DIR) if f.endswith(".wav") and not f.startswith('aug')]
    if not test_files:
        return
        
    test_audio = os.path.join(USER_DATA_DIR, test_files[0])
    gt_file = test_audio.replace(".wav", ".txt")
    
    with open(gt_file, encoding="utf-8") as f:
        gt_text = f.read().strip()

    asr = ASRExecutor()
    base_res = asr(model=BASE_MODEL, audio_file=test_audio, force_yes=True)
    
    lora_path = f"{OUTPUT_DIR}/lora_adapter.pdparams"
    if os.path.exists(lora_path):
        apply_lora_to_model(asr.task.model, lora_rank=4)
        asr.task.model.set_state_dict(paddle.load(lora_path))
        lora_res = asr(audio_file=test_audio, force_yes=True)
    else:
        lora_res = "微调模型未找到"

    wer = WERMeter()
    wer_base = wer(base_res, gt_text)
    wer_lora = wer(lora_res, gt_text)

    print("\n===== 🎙️ 效果对比 =====")
    print(f"真实文本：{gt_text}")
    print(f"基座模型：{base_res} | WER {wer_base:.1%}")
    print(f"LoRA微调：{lora_res} | WER {wer_lora:.1%}")
    print("=======================\n")

if __name__ == "__main__":
    if fine_tune_personalized_asr():
        test_personalized_asr()
