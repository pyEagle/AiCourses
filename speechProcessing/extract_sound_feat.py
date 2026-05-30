# -*- coding:utf-8 -*-

import webrtcvad
import torch
import numpy as np

from speechbrain.pretrained import EncoderClassifier


class VoiceFeatureExtractor:
    def __init__(self, sample_rate=16000, vad_aggressiveness=3):
        self.SR = sample_rate
        if self.SR not in [8000, 16000, 32000, 48000]:
            raise ValueError("VAD 模块要求采样率必须是 8000, 16000, 32000 或 48000 Hz")
            
        self.MAX_INT16 = np.iinfo(np.int16).max

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vad = webrtcvad.Vad(vad_aggressiveness)
        # 16000Hz 下，30ms 对应的采样点数是 480，字节数是 960 (int16 占 2 bytes)
        self.frame_duration_ms = 30
        self.frame_size_bytes = int(self.SR * (self.frame_duration_ms / 1000.0) * 2)

        # ECAPA-TDNN模型
        try:
            self.model = EncoderClassifier.from_hparams(
                source="model/speechbrain/spkrec-ecapa-voxceleb",
                savedir="model/pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self.device)}
            )
            self.model.eval()
        except Exception as e:
            raise

    def filter_silence(self, pcm_bytes):
        active_audio_bytes = bytearray()
        
        for i in range(0, len(pcm_bytes) - self.frame_size_bytes + 1, self.frame_size_bytes):
            chunk = pcm_bytes[i : i + self.frame_size_bytes]
            if self.vad.is_speech(chunk, self.SR):
                active_audio_bytes.extend(chunk)
                
        dropped_ratio = 1.0 - (len(active_audio_bytes) / (len(pcm_bytes) + 1e-8))
        
        return bytes(active_audio_bytes)

    def extract_embedding_from_bytes(self, pcm_bytes):
        if not pcm_bytes:
            return None

        clean_pcm_bytes = self.filter_silence(pcm_bytes)
        
        if len(clean_pcm_bytes) < self.frame_size_bytes:
            return None

        audio_int16 = np.frombuffer(clean_pcm_bytes, dtype=np.int16)
        y = audio_int16.astype(np.float32) / self.MAX_INT16

        if len(y) < self.SR: 
            print(f'有效人声较短 ({len(y)/self.SR:.2f}秒)，特征区分度可能下降')

        try:
            signal_tensor = torch.from_numpy(y).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding_tensor = self.model.encode_batch(signal_tensor)
            
            embedding = embedding_tensor.squeeze().cpu().numpy()

            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            return None



if __name__ == "__main__":
    v = VoiceFeatureExtractor()
    v.extract_embedding_from_bytes()

