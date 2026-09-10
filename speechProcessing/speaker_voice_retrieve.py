import os
import sys
import glob
import math
import argparse
import numpy as np
import threading
import collections
import scipy.io.wavfile as wavfile
import scipy.signal as signal

from speaker_vad import robust_vad as unified_robust_vad

SYSTEM_VERSION = "3.5.0-RK3588-Ultimate-End2End"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from funasr import AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch / FunASR missing.")

try:
    import torchaudio.compliance.kaldi as kaldi
    KALDI_AVAILABLE = "torchaudio"
except ImportError:
    try:
        import kaldi_native_fbank as knf
        KALDI_AVAILABLE = "kaldi_native_fbank"
    except ImportError:
        KALDI_AVAILABLE = None

try:
    from rknn.api import RKNN
    RKNN_TOOLKIT_AVAILABLE = True
except ImportError:
    RKNN_TOOLKIT_AVAILABLE = False

try:
    from rknnlite.api import RKNNLite
    RKNNLITE_AVAILABLE = True
except ImportError:
    RKNNLITE_AVAILABLE = False


if TORCH_AVAILABLE:
    class ResidualProjector(nn.Module):
        def __init__(self, dim=192, hidden_dim=256):
            super(ResidualProjector, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            )
            
        def forward(self, x):
            return self.net(x)


class VoicePrintSystem:
    def __init__(self, model_file="iic/speech_eres2netv2_sv_zh-cn_16k-common", engine="pytorch", db_path="./model/my_vp_db.npz"):
        self.engine = engine
        self.model_file = model_file
        self.embedding_dim = 192
        self.max_samples_per_spk = 10 
        self.fixed_frames = 150
        self.db_path = db_path if db_path.endswith('.npz') else db_path + ".npz"
        
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.write_lock = threading.RLock()
        
        self.db_features = []
        self.db_labels = []
        self.projector = None

        if self.engine == "pytorch":
            if not TORCH_AVAILABLE: 
                raise RuntimeError("PyTorch missing.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModel(model=model_file, disable_update=True)
            
            self.projector = ResidualProjector(dim=self.embedding_dim).to(self.device)
            
            weight_paths = [
                "./checkpoints/best_speaker_model.pth", 
                "./voice_saved_weights/projector_best.pt",
                "./projector_best.pt"
            ]
            
            loaded = False
            for wp in weight_paths:
                if os.path.exists(wp):
                    state = torch.load(wp, map_location=self.device)
                    if 'model_state_dict' in state:
                        self.projector.load_state_dict(state['model_state_dict'])
                    elif 'projector' in state:
                        self.projector.load_state_dict(state['projector'])
                    else:
                        self.projector.load_state_dict(state)
                        
                    self.projector.eval()
                    loaded = True
                    break
                    
            if not loaded:
                self.projector = None
                
        elif self.engine == "rknn":
            if not RKNNLITE_AVAILABLE: 
                raise RuntimeError("RKNNLite missing.")
            self.rknn_lite = RKNNLite()
            self.rknn_lite.load_rknn("./speaker_combined_model.rknn")
            self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)

        self.load_voice_print(self.db_path)

    def load_voice_print(self, load_path):
        db_path = load_path if load_path.endswith('.npz') else load_path + ".npz"
        if os.path.exists(db_path):
            try:
                data = np.load(db_path)
                self.db_features = list(data['features'])
                self.db_labels = list(data['labels'])
            except Exception as e:
                print(f"⚠️ 警告: 无法加载 {db_path}，视为空声纹库重新开始。错误信息: {e}")
                self.db_features = []
                self.db_labels = []

    def save_voice_print(self, save_path=None):
        db_path = save_path if save_path else self.db_path
        db_path = db_path if db_path.endswith('.npz') else db_path + ".npz"
        base_path = db_path.replace('.npz', '')
        index_dummy = base_path + ".index"
        
        with self.write_lock:
            tmp_path = db_path + ".tmp.npz"
            np.savez(tmp_path, features=self.db_features, labels=self.db_labels)
            os.replace(tmp_path, db_path)
            open(index_dummy, 'w').close()

    def register(self, audio_path, name, auto_rebuild=True, apply_vad=True):
        norm_emb = self.voice_print(audio_path, apply_vad=apply_vad)
        with self.write_lock:
            if name in self.db_labels:
                count = self.db_labels.count(name)
                if count >= self.max_samples_per_spk:
                    idx = self.db_labels.index(name)
                    self.db_features.pop(idx)
                    self.db_labels.pop(idx)
            self.db_features.append(norm_emb)
            self.db_labels.append(name)
        self.save_voice_print()

    @staticmethod
    def _l2_normalize(vec):
        v = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v if norm < 1e-10 else v / norm

    def robust_vad(self, data, sr=16000, frame_len=0.03, hop_len=0.015):
        return unified_robust_vad(data, sr=sr, frame_len=frame_len, hop_len=hop_len)

    def preprocessing_in_memory(self, input_audio_path, apply_vad=True):
        sr, data = wavfile.read(input_audio_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        orig_dtype = data.dtype
        if orig_dtype == np.int16:
            data = data.astype(np.float32)
        elif orig_dtype == np.int32:
            data = data.astype(np.float32) / 65536.0 
        else:
            data = data.astype(np.float32)
            if data.max() <= 1.0 and data.min() >= -1.0:
                data = data * 32768.0

        if sr != 16000:
            data = signal.resample_poly(data, 16000, sr)
            sr = 16000
            
        if apply_vad:
            data = self.robust_vad(data, sr=sr)
            if len(data) < sr * 0.3: 
                raise ValueError("Audio too short after VAD")
        
        return data

    def _extract_raw_embedding(self, audio_path, apply_vad=True):
        clean_audio = self.preprocessing_in_memory(audio_path, apply_vad=apply_vad)
        
        if KALDI_AVAILABLE == "torchaudio":
            wav_tensor = torch.from_numpy(clean_audio).unsqueeze(0)
            fbank = kaldi.fbank(wav_tensor, num_mel_bins=80, frame_length=25, frame_shift=10)
            fbank = fbank.numpy()
        elif KALDI_AVAILABLE == "kaldi_native_fbank":
            import kaldi_native_fbank as knf
            opts = knf.FbankOptions()
            opts.frame_opts.samp_freq, opts.frame_opts.frame_length_ms, opts.frame_opts.frame_shift_ms = 16000, 25, 10
            opts.mel_opts.num_bins = 80
            fbank_extractor = knf.OnlineFbank(opts)
            fbank_extractor.accept_waveform(16000, clean_audio.astype(np.float32))
            fbank_extractor.input_finished()
            fbank = np.stack([fbank_extractor.get_frame(i) for i in range(fbank_extractor.num_frames_ready)])
        else:
            raise RuntimeError("kaldi feature extractor missing")

        fbank = fbank - fbank.mean(axis=0, keepdims=True)

        if self.engine == "pytorch":
            with torch.no_grad():
                fbank_t = torch.from_numpy(fbank).unsqueeze(0).to(self.device)
                res = self.model.model(fbank_t)
                raw_emb = res[0] if isinstance(res, tuple) else (res.get('spk_embedding', res) if isinstance(res, dict) else res)
                return raw_emb.flatten().cpu().numpy().astype(np.float32)
        else:
            if fbank.shape[0] < self.fixed_frames:
                pad_total = self.fixed_frames - fbank.shape[0]
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                fbank = np.pad(fbank, ((pad_left, pad_right), (0, 0)), mode='edge')
            elif fbank.shape[0] > self.fixed_frames:
                fbank = fbank[:self.fixed_frames, :]
                
            input_feat = np.expand_dims(fbank, 0).astype(np.float32)
            feature_lengths = np.array([self.fixed_frames], dtype=np.int32)
            outputs = self.rknn_lite.inference(inputs=[input_feat, feature_lengths])
            return outputs[0].flatten().astype(np.float32)

    def voice_print(self, audio_path, apply_vad=True):
        emb = self._extract_raw_embedding(audio_path, apply_vad=apply_vad)
        if self.engine == "pytorch" and self.projector is not None:
            with torch.no_grad():
                t = torch.from_numpy(emb).float().unsqueeze(0).to(self.device)
                emb = self.projector(t).squeeze(0).cpu().numpy()
        return self._l2_normalize(emb)

    def search(self, audio_path, topk=1, threshold=0.6, margin=0.05, apply_vad=True):
        try:
            query_emb = self.voice_print(audio_path, apply_vad=apply_vad)
        except Exception:
            return [("UNKNOWN", 0.0)]
            
        with self.write_lock:
            if not self.db_features: 
                return [("UNKNOWN", 0.0)]
            db_feats = np.array(self.db_features, dtype=np.float32)
            db_lbls = list(self.db_labels)
            
        speaker_embs = collections.defaultdict(list)
        for feat, label in zip(db_feats, db_lbls):
            speaker_embs[label].append(feat)

        centroids, unique_labels = [], []
        for label, feats in speaker_embs.items():
            centroids.append(self._l2_normalize(np.mean(feats, axis=0)))
            unique_labels.append(label)

        sims = np.dot(np.stack(centroids), query_emb)
        results = [(unique_labels[i], float(sims[i])) for i in range(len(unique_labels))]
        results.sort(key=lambda x: x[1], reverse=True)
        
        if not results or results[0][1] < threshold:
            return [("UNKNOWN", results[0][1] if results else 0.0)]
        
        return results[:topk]
