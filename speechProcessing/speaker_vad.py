cat speaker_vad.py 
# -*- coding:utf-8 -*-

import numpy as np
from collections import deque

class SpeakerVAD:
    def __init__(self, chunk_dur, energy_thres, silence_dur, min_dur, max_dur, pre_roll_len):
        self.chunk_dur = chunk_dur
        self.energy_thres = energy_thres
        self.silence_limit = int(silence_dur / chunk_dur)
        self.min_chunks = int(min_dur / chunk_dur)
        self.max_chunks = int(max_dur / chunk_dur)
        self.pre_roll_len = pre_roll_len

    def collect(self, audio_stream):
        pre_roll = deque(maxlen=self.pre_roll_len)
        audio_frames = []
        is_speaking = False
        silence_count = 0
        
        while True:
            try:
                chunk = next(audio_stream)
            except StopIteration:
                break
                
            chunk_arr = np.frombuffer(chunk, dtype=np.int16)
            rms = np.sqrt(np.mean(chunk_arr.astype(np.float32)**2))
            
            if not is_speaking:
                if rms > self.energy_thres:
                    is_speaking = True
                    audio_frames.extend(list(pre_roll))
                    audio_frames.append(chunk)
                    silence_count = 0
                else:
                    pre_roll.append(chunk)
            else:
                audio_frames.append(chunk)
                
                if rms < self.energy_thres:
                    silence_count += 1
                else:
                    silence_count = 0
                    
                if silence_count >= self.silence_limit or len(audio_frames) >= self.max_chunks:
                    if len(audio_frames) >= self.min_chunks:
                        yield b"".join(audio_frames)
                    
                    is_speaking = False
                    audio_frames = []
                    silence_count = 0
                    pre_roll.clear()

def robust_vad(data, sr=16000, frame_len=0.03, hop_len=0.015):
    frame_length, hop_length = int(sr * frame_len), int(sr * hop_len)
    data_f = data.astype(np.float32)
    num_frames = 1 + (len(data_f) - frame_length) // hop_length
    if num_frames < 1: return data
    frames = np.lib.stride_tricks.as_strided(
        data_f, shape=(num_frames, frame_length),
        strides=(data_f.strides[0] * hop_length, data_f.strides[0])
    )
    energies = np.mean(frames**2, axis=1)
    smoothed_energies = np.convolve(energies, np.ones(5)/5, mode='same')
    
    dynamic_energy_th = max(1e-5, np.median(smoothed_energies) * 2.0)
    active_frames = np.where(smoothed_energies > dynamic_energy_th)[0]
    
    if len(active_frames) == 0: return data
    start_idx = max(0, active_frames[0] * hop_length - int(sr*0.1))
    end_idx = min(len(data), active_frames[-1] * hop_length + frame_length + int(sr*0.2))

    return data[start_idx:end_idx]
