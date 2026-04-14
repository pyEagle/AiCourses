# -*- coding:utf-8 -*-

import torch
 
import numpy as np
import matplotlib.pyplot as plt
 
from torch.utils.data import Dataset, DataLoader
 
 
class WeatherDataset(Dataset):
    def __init__(self, seq_len=6, pred_len=1, num_samples=1000):
        self.seq_len = seq_len
        self.pred_len = pred_len
 
        np.random.seed(42)  # 固定随机种子，结果可复现
        time = np.linspace(0, 100, num_samples + seq_len)
 
        temp = 20 + 5 * np.sin(time) + np.random.randn(len(time)) * 0.5
        humidity = 60 - 8 * np.sin(time) + np.random.randn(len(time)) * 0.8
        pressure = 1013 + np.random.randn(len(time)) * 0.3
 
        self.data = np.stack([temp, humidity, pressure], axis=1).astype(np.float32)
 
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
 
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len, :]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, 0]
        return torch.from_numpy(x), torch.from_numpy(y)
 
