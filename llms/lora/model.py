# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from globalSetting import device


class WeatherLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=1):
        super(WeatherLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
 
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True  # 输入形状：(batch_size, seq_len, input_dim)
        )
 
        self.fc = nn.Linear(hidden_dim, output_dim)
 
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
 
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)  # (batch_size, output_dim)
        return out
 
