# -*- coding:utf-8 -*-

import torch
import math


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TorchLiteMFCC] Using device: {DEVICE}")


def hz_to_mel(freq):
    return 2595 * torch.log10(1 + freq / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def build_mel_filter(
    sr, n_fft, n_mels=40, fmin=0, fmax=None
):
    if fmax is None:
        fmax = sr / 2

    # 频率点
    mel_min = hz_to_mel(torch.tensor(fmin, device=DEVICE))
    mel_max = hz_to_mel(torch.tensor(fmax, device=DEVICE))

    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=DEVICE)
    hz_points = mel_to_hz(mel_points)

    # FFT bins
    bins = torch.floor((n_fft + 1) * hz_points / sr).long()

    filter_bank = torch.zeros(n_mels, n_fft // 2 + 1, device=DEVICE)

    for m in range(1, n_mels + 1):
        f_left = bins[m - 1]
        f_center = bins[m]
        f_right = bins[m + 1]

        for k in range(f_left, f_center):
            filter_bank[m - 1, k] = (k - f_left) / (f_center - f_left + 1e-8)

        for k in range(f_center, f_right):
            filter_bank[m - 1, k] = (f_right - k) / (f_right - f_center + 1e-8)

    return filter_bank


def stft(signal, n_fft=400, hop_length=160):
    window = torch.hann_window(n_fft).to(DEVICE)

    signal_length = signal.shape[0]
    frames = []

    for i in range(0, signal_length - n_fft, hop_length):
        frame = signal[i:i + n_fft]
        frame = frame * window

        fft = torch.fft.rfft(frame)
        frames.append(fft)

    return torch.stack(frames)  # [T, F]


def power_spectrogram(stft_result):
    return torch.abs(stft_result) ** 2


def dct(x, num_ceps):
    N = x.shape[-1]

    result = torch.zeros(x.shape[0], num_ceps, device=DEVICE)

    for k in range(num_ceps):
        basis = torch.cos(
            math.pi * k * (torch.arange(N, device=DEVICE) + 0.5) / N
        )
        result[:, k] = torch.matmul(x, basis)

    return result


def compute_mfcc(
    signal,
    sr=16000,
    n_fft=400,
    hop_length=160,
    n_mels=40,
    n_mfcc=13
):
    signal = signal.to(DEVICE)

    spec = stft(signal, n_fft, hop_length)
    power_spec = power_spectrogram(spec)

    mel_filter = build_mel_filter(sr, n_fft, n_mels)
    mel_spec = torch.matmul(power_spec, mel_filter.T)
    log_mel = torch.log(mel_spec + 1e-8)
    mfcc = dct(log_mel, n_mfcc)

    return mfcc


if __name__ == "__main__":
    sr = 16000
    duration = 1.0

    t = torch.linspace(0, duration, int(sr * duration))
    signal = torch.sin(2 * math.pi * 440 * t)

    signal = signal.to(DEVICE)

    mfcc = compute_mfcc(signal)

    print(f"MFCC shape: {mfcc.shape}")
    print("MFCC sample:\n", mfcc[:5])

