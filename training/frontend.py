"""
Log-mel frontend built entirely from matmuls.

Why not torch.stft? Because this module has to survive torch.onnx.export and
then run inside ONNX Runtime on Android. aten::stft maps to an ONNX STFT op
whose support is patchy across runtimes. Framing with unfold and multiplying
by a precomputed DFT basis produces nothing but MatMul and Mul nodes, which
every runtime handles.

The mel filterbank comes from librosa itself, so the features here are the
same ones a conventional librosa pipeline would produce.
"""

from __future__ import annotations

import librosa
import numpy as np
import torch
import torch.nn as nn

SAMPLE_RATE = 16_000
DURATION_SEC = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)  # 48000
N_FFT = 1024
HOP = 512
N_MELS = 40


def num_frames(num_samples: int = NUM_SAMPLES, n_fft: int = N_FFT, hop: int = HOP) -> int:
    """Frame count for non-centered framing."""
    return 1 + (num_samples - n_fft) // hop


class MelFrontend(nn.Module):
    """Raw PCM [B, NUM_SAMPLES] -> log-mel [B, T, N_MELS], per-utterance standardized."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop: int = HOP,
        n_mels: int = N_MELS,
        fmin: float = 20.0,
        fmax: float | None = 8000.0,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.n_mels = n_mels

        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
        self.register_buffer("window", window)

        # Real DFT basis, keeping only the non-redundant bins.
        n_freq = n_fft // 2 + 1
        k = np.arange(n_freq)[:, None]
        t = np.arange(n_fft)[None, :]
        angle = 2.0 * np.pi * k * t / n_fft
        self.register_buffer("dft_real", torch.tensor(np.cos(angle).T, dtype=torch.float32))
        self.register_buffer("dft_imag", torch.tensor(-np.sin(angle).T, dtype=torch.float32))

        mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )  # [n_mels, n_freq]
        self.register_buffer("mel_fb", torch.tensor(mel_fb.T, dtype=torch.float32))

    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        if pcm.dim() == 1:
            pcm = pcm.unsqueeze(0)

        # [B, T, n_fft]
        frames = pcm.unfold(dimension=1, size=self.n_fft, step=self.hop)
        frames = frames * self.window

        real = frames @ self.dft_real          # [B, T, n_freq]
        imag = frames @ self.dft_imag
        power = real * real + imag * imag

        mel = power @ self.mel_fb              # [B, T, n_mels]
        log_mel = torch.log(mel + 1e-6)

        # Per-utterance standardization. Cheap, and it removes most of the
        # channel/recording-level variation between the training corpus and a
        # phone microphone.
        mean = log_mel.mean(dim=(1, 2), keepdim=True)
        std = log_mel.std(dim=(1, 2), keepdim=True) + 1e-5
        return (log_mel - mean) / std
