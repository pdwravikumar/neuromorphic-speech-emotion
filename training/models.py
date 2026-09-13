"""
The spiking classifier and the conventional baseline it has to be measured
against.

The mel frontend lives *inside* both models. That is deliberate: the exported
ONNX graph then takes raw PCM, so the Android app does no signal processing
and train/serve feature skew is structurally impossible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from .frontend import MelFrontend, N_MELS, NUM_SAMPLES, num_frames

NUM_CLASSES = 8
EMOTIONS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised",
]


def encode(mel: torch.Tensor, mode: str, threshold: float = 0.25) -> torch.Tensor:
    """
    Turn a log-mel spectrogram [B, T, C] into layer-1 input [B, T, C].

    The mel time axis *is* the SNN time axis. This is the part most SNN audio
    projects get wrong: they collapse the clip into one static feature vector
    and then rate-encode it over synthetic timesteps, which throws away the
    temporal structure that justified using an SNN at all.

    modes:
      direct - inject the mel value as analog input current. Not spikes at the
               input, but standard practice and usually the accuracy ceiling.
      delta  - spike when a channel changes by more than `threshold` between
               frames. Sparse, event-driven, closest to a silicon cochlea.
               This is the configuration that earns the neuromorphic claim.
      rate   - Bernoulli sampling with probability proportional to amplitude.
               Simple, noisy, generally the weakest of the three.
    """
    if mode == "direct":
        return mel

    if mode == "delta":
        prev = torch.cat([mel[:, :1], mel[:, :-1]], dim=1)
        diff = mel - prev
        # Signed spikes: +1 on rise, -1 on fall, 0 otherwise. Straight-through
        # so the frontend stays trainable through the quantization.
        hard = (diff > threshold).float() - (diff < -threshold).float()
        return hard.detach() + diff - diff.detach()

    if mode == "rate":
        p = torch.sigmoid(mel).clamp(0.0, 1.0)
        hard = torch.bernoulli(p)
        return hard.detach() + p - p.detach()

    raise ValueError(f"unknown encoding mode: {mode}")


class SpikingSER(nn.Module):
    """Leaky integrate-and-fire network over the mel time axis."""

    def __init__(
        self,
        hidden: int = 256,
        beta: float = 0.9,
        encoding: str = "delta",
        slope: float = 25.0,
        count_spikes: bool = False,
    ) -> None:
        super().__init__()
        self.encoding = encoding
        self.count_spikes = count_spikes
        self.hidden = hidden
        self.frontend = MelFrontend()

        grad = surrogate.fast_sigmoid(slope=slope)

        self.fc1 = nn.Linear(N_MELS, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=grad, learn_beta=True)
        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=grad, learn_beta=True)
        self.fc3 = nn.Linear(hidden, NUM_CLASSES)
        # The readout is a NON-spiking leaky integrator. If the output layer
        # fires, an untrained delta-encoded net emits zero output spikes, the
        # logits are all zero and the loss has no gradient -- the network is
        # dead before training starts. Accumulating membrane potential instead
        # keeps the readout continuous and differentiable from step one.
        self.readout_beta = nn.Parameter(torch.tensor(beta))

    def forward(self, pcm: torch.Tensor):
        mel = self.frontend(pcm)
        x = encode(mel, self.encoding)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = torch.zeros(x.shape[0], NUM_CLASSES, device=x.device, dtype=x.dtype)

        beta3 = torch.clamp(self.readout_beta, 0.0, 1.0)
        out_sum = 0.0
        hidden_spikes = 0.0

        for t in range(x.shape[1]):
            s1, mem1 = self.lif1(self.fc1(x[:, t]), mem1)
            s2, mem2 = self.lif2(self.fc2(s1), mem2)
            mem3 = beta3 * mem3 + self.fc3(s2)
            out_sum = out_sum + mem3
            hidden_spikes = hidden_spikes + s1.sum() + s2.sum()

        out_sum = out_sum / x.shape[1]

        # Spike count per class is the logit. ce_count_loss expects this shape
        # of evidence and it survives ONNX export as a plain sum.
        if self.count_spikes:
            return out_sum, hidden_spikes
        return out_sum

    @torch.no_grad()
    def synops(self, pcm: torch.Tensor) -> float:
        """
        Synaptic operations for one clip: each hidden spike triggers one
        accumulate per outgoing synapse. This is the number to compare against
        the baseline's MAC count -- an accumulate costs roughly 1/5 the energy
        of a multiply-accumulate in 45nm silicon (Horowitz, ISSCC 2014).
        """
        was = self.count_spikes
        self.count_spikes = True
        _, spikes = self.forward(pcm)
        self.count_spikes = was
        return float(spikes) * self.hidden


class CnnLstmSER(nn.Module):
    """Conventional baseline: 1D conv stack over mel, then a BiLSTM."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.frontend = MelFrontend()
        self.conv = nn.Sequential(
            nn.Conv1d(N_MELS, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(128, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, NUM_CLASSES)

    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        mel = self.frontend(pcm)                 # [B, T, C]
        h = self.conv(mel.transpose(1, 2))       # [B, 128, T/4]
        h, _ = self.lstm(h.transpose(1, 2))
        return self.head(h.mean(dim=1))

    def macs(self) -> int:
        """Multiply-accumulates per clip, dense by construction."""
        t = num_frames()
        conv1 = N_MELS * 64 * 5 * t
        conv2 = 64 * 128 * 5 * (t // 2)
        lstm = 4 * 2 * (128 + 128) * 128 * (t // 4)
        head = 256 * NUM_CLASSES
        return conv1 + conv2 + lstm + head


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
