"""
RAVDESS loader.

Filenames encode everything:

    03-01-06-01-02-01-12.wav
     |  |  |  |  |  |  |
     |  |  |  |  |  |  actor 01-24 (odd = male, even = female)
     |  |  |  |  |  repetition
     |  |  |  |  statement
     |  |  |  intensity 01 normal, 02 strong
     |  |  emotion 01-08
     |  vocal channel 01 speech, 02 song
     modality

The split is by ACTOR, never random. A random split puts the same speaker in
train and test, and the network learns to identify the speaker rather than the
emotion -- which inflates accuracy by 20-30 points and collapses the moment a
stranger talks into the phone. Speaker-independent numbers are the only ones
worth putting on a CV.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from .frontend import NUM_SAMPLES, SAMPLE_RATE

# RAVDESS emotion codes are 1-based and in our label order already.
EMOTION_CODE_TO_INDEX = {i + 1: i for i in range(8)}


@dataclass(frozen=True)
class Clip:
    path: Path
    label: int
    actor: int


def scan(root: str | Path, speech_only: bool = True) -> list[Clip]:
    root = Path(root)
    clips: list[Clip] = []
    for wav in sorted(root.rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        try:
            channel, emotion, actor = int(parts[1]), int(parts[2]), int(parts[6])
        except ValueError:
            continue
        if speech_only and channel != 1:
            continue
        if emotion not in EMOTION_CODE_TO_INDEX:
            continue
        clips.append(Clip(wav, EMOTION_CODE_TO_INDEX[emotion], actor))
    if not clips:
        raise FileNotFoundError(f"no RAVDESS wav files under {root}")
    return clips


def speaker_split(
    clips: list[Clip], val_actors: int = 4, seed: int = 0
) -> tuple[list[Clip], list[Clip]]:
    actors = sorted({c.actor for c in clips})
    rng = random.Random(seed)
    held = set(rng.sample(actors, k=min(val_actors, len(actors) - 1)))
    train = [c for c in clips if c.actor not in held]
    val = [c for c in clips if c.actor in held]
    print(f"held-out actors: {sorted(held)}  train {len(train)} / val {len(val)}")
    return train, val


def load_pcm(path: Path) -> np.ndarray:
    """
    Load to exactly NUM_SAMPLES, peak-normalized.

    This must stay identical to AudioRecorder.kt on the phone: 16 kHz mono,
    3 seconds, peak normalization to [-1, 1]. Any drift here shows up as an
    accuracy cliff that is very hard to diagnose from the app side.
    """
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)

    # RAVDESS clips open with roughly half a second of silence.
    trimmed, _ = librosa.effects.trim(y, top_db=30)
    if trimmed.size > 0:
        y = trimmed

    if y.size < NUM_SAMPLES:
        y = np.pad(y, (0, NUM_SAMPLES - y.size))
    else:
        y = y[:NUM_SAMPLES]

    peak = np.abs(y).max()
    if peak > 1e-5:
        y = y / peak
    return y.astype(np.float32)


class RavdessDataset(Dataset):
    def __init__(self, clips: list[Clip], augment: bool = False, cache: bool = True) -> None:
        self.clips = clips
        self.augment = augment
        self._cache: dict[int, np.ndarray] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.clips)

    def _pcm(self, i: int) -> np.ndarray:
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        y = load_pcm(self.clips[i].path)
        if self._cache is not None:
            self._cache[i] = y
        return y

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        y = self._pcm(i).copy()

        if self.augment:
            # Shift in time so the model cannot key on onset position.
            shift = np.random.randint(-SAMPLE_RATE // 10, SAMPLE_RATE // 10)
            y = np.roll(y, shift)
            if shift > 0:
                y[:shift] = 0.0
            elif shift < 0:
                y[shift:] = 0.0

            y = y * np.random.uniform(0.7, 1.0)
            y = y + np.random.normal(0.0, 0.005, size=y.shape).astype(np.float32)
            y = np.clip(y, -1.0, 1.0)

        return torch.from_numpy(y.astype(np.float32)), self.clips[i].label


def class_weights(clips: list[Clip]) -> torch.Tensor:
    """RAVDESS has half as many neutral clips as the other seven emotions."""
    counts = np.bincount([c.label for c in clips], minlength=8).astype(np.float32)
    counts[counts == 0] = 1.0
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)
