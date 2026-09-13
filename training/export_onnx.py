"""
Export the trained spiking network to the contract the Android app expects:

    input   pcm      float32 [1, 48000]   normalized to [-1, 1] @ 16 kHz
    output  logits   float32 [1, 8]       spike counts per class
    output  spikes   int64   [1]          total hidden spikes for the run

The time loop unrolls at export because the window is a fixed 3 seconds, so
the graph is static and needs no ONNX Loop op.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .frontend import NUM_SAMPLES
from .models import SpikingSER


class ExportWrapper(nn.Module):
    """Adds the int64 spike counter the app reads for its efficiency readout."""

    def __init__(self, model: SpikingSER) -> None:
        super().__init__()
        model.count_spikes = True
        self.model = model

    def forward(self, pcm: torch.Tensor):
        logits, spikes = self.model(pcm)
        return logits, spikes.reshape(1).to(torch.int64)


def export(checkpoint: str | None, out_path: str, encoding: str = "delta") -> Path:
    model = SpikingSER(encoding=encoding)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    wrapper = ExportWrapper(model).eval()
    dummy = torch.zeros(1, NUM_SAMPLES)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out),
        input_names=["pcm"],
        output_names=["logits", "spikes"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="build/snn_ser.onnx")
    ap.add_argument("--encoding", default="delta", choices=["direct", "delta", "rate"])
    args = ap.parse_args()

    path = export(args.checkpoint, args.out, args.encoding)
    size_mb = path.stat().st_size / 1e6
    print(f"wrote {path} ({size_mb:.2f} MB)")
    print("copy it to app/src/main/assets/snn_ser.onnx")
