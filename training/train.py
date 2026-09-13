"""
Train the spiking network (or the CNN-LSTM baseline) on RAVDESS.

    python -m training.train --data /path/to/RAVDESS --encoding delta --epochs 40
    python -m training.train --data /path/to/RAVDESS --baseline --epochs 40

The run ends by printing the energy comparison, which is the number the whole
project exists to produce.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import RavdessDataset, class_weights, scan, speaker_split
from .models import CnnLstmSER, EMOTIONS, SpikingSER, count_params

# Horowitz, ISSCC 2014: 45nm, 32-bit. An accumulate is ~0.9 pJ, a
# multiply-accumulate ~4.6 pJ. The ~5x ratio is what makes spikes cheap.
ENERGY_AC_PJ = 0.9
ENERGY_MAC_PJ = 4.6


def evaluate(model, loader, device, spiking: bool) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    spikes_total = 0.0

    was = getattr(model, "count_spikes", False)
    if spiking:
        model.count_spikes = True

    with torch.no_grad():
        for pcm, y in loader:
            pcm, y = pcm.to(device), y.to(device)
            if spiking:
                logits, spikes = model(pcm)
                spikes_total += float(spikes)
            else:
                logits = model(pcm)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()

    if spiking:
        model.count_spikes = was
    return correct / max(total, 1), spikes_total / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="RAVDESS root directory")
    ap.add_argument("--encoding", default="delta", choices=["direct", "delta", "rate"])
    ap.add_argument("--baseline", action="store_true", help="train the CNN-LSTM instead")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--val-actors", type=int, default=4)
    ap.add_argument("--out", default="build")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    clips = scan(args.data)
    train_clips, val_clips = speaker_split(clips, args.val_actors, args.seed)

    train_loader = DataLoader(
        RavdessDataset(train_clips, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True,
    )
    val_loader = DataLoader(
        RavdessDataset(val_clips), batch_size=args.batch_size, num_workers=2
    )

    spiking = not args.baseline
    model = (
        SpikingSER(hidden=args.hidden, encoding=args.encoding)
        if spiking else CnnLstmSER()
    ).to(device)
    tag = f"snn-{args.encoding}" if spiking else "baseline"
    print(f"{tag}: {count_params(model):,} trainable parameters")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_clips).to(device), label_smoothing=0.1
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(train_loader), 1)
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"best-{tag}.pt"
    best = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        running = 0.0

        for pcm, y in train_loader:
            pcm, y = pcm.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(pcm), y)
            loss.backward()
            # Surrogate gradients through 92 timesteps can spike hard.
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            running += loss.item()

        acc, spikes = evaluate(model, val_loader, device, spiking)
        flag = ""
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
            flag = " *"

        msg = (
            f"epoch {epoch:3d}  loss {running / max(len(train_loader), 1):.4f}  "
            f"val {acc * 100:5.2f}%  {time.time() - started:5.1f}s"
        )
        if spiking:
            msg += f"  spikes/clip {spikes:8.0f}"
        print(msg + flag)

    print(f"\nbest speaker-independent accuracy: {best * 100:.2f}%  -> {ckpt}")

    if spiking:
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        _, spikes = evaluate(model, val_loader, device, spiking=True)
        synops = spikes * args.hidden
        baseline_macs = CnnLstmSER().macs()
        snn_pj = synops * ENERGY_AC_PJ
        base_pj = baseline_macs * ENERGY_MAC_PJ

        print("\nenergy comparison (per clip, 45nm estimates)")
        print(f"  SNN     {synops:14,.0f} SynOps  {snn_pj / 1e6:8.2f} uJ")
        print(f"  CNN-LSTM{baseline_macs:14,} MACs    {base_pj / 1e6:8.2f} uJ")
        print(f"  ratio   {base_pj / max(snn_pj, 1e-9):.1f}x cheaper")
        print("\nThese are analytical estimates, not measurements. Say so when")
        print("you quote them -- real silicon depends on memory traffic too.")

    print("\nper-class check on held-out speakers:")
    model.eval()
    conf = torch.zeros(8, 8, dtype=torch.long)
    with torch.no_grad():
        for pcm, y in val_loader:
            logits = model(pcm.to(device))
            if isinstance(logits, tuple):
                logits = logits[0]
            for t, p in zip(y.tolist(), logits.argmax(1).cpu().tolist()):
                conf[t, p] += 1
    for i, name in enumerate(EMOTIONS):
        n = conf[i].sum().item()
        rate = conf[i, i].item() / n * 100 if n else 0.0
        print(f"  {name:10s} {rate:5.1f}%  ({conf[i, i].item()}/{n})")


if __name__ == "__main__":
    main()
