# Neuromorphic Emotion Recognition from Speech

On-device speech emotion recognition using a spiking neural network. Android app
in Kotlin/Compose, SNN trained in PyTorch + snnTorch, inference via ONNX Runtime.

Nothing leaves the phone: audio is captured, classified and discarded in memory.

## Why an SNN

A spiking network only does work when a neuron fires. On sparse, event-driven
input like speech that means far fewer synaptic operations than a dense
CNN-LSTM doing the same job. The point of this project is to measure that gap
rather than assert it, so the app displays the live spike count next to the
prediction and the training script reports SynOps against baseline MACs.

## Architecture

```
mic 16 kHz mono, 3 s  ->  48000 float samples in [-1, 1]
                          |
                          |  (inside the ONNX graph)
                          v
                     log-mel, 40 ch x 92 frames
                          |
                          v
                     spike encoding (delta / direct / rate)
                          |
                          v
          Linear 40->256 -> LIF -> Linear 256->256 -> LIF
                          |
                          v
              Linear 256->8 -> leaky integrator readout
                          |
                          v
                  logits [8] + total spike count
```

The mel frontend is part of the exported model. That is deliberate: Kotlin does
no DSP, so training features and inference features cannot drift apart. It is
built from plain matmuls rather than `torch.stft`, because STFT support in ONNX
runtimes is uneven.

### Model contract

| | name | type | shape |
|---|---|---|---|
| input | `pcm` | float32 | `[1, 48000]` |
| output | `logits` | float32 | `[1, 8]` |
| output | `spikes` | int64 | `[1]` |

Labels in order: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

## Results

| encoding | accuracy | spikes / clip | SynOps | vs baseline |
| --- | --- | --- | --- | --- |
| direct | _tbd_ | ~6,800 | _tbd_ | |
| delta | _tbd_ | ~3,800 | _tbd_ | |
| rate | _tbd_ | ~11,900 | _tbd_ | |
| CNN-LSTM baseline | _tbd_ | n/a | 9.09M MACs | 1x |

Parameter counts: SNN 78,347 · CNN-LSTM baseline 320,584.
Spike figures are from untrained nets on synthetic audio; fill the rest in after
training on RAVDESS.

## Build

CI builds the APK on every push to `main` and publishes it as a GitHub Release,
so the download link is shareable. The Actions tab has the artifact too.

Locally:

```bash
gradle assembleRelease
```

Without a keystore at `keystore/release.jks` the release build is debug-signed —
installable for testing, not for distribution.

## Training

```bash
pip install torch snntorch librosa numpy
python -m training.train --data /path/to/RAVDESS --encoding delta --epochs 40
python -m training.export_onnx --checkpoint build/best.pt --out build/snn_ser.onnx
cp build/snn_ser.onnx app/src/main/assets/
```

Until `snn_ser.onnx` is in `app/src/main/assets/`, the app runs on
`StubEmotionClassifier` and the UI header says so. Dropping the file in is the
only change needed — `EmotionViewModel` picks the real path automatically.

## Verified

- Matmul mel frontend matches `librosa.stft` to 1e-6
- ONNX export of the 92-step unrolled spiking loop succeeds (5.2 MB)
- ONNX Runtime output matches PyTorch exactly, spike counts included
- Network overfits a synthetic batch to 100% in 20 steps
