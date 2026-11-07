# Yuaz SGR 1.2

Yuaz SGR (Singing Generative Refinement) — Chinese singing voice synthesis built on DDSP + an **iterative residual refiner (IRR)**.

## Overview

The vocoder backbone is DDSP (Differentiable Digital Signal Processing). Latent generation is handled by a **supervisable iterative residual refiner** instead of flow matching, which gives higher fidelity Chinese singing synthesis with a much smaller network.

The original paper is kept as a reference baseline; from this version on the generator **no longer uses** the paper's flow matching. See `paper_diff.md`.

### Reference paper

> *Singing voice synthesis via latent flow matching and differentiable digital signal processing*
> Full text in `论文/paper15_full.txt`

## What's new in 1.2

### Bundled checkpoint

This release ships `checkpoint_75k_pure.pt`, trained **from scratch for 75k steps**.
75k is the sweet spot: training further makes the `rvq` codebook commitment drift and the
aperiodicity rise, which gradually turns plosives into noise (see `TRAINING.md` §6.3).

### Generator: flow matching → IRR

The paper and older versions train a velocity field with CFM and integrate an ODE at inference.
This version replaces that with:

```
z₀ = μ
z_{k+1} = z_k + g(z_k, μ, k)      k = 0..K-1,  K = 4 (fixed during training)
```

- `g`: 3 × 1D conv + step embedding, ≈1.34M parameters
- no noise sampling, no ODE, no CFM loss; `z` receives decoder gradients directly
- inference costs K small convolutions; K is adjustable (default 4, 8 for more detail)

IRR aligns `z` with the decoder and the target audio end to end. Measured:

| Metric | 224k Flow | 264k IRR (fine-tuned) |
| --- | --- | --- |
| Spectral flatness (SFM) | 0.0268 / 0.0463 | **0.0253 / 0.0410** |
| HF energy ratio (>4k) | 0.0702 / 0.1217 | 0.0686 / 0.1095 |

Subjectively: clearer articulation, more natural overall.

### Other changes

| # | Problem | Fix | Where |
| --- | --- | --- | --- |
| 1 | ~28% of F0 labels saturated at 2006 Hz | peak normalization + confidence gating, `fmax` lowered to 1100 Hz | `scripts/preprocess_opencpop_v2.py` |
| 2 | Fricatives/affricates lacked HF detail | log-mel representation in the encoder | `models/encoder.py` |
| 3 | Decoder had no phoneme conditioning | added `decoder_cond` | `models/ddsp_decoder.py` |
| 4 | F0 had no portamento/vibrato structure | F0Predictor v2 + inter-frame smoothing loss | `models/conditioning.py` |
| 5 | Joint-stage synthesis loss overflowed to NaN | `z_pred` clamping + `isfinite` guards on every loss | `train/trainer.py` |
| 6 | Envelope resolution too low | envelope mel 128 → 256 | `models/ddsp_decoder.py` |
| 7 | Jittery generations | generator weight EMA | `train/trainer.py` |
| 8 | Training signal precision | STFT loss in fp32 + mel perceptual loss | `train/trainer.py` |
| 9 | `lag` / `note_lengths` labels misaligned | merge & dedupe by (note, note_dur) | `scripts/preprocess_opencpop_v2.py` |
| 10 | Broadband "current" noise in the harmonic branch | Nyquist soft rolloff, learnable harmonic weights, time-domain harmonic synthesis switches | `models/ddsp_decoder.py` |
| 11 | Noise glued in front of plosives | phoneme-duration statistics table + fixed 32 ms plosives | `yuaz_sgr/synthesizer.py` |

### Inference & engineering

| Feature | Notes |
| --- | --- |
| Harmonic anti-aliasing | per-harmonic accumulation + Nyquist masking; memory O(K·N) → O(N) |
| Silence gate | no harmonic excitation where `f0=0` |
| Chunked decoding | `max_chunk_frames` chunking + cross-fade |
| Voicing prior | `voicing_prior=True` (off by default) |
| Soft quantization | `rvq_residual_alpha` adds back the continuous RVQ residual |
| UST support | `infer_ust.py` reads Utau/OpenUtau projects directly |
| Checkpoint capability autodetect | reads `flags` (`generator` / `irr_steps` / `decoder_cond` / `f0_predictor` / `encoder_feature` / `dec_mels`) |
| Resumable training | `--resume auto`, `--max_hours`, `--keep_ckpts` |
| Training dashboard | none |

## Compatibility

**The bundled `checkpoint_75k_pure.pt` (trained from scratch) uses the IRR architecture and is NOT compatible with any Flow Matching checkpoint** (old weights have no `refiner.*`, and the generator code is gone).

Inference reads `generator` / `irr_steps` from the checkpoint `flags`.

## Repository layout

```
.
├── yuaz_sgr/                     # core training code
│   ├── models/                   # Encoder / RVQ / DDSP Decoder / IterativeRefiner / Conditioning / Discriminator
│   ├── train/                    # train.py / trainer.py / progress.py
│   ├── data/ scripts/ losses/ utils/
│   ├── config/                   # base.yaml / model.yaml / train.yaml
│   └── inference/                # in-package inference entry (YuazSinger)
├── yuaz_sgr_inference/           # inference package
│   ├── yuaz_sgr/                 # model code + synthesizer.py + assets/
│   ├── checkpoint_75k_pure.pt    # bundled model (75k, 34.3 MB)
│   ├── 示例_75k_默认.wav          # default-parameter demo
│   ├── infer_ust.py              # UST inference
│   ├── infer_vsqx.py             # VSQX inference
│   └── example.py                # usage example
├── 论文/                         # original paper
├── 示例.wav                      # audio demo
├── paper_diff.md                 # differences vs the paper
└── README.md
```

## Architecture

- **Encoder**: 160-dim log-mel
- **RVQ**: residual vector quantization (8 × 1024 × 128)
- **Conditioning Encoder**: phonemes + durations + F0 → μ
- **Iterative Refiner**: `z₀=μ`, K residual steps (1.34M)
- **DDSP Decoder**: harmonics + noise, weight_net gating, Nyquist soft anti-aliasing, silence gate, 256-mel envelope
- **F0Predictor v2**: predicts frame-level F0 from μ + step-wise F0 (not converged yet in this model — use `use_predicted_f0=False`)

## Audio demo

> "这世界那么多人" — synthesized by the bundled model (`checkpoint_75k_pure.pt`), IRR K=4, statistics-table durations with 32 ms plosives.

File: `示例.wav` (~4 min 15 s).

## Inference speed

| Platform | Config | RTF |
|---|---|---|
| CPU (4 cores) | K=4 | 0.1–0.2 |

Each IRR step is a small convolution, so K from 1 to 8 barely changes latency.

## Pretrained model

| Model | File | Size | Notes |
|---|---|---|---|
| Base | `yuaz_sgr_inference/checkpoint_75k_pure.pt` | 34.3 MB | 75k steps (IRR, K=4) |

## Inference

```bash
cd yuaz_sgr_inference/
pip install -r requirements.txt
python example.py              # basic example
python infer_ust.py song.ust   # synthesize from a UST project
```

```python
from yuaz_sgr import YuazSinger
singer = YuazSinger("checkpoint_75k_pure.pt", device="cuda")
wav = singer.sing(
    phoneme_ids, durations, f0_curve,
    irr_steps=4,             # refinement steps: 4 default, 8 for more detail
    use_predicted_f0=False,  # keep False: the F0 predictor is not converged
    consonant_gain=0.5,      # consonant gain
    f0_portamento_frames=4,  # portamento length in frames
    # consonant durations default to assets/cons_dur_table.json (plosives fixed at 32 ms)
)
```

## Training

Configuration lives in `yuaz_sgr/config/`:

```bash
python train/train.py --config_dir config --data_dir <preprocessed> \
    --phoneme_map <preprocessed>/phoneme_map.json \
    --output_dir checkpoints --resume auto
```

Key knobs: `generator.irr_steps` (refinement steps, default 4) and `loss_weights.irr_steps`.

## Dataset

[OpenCPOP](https://wenet-e2e.github.io/opencpop/) Chinese singing dataset (24 kHz, 3450/110/206 split).

## License

MIT License.

## Acknowledgements

- [OpenCPOP](https://wenet-e2e.github.io/opencpop/)
- [DDSP](https://github.com/magenta/ddsp)
- Original paper *Singing voice synthesis via latent flow matching and DDSP* (baseline reference)
