# Yuaz SGR

**Yuaz SGR (Singing Generative Refinement)** — A singing voice synthesis system based on DDSP + Flow Matching.

## Overview

This project reproduces a cutting-edge singing voice synthesis system using DDSP (Differentiable Digital Signal Processing) and Flow Matching technologies, achieving high-fidelity Chinese singing voice synthesis.

**The world's first complete reproduction and open-source implementation using a Chinese dataset.**

### Paper

> Title: *A Vector Quantized Approach for Text to Speech Synthesis on Real-World Spontaneous Speech*
> See `论文/paper15_full.txt` for the full text.

## Project Structure

```
yuaz-sgr/
├── yuaz_sgr/                           # Core training code
│   ├── models/                         # Model definitions (Encoder, RVQ, DDSP, Flow Matching)
│   ├── train/                          # Training scripts
│   ├── data/                           # Data processing & OpenCPOP dataset
│   ├── scripts/                        # Preprocessing / inference scripts
│   ├── losses/                         # Loss functions
│   └── utils/                          # Utility functions
├── yuaz_sgr_inference/                 # Lightweight inference package
│   ├── yuaz_sgr/                       # Model code
│   ├── checkpoint_300k.pt              # Pretrained model (300K steps, 94MB)
│   ├── infer_vsqx.py                   # VSQX inference script
│   └── example.py                      # Inference example
├── yuaz_sgr_finetune_inference/        # Fine-tuning inference package
│   ├── yuaz_sgr/                       # Model code
│   ├── freeze_final.pt                 # Fine-tuned model (94MB)
│   ├── train_freeze.py                 # Fine-tuning script
│   └── example.py                      # Inference example
├── 示例.wav                            # VSQX inference demo output
├── 论文/                               # Reference paper
├── README.md                           # Chinese docs
└── README.en.md                        # English docs
```

## Model Architecture

- **Encoder**: MFCC feature extraction + CREPE F0 estimation
- **RVQ**: Residual Vector Quantization (4-level quantization)
- **DDSP Decoder**: Harmonic + noise synthesis with weight_net gating
- **Conditioning Encoder**: Phoneme embedding + F0 → conditioning vector μ
- **Flow Matching Generator**: ODE integration for latent variable generation
- **F0Predictor**: Independent F0 refinement from μ

## Audio Demo

<audio controls src="示例.wav">
  Your browser does not support the audio tag. Download [示例.wav](示例.wav) to listen.
</audio>

> Song: 这世界那么多人 (VSQX tuned), synthesized with original F0, n_steps=8

## Inference Performance

Measured on **CPU (AMD Ryzen 5 3500U)**:

| Metric | Value |
|--------|-------|
| Model size | 94MB (purified, weights only) |
| Inference speed (n_steps=8) | RTF = 0.821 (255s song → 209s synthesis, faster than real-time) |
| CPU memory usage | ~283 MB (load+inference delta) |
| ODE steps | Supports 1~16 steps, 8 steps for high quality |
| Sample rate | 24 kHz |

> RTF (Real-Time Factor) < 1 means synthesis is faster than the audio duration, achieving real-time or better performance even on CPU.

## Pretrained Models

This repository includes two pretrained models:

| Model | File | Size | Description |
|-------|------|------|-------------|
| Base Model | `yuaz_sgr_inference/checkpoint_300k.pt` | 94MB | 300K steps, purified (weights only) |
| Fine-tuned Model | `yuaz_sgr_finetune_inference/freeze_final.pt` | 94MB | Fine-tuned with frozen decoder for better pronunciation |

## Inference

```bash
# Inference with base model (phoneme-level synthesis)
cd yuaz_sgr_inference/
pip install -r requirements.txt
python example.py

# Inference with fine-tuned model
cd yuaz_sgr_finetune_inference/
pip install -r requirements.txt
python example.py
```

## Dataset

This project uses the [OpenCPOP](https://github.com/yxlllc/OpenCPOP) Chinese singing dataset for training.

## License

This project is licensed under the MIT License — free to use and modify.

## Acknowledgements

- [OpenCPOP](https://github.com/yxlllc/OpenCPOP) Chinese singing dataset
- DDSP: Differentiable Digital Signal Processing
- Flow Matching for Generative Modeling
