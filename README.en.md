# Yuaz SGR

**Yuaz SGR (Singing Generative Refinement)** — A singing voice synthesis system based on DDSP + Flow Matching.

## Overview

This project reproduces a cutting-edge singing voice synthesis system using DDSP (Differentiable Digital Signal Processing) and Flow Matching technologies, achieving high-fidelity Chinese singing voice synthesis.

**The world's first complete reproduction and open-source implementation of this paper using a Chinese dataset.** (See [论文/paper15.pdf](论文/paper15.pdf) for the original paper)

## Project Structure

```
yuaz-sgr/
├── yuaz_sgr/                    # Core training code
│   ├── models/                  # Model definitions (Encoder, RVQ, DDSP, Flow Matching)
│   ├── train/                   # Training scripts
│   ├── data/                    # Data processing
│   ├── scripts/                 # Preprocessing / inference scripts
│   ├── losses/                  # Loss functions
│   └── utils/                   # Utility functions
├── yuaz_sgr_inference/          # Lightweight inference package
│   └── checkpoint_300k.pt       # Final purified model (94MB)
├── yuaz_sgr_finetune_inference/ # Fine-tuning code
├── 训练/                         # Training configs and scripts
├── 论文/                         # Reference paper
└── vsqx/                        # VSQX test files
```

## Model Architecture

- **Encoder**: MFCC feature extraction + CREPE F0 estimation
- **RVQ**: Residual Vector Quantization (4-level quantization)
- **DDSP Decoder**: Harmonic + noise synthesis with weight_net gating
- **Conditioning Encoder**: Phoneme embedding + F0 → conditioning vector μ
- **Flow Matching Generator**: ODE integration for latent variable generation
- **F0Predictor**: Independent F0 refinement from μ

## Inference

```bash
cd yuaz_sgr_inference/
pip install -r requirements.txt
python example.py
```

## License

This project is licensed under the MIT License — free to use and modify.

## Acknowledgements

- [OpenCPOP](https://github.com/yxlllc/OpenCPOP) Chinese singing dataset
- DDSP: Differentiable Digital Signal Processing
- Flow Matching for Generative Modeling
