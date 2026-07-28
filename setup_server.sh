#!/bin/bash
export PATH=~/miniconda3/bin:$PATH

# Create conda environment
conda create -n yuaz_sgr python=3.10 -y

# Activate and install packages
source ~/miniconda3/bin/activate yuaz_sgr

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy librosa pyworld soundfile pyyaml textgrid pretty_midi torchcrepe onnxruntime tqdm tensorboard matplotlib

# Test
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
