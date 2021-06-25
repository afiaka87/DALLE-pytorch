#!/bin/bash 
## afiaka87 Full install for RTX 2070 Super on PopOS 20.10, CUDA Toolkit 11.2, Pytorch 1.9, Python 3.9
# Run at your own risk if your own a debian setup

# cuda 10.2
sudo update-alternatives --config cuda 
nvcc -V # should be CUDA 10.2

# Clone and install
git clone 'https://github.com/afiaka87/dalle-pytorch@working_directory'
cd dalle-pytorch && python3 -m venv DALLE-pytorch_venv
source DALLE-pytorch_venv/bin/activate; 
pip3 install wheel Cython wandb gpustat ipython jupyter
pip3 install torch torchvision torchaudio # requires pytorch 1.9.0, CUDA 11.2

# install apex
sudo rm -rf /tmp/apex
git clone https://github.com/NVIDIA/apex.git /tmp/apex
cd /tmp/apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# install deep speed
sudo rm -rf /tmp/DeepSpeed
git clone https://github.com/microsoft/DeepSpeed.git /tmp/DeepSpeed
cd /tmp/DeepSpeed && DS_BUILD_OPS=1 DS_BUILD_SPARSE_ATTN=1 ./install.sh -s

# install CompVis taming-transformers
pip install git+https://github.com/CompVis/taming-transformers.git