ARG IMG_TAG_TORCH=1.8.1
ARG IMG_TAG_CUDA=cuda10.2
ARG IMG_TAG_CUDNN=cudnn7-devel
ARG IMG_TAG=$IMG_TAG_TORCH-$IMG_TAG_CUDA-$IMG_TAG_CUDNN
ARG IMG_REPO=pytorch

FROM pytorch/$IMG_REPO:$IMG_TAG

# Ubuntu dependencies.
RUN apt-get -y update && apt-get -y install git gcc llvm-9-dev cmake libaio-dev vim cargo python3-pip python3-venv sudo vim tree wget tldr aria2 pigz zsh

# Rust.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Tiny ~4200 image-text pair dataset.
RUN mkdir -p XKCD
RUN wget https://www.dropbox.com/s/5lby10gezot4obn/XKCD_Dataset.tar.gz && tar xf XKCD_Dataset.tar.gz --directory=XKCD 

# dalle-pytorch
RUN git clone https://github.com/afiaka87/dalle-pytorch.git /tmp/dalle-pytorch
RUN cd /tmp/dalle-pytroch && python setup.py install
# W&B
RUN python -m pip install wandb gpustat 

# Starship.rs
RUN sh -c "$(curl -fsSL https://starship.rs/install.sh)" 

# DeepSpeed
RUN git clone https://github.com/microsoft/DeepSpeed.git /tmp/DeepSpeed
RUN cd /tmp/DeepSpeed && DS_BUILD_OPS=1 ./install.sh -r

# CompVis taming-transformers
RUN git clone https://github.com/CompVis/taming-transformers.git /tmp/taming
RUN cd /tmp/taming && pip install . --editable

WORKDIR dalle