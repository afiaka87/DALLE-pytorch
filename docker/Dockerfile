ARG IMG_TAG=1.7.0-cuda10.2-cudnn7-devel
ARG IMG_REPO=pytorch
ARG BRANCH=main

FROM pytorch/$IMG_REPO:$IMG_TAG

RUN apt-get -y update && apt-get -y install git gcc llvm-9-dev cmake libaio-dev vim
RUN git clone https://github.com/microsoft/DeepSpeed.git /tmp/DeepSpeed
RUN cd /tmp/DeepSpeed && DS_BUILD_SPARSE_ATTN=1 ./install.sh -r
RUN cd ~ && git clone https://github.com/afiaka87/dalle-pytorch --single-branch 'working_directory'
RUN cd dalle-pytorch
RUN python3 setup.py install

WORKDIR 
