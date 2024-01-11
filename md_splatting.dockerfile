FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    git python3-dev python3-pip \ 
    libxrender1 \
    libxxf86vm-dev \
    libxfixes-dev \
    libxi-dev \
    libxkbcommon-dev \
    libsm-dev \
    libgl-dev

RUN pip install torch \
torchvision \
torchaudio \
mmcv \
matplotlib \
argparse \
lpips \
plyfile \
imageio-ffmpeg \ 
h5py \
imageio \
natsort \ 
numpy \ 
wandb \
tqdm 

