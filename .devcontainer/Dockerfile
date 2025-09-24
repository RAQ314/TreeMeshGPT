FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    wget

# Create Volume Dir
RUN mkdir /volume
WORKDIR /volume

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && /bin/bash ~/miniforge.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Create tmgpt conda env
RUN conda update conda
RUN conda create -n tmgpt python==3.11 -y -v
RUN conda init bash
RUN /bin/bash -c conda activate tmgpt

# Install tmgpt dependencies
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

WORKDIR /volume
