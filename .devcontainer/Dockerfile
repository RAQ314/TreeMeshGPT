FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
#FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ARG CONDA_ENV_NAME
#ENV CONDA_ENV_NAME_Value "$CONDA_ENV_NAME"

RUN echo $CONDA_ENV_NAME

ENV DEBIAN_FRONTEND=noninteractive


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    wget \
    libx11-6 libgl1

# Create Volume Dir
RUN mkdir /volume
WORKDIR /volume

#--- Conda
#Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && /bin/bash ~/miniforge.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Create tmgpt conda env
RUN conda update conda
RUN conda create -n $CONDA_ENV_NAME python==3.11 -y -v

#Init conda for bash
RUN conda init bash \
    && echo "conda activate $CONDA_ENV_NAME" >> ~/.bashrc

#Override shell to use conda environement
RUN echo "conda run -n $CONDA_ENV_NAME /bin/bash -c \"\$@\"" > /tmp/conda_shell.sh
SHELL ["sh", "/tmp/conda_shell.sh"]

#Debug
# RUN conda env list >> /tmp/debug_conda.txt
# RUN pip install numpy --no-cache-dir

#--- Install dependancies
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
#RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt --no-cache-dir   \
    && conda clean -a

#--- Patches to enable torch to use P6000 (arch 6.1)
#See : https://github.com/vllm-project/vllm/issues/963
#See : https://www.reddit.com/r/LocalLLaMA/comments/1lrerwe/pytorch_27x_no_longer_supports_pascal_architecture/ and https://github.com/pytorch/pytorch/issues/157517
#See : https://github.com/sasha0552/pascal-pkgs-ci
# RUN sed -i 's+>= 7+>= 6+' /opt/conda/envs/$CONDA_ENV_NAME/lib/python3.11/site-packages/torch/utils/_triton.py
# RUN sed -i 's+>= 7+>= 6+' /opt/conda/envs/$CONDA_ENV_NAME/lib/python3.11/site-packages/torch/_inductor/scheduler.py

WORKDIR /volume

