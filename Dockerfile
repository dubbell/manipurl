FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive


# Install Basic Dependencies
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:beineri/opt-qt-5.15.2-focal

RUN apt-get update && apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
    
RUN apt-get update && apt-get install -y \
    python3.9 python3-pip git python3.9-dev python3.9-distutils python3.9-venv

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

RUN set -x \ rm -f /usr/bin/python3 /usr/bin/pip3 /usr/bin/pip && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3.9 /usr/bin/pip3 && \
    ln -sf /usr/bin/pip3.9 /usr/bin/pip

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    xvfb \
    wget unrar tar && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y qt515base qt515tools qt515charts-no-lgpl

# Install PyTorch (Match CUDA version)
RUN pip3 install torch torchvision ipykernel gymnasium

# Setup CoppeliaSim (User must provide the tar.xz)
COPY CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz /opt/
RUN tar -xvf /opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C /opt/ && \
    rm /opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# Environment Variables for PyRep
ENV COPPELIASIM_ROOT=/opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# Install PyRep and RLBench
RUN git clone https://github.com/stepjam/PyRep.git && \
    pip3 install -r PyRep/requirements.txt && \
    pip3 install ./PyRep
RUN pip3 install git+https://github.com/stepjam/RLBench.git#egg=rlbench[gym,dev]