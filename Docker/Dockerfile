ARG PARENT_IMAGE
FROM $PARENT_IMAGE
ARG PYTORCH_INSTALL="pip install --no-input torch==1.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu"
ARG FAISS_INSTALL="conda install -c pytorch faiss-gpu cudatoolkit=11.3"
ARG PYTHON_VERSION="3.7"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        xvfb \
        ffmpeg \
        xorg-dev \
        libsdl2-dev \
        swig \
        cmake \
        unzip \
        libglu1-mesa-dev \
        libgl1-mesa-dev \
        libosmesa6-dev \
        patchelf \
        gcc \
        openmpi-bin \
        openmpi-common \
        libopenmpi-dev \
        libgtk2.0-dev \
        libgl1-mesa-glx \
        libglfw3 \
        libglew-dev \
        git \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        libglib2.0-0 \
        wget \
        bash
RUN rm -rf /var/lib/apt/lists/*

# Install Anaconda and dependencies
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && mkdir /root/.conda \
    && bash miniconda.sh -b \
    && rm miniconda.sh
ENV PATH /root/miniconda3/bin:$PATH
RUN conda install -y python=$PYTHON_VERSION numpy

# Download mujoco
RUN mkdir /root/.mujoco \
    && wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -zxf mujoco.tar.gz -C "/root/.mujoco" \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so

# Copy to image and set workdir
COPY ./Docker/ /home/
WORKDIR /home/

# Upgrade pip
RUN pip install --upgrade pip
# Install requirements and mpi4py
RUN pip install -r /home/requirements.txt \
    && conda install -y -c conda-forge mpi4py openmpi
# Install pytorch
RUN $PYTORCH_INSTALL
# Install faiss
RUN $FAISS_INSTALL

# Remove requirements and Dockerfile
RUN rm /home/requirements.txt \
    && rm /home/Dockerfile

# Install lcer and shadowhand-gym and prepare to run the experiments
RUN chmod +x ./update.sh \
    && ./update.sh

ENTRYPOINT ["/bin/bash"]