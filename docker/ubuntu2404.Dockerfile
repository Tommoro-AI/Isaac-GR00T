# Ubuntu 24.04 + CUDA 12.5.1
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace:${PYTHONPATH}
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Python + pip
RUN apt update && \
    apt install -y python3 python3-venv curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# System dependencies
RUN apt update && \
    apt install -y tzdata && \
    ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && \
    apt install -y \
        netcat-openbsd dnsutils \
        libgl1 git libvulkan-dev \
        zip unzip wget git-lfs build-essential cmake \
        vim less sudo htop ca-certificates man tmux ffmpeg \
        libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# ROS 2 (Jazzy)
RUN apt update && apt install -y software-properties-common && \
    add-apt-repository universe && \
    apt update && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb [arch=amd64] http://packages.ros.org/ros2/ubuntu noble main" > /etc/apt/sources.list.d/ros2-latest.list && \
    apt update && \
    apt install -y ros-jazzy-desktop python3-argcomplete && \
    echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

ENV ROS_DISTRO=jazzy
ENV ROS_DOMAIN_ID=30
ENV ROS_ROOT=/opt/ros/jazzy
ENV ROS_PYTHON_VERSION=3
ENV PATH=$ROS_ROOT/bin:$PATH
ENV LD_LIBRARY_PATH=$ROS_ROOT/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=$ROS_ROOT/lib/python3.12/site-packages:$PYTHONPATH

# PyTorch + extras
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 numpy==1.26.4
RUN pip install gpustat wandb==0.19.0

# Workspace setup
WORKDIR /workspace

COPY pyproject.toml .
RUN pip install -e .[base] --no-deps && \
    pip install 'tensorflow==2.16.1'

RUN pip uninstall -y transformer-engine || true
RUN pip install -U --force-reinstall flash_attn==2.7.1.post4

RUN pip uninstall -y opencv-python opencv-python-headless || true
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2 \
           /usr/local/lib/python3.12/dist-packages/cv2 || true
RUN pip install opencv-python==4.8.0.74

COPY getting_started /workspace/getting_started
COPY scripts         /workspace/scripts
COPY demo_data       /workspace/demo_data
RUN pip install -e . --no-deps

RUN pip install 'accelerate>=0.26.0'

COPY gr00t /workspace/gr00t
COPY Makefile /workspace/Makefile
RUN pip install -e . --no-deps --force-reinstall


RUN pip install decord zmq diffusers
RUN pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d
RUN apt update && apt install -y ros-jazzy-rmw-cyclonedds-cpp