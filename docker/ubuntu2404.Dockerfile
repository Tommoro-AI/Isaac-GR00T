# Ubuntu 24.04 + CUDA 12.5.1
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace:${PYTHONPATH}
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv curl ca-certificates gnupg ubuntu-keyring \
      && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
      && ln -s /usr/bin/python3 /usr/bin/python \
      && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && \
    apt-get install -y --no-install-recommends \
      netcat-openbsd dnsutils libgl1 git libvulkan-dev \
      zip unzip wget git-lfs build-essential cmake \
      vim less sudo htop man tmux ffmpeg \
      libglib2.0-0 libsm6 libxext6 libxrender-dev software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list

RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
    > /etc/apt/sources.list.d/ros2.list

RUN add-apt-repository universe && apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-jazzy-desktop python3-argcomplete ros-jazzy-rmw-cyclonedds-cpp \
    && echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV ROS_DISTRO=jazzy
ENV ROS_DOMAIN_ID=30
ENV ROS_ROOT=/opt/ros/jazzy
ENV ROS_PYTHON_VERSION=3
ENV PATH=$ROS_ROOT/bin:$PATH
ENV LD_LIBRARY_PATH=$ROS_ROOT/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=$ROS_ROOT/lib/python3.12/site-packages:$PYTHONPATH

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 && \
    pip install --no-cache-dir gpustat wandb==0.18.0

WORKDIR /workspace

COPY pyproject.toml .
RUN pip install -e .[base] --no-deps && \
    pip install --no-cache-dir tensorflow==2.16.1

RUN pip uninstall -y transformer-engine || true && \
    pip install -U --no-cache-dir --force-reinstall flash_attn==2.7.1.post4

RUN pip uninstall -y opencv-python opencv-python-headless || true && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2 /usr/local/lib/python3.12/dist-packages/cv2 || true && \
    pip install --no-cache-dir opencv_python_headless==4.11.0.86

COPY getting_started /workspace/getting_started
COPY scripts         /workspace/scripts
COPY demo_data       /workspace/demo_data
RUN pip install -e . --no-deps

RUN pip install --no-cache-dir accelerate==1.2.1

COPY gr00t /workspace/gr00t
COPY Makefile /workspace/Makefile
RUN pip install -e . --no-deps --force-reinstall

RUN pip install --no-cache-dir decord zmq diffusers && \
    pip install --no-cache-dir --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d

RUN pip install --no-cache-dir --ignore-installed kiwisolver==1.4.7 && \
    pip install --no-cache-dir --ignore-installed PyYAML==6.0.2 && \
    pip install --no-cache-dir \
      albumentations==1.4.18 av==12.3.0 blessings==1.7 fastparquet==2024.11.0 \
      gymnasium==1.0.0 hydra-core==1.3.2 imageio==2.34.2 kornia==0.7.4 \
      numpydantic==1.6.7 omegaconf==2.3.0 onnx==1.17.0 peft==0.17.0 \
      pyarrow==14.0.1 ray==2.40.0 tianshou==0.5.1 timm==1.0.14 \
      transformers==4.51.3 tyro dm-tree==0.1.8 \
      h5py==3.12.1 matplotlib==3.10.0 pandas==2.2.3 protobuf==3.20.3 \
      pydantic==2.10.6 requests==2.32.3 typing_extensions==4.12.2 \
      imageio-ffmpeg==0.4.9 tf-keras

RUN pip uninstall -y wandb
RUN pip install --upgrade pip setuptools wheel
RUN pip install --upgrade "protobuf>=4.25" "wandb>=0.17.0"
RUN pip install --no-cache-dir "protobuf<5,>=3.20.3"