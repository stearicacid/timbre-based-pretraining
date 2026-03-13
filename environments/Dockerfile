ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG PROJECT_NAME=timbre_based_pretraining
ARG USER_NAME=your_user_name
ARG GROUP_NAME=your_user_name
ARG UID=your_user_id
ARG GID=your_group_id
ARG PYTHON_VERSION=3.8
ARG APPLICATION_DIRECTORY=/home/${USER_NAME}/${PROJECT_NAME}
ARG RUN_UV_SYNC_AT_BUILD_TIME="false"

ENV DEBIAN_FRONTEND="noninteractive" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    PYTHONPATH=${APPLICATION_DIRECTORY}

RUN chmod 1777 /tmp && \
    apt update -y && apt install --no-install-recommends -y software-properties-common gpg-agent \
    # Add deadsnakes PPA to install older Python versions
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    # Install specified Python
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip python3-dev python-is-python3 \
    build-essential openssl libssl-dev libffi-dev ca-certificates git curl ssh openssh-client \
    # Add wget for downloading CUDA packages
    wget gnupg2 \
    && update-ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit and cuDNN for TensorFlow GPU support
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y \
    # Install CUDA 11.x libraries (compatible with TensorFlow 2.10.0)
    cuda-toolkit-11-8 \
    libcudnn8=8.6.0.163-1+cuda11.8 \
    libcudnn8-dev=8.6.0.163-1+cuda11.8 \
    # Install additional CUDA libraries
    libcusolver-11-0 \
    libcusparse-11-0 \
    libcublas-11-0 \
    && apt-mark hold libcudnn8 libcudnn8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring_1.0-1_all.deb

# Optionally install TensorRT (if needed for optimization)
# This step is separated and made optional since TensorRT packages require additional setup
RUN apt-get update && apt-get install -y \
    # Download and install TensorRT manually if needed
    wget \
    && wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.4.3/tars/tensorrt-8.4.3.1.linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz \
    && tar -xzf tensorrt-8.4.3.1.linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz -C /opt/ \
    && rm tensorrt-8.4.3.1.linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    || echo "TensorRT installation skipped - not critical for basic operation"

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/opt/TensorRT-8.4.3.1/lib:${LD_LIBRARY_PATH}
ENV PATH=${CUDA_HOME}/bin:${PATH}

# Install ffmpeg for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch default python3 version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Install pip for specified Python version
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python${PYTHON_VERSION}

# Install Poethepoet directly using pip to type "poe" command without "uv run".
RUN python3 -m pip install --upgrade pip setuptools requests wheel six urllib3 \
    && python3 -m pip install uv poethepoet

# Add user. Without this, following process is executed as admin.
RUN groupadd -g ${GID} ${GROUP_NAME} \
    && useradd -ms /bin/sh -u ${UID} -g ${GID} ${USER_NAME}

USER ${USER_NAME}
WORKDIR ${APPLICATION_DIRECTORY}

COPY --chown=${UID}:${GID} pyproject.toml uv.lock ./

# If ${RUN_UV_SYNC_AT_BUILD_TIME} = "true", install Python package by uv and move .venv under ${HOME}.
# This process is for CI (GitHub Actions). To prevent overwrite by volume of docker compose, .venv is moved under ${HOME}.
RUN test ${RUN_UV_SYNC_AT_BUILD_TIME} = "true" && uv sync || echo "skip to run uv sync."
RUN test ${RUN_UV_SYNC_AT_BUILD_TIME} = "true" && mv ${APPLICATION_DIRECTORY}/.venv ${HOME}/.venv || echo "skip to move .venv."
