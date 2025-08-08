# Base image with system dependencies - support both CPU and GPU
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

# System package versions for reproducibility
ARG DEBIAN_FRONTEND=noninteractive

# Conditional installation based on base image
# For python:3.10-slim (default): Python/pip already installed
# For CUDA images: Need to install Python/pip
RUN apt-get update && \
    if python --version 2>/dev/null; then \
        # Python base image - install minimal packages \
        apt-get install -y build-essential git libopencv-dev python3-opencv; \
    else \
        # CUDA base image - install Python and packages \
        apt-get install -y build-essential git python3 python3-pip python3-dev libopencv-dev python3-opencv && \
        ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Python environment
COPY requirements.txt .
RUN if [ -f /etc/debian_version ] && grep -q "12" /etc/debian_version; then \
        # Debian 12 (Python base image) - normal pip \
        pip install --no-cache-dir --force-reinstall --requirement requirements.txt; \
    else \
        # Ubuntu 24.04 (CUDA base image) - needs --break-system-packages \
        pip install --no-cache-dir --force-reinstall --requirement requirements.txt --break-system-packages; \
    fi

# Optional: Add a non-root user for dev containers
RUN useradd -ms /bin/bash devuser
USER devuser
WORKDIR /home/devuser/workspace

# Default shell
CMD ["/bin/bash"]