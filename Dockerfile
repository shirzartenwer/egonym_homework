# Base image with system dependencies
FROM python:3.10-slim

# System package versions for reproducibility
ARG DEBIAN_FRONTEND=noninteractive

# Install core OS and build tools, OpenCV libs, and git
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Python environment
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --requirement requirements.txt

# Optional: Add a non-root user for dev containers
RUN useradd -ms /bin/bash devuser
USER devuser
WORKDIR /home/devuser/workspace

# Default shell
CMD ["/bin/bash"]