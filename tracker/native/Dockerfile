# Cross-build image for libinfer.so / infer_bench (aarch64, Debian 13 glibc to match gatos).
FROM debian:trixie
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      crossbuild-essential-arm64 cmake ninja-build git python3 python3-pip python3-venv \
      glslc ca-certificates file && rm -rf /var/lib/apt/lists/*
# Codegen needs torchgen (ships with torch) and pyyaml; shader gen needs pyyaml.
RUN python3 -m venv /opt/py && /opt/py/bin/pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu "torch==2.14.*" && \
    /opt/py/bin/pip install --no-cache-dir pyyaml ruamel.yaml
RUN apt-get update && apt-get install -y --no-install-recommends build-essential make && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/py/bin:$PATH
