# subtune: Gradio app + training, with PyTorch for the AMD R9700 (ROCm) by default.
FROM python:3.12-slim

# PyTorch wheel index. ROCm wheels bundle the ROCm user-space libraries; the host only needs the
# amdgpu kernel driver. For a machine without an AMD GPU (Modal-only), build with
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SUBTUNE_DATA_DIR=/data \
    SUBTUNE_OUTPUT_DIR=/outputs \
    HF_HOME=/hf-cache

# Runtime libraries the ROCm PyTorch wheel links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libnuma1 libdrm2 libdrm-amdgpu1 libelf1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installed first, in its own layer: it is by far the largest download.
RUN pip install torch --index-url "${TORCH_INDEX_URL}"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[train,qlora,cloud]"

EXPOSE 7860
CMD ["subtune", "ui", "--host", "0.0.0.0", "--port", "7860"]
