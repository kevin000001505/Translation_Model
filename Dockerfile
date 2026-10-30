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

# Runtime libraries the ROCm PyTorch wheel links against, and the tools to build llama.cpp.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libnuma1 libdrm2 libdrm-amdgpu1 libelf1 \
       git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# llama.cpp for exporting fine-tuned models to GGUF: its converters plus llama-quantize and
# llama-export-lora. Keep the tag in sync with LLAMA_CPP_TAG in src/subtune/gguf_export.py.
ARG LLAMA_CPP_TAG=b11989
ENV SUBTUNE_LLAMA_CPP=/opt/llama.cpp
RUN git clone -q --depth 1 --branch "${LLAMA_CPP_TAG}" https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
       -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    && cmake --build /opt/llama.cpp/build --target llama-quantize llama-export-lora -j "$(nproc)"

WORKDIR /app

# Installed first, in its own layer: it is by far the largest download.
RUN pip install torch --index-url "${TORCH_INDEX_URL}"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[train,qlora,cloud]"

EXPOSE 7860
CMD ["subtune", "ui", "--host", "0.0.0.0", "--port", "7860"]
