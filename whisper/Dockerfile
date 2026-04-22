FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# CUDA 12.4.1 runtime is backward-compatible with RunPod's CUDA 13.0 driver.
# cudnn-runtime includes the cuDNN shared libraries that CTranslate2 requires —
# no -devel bloat (no compiler headers, no static libs).

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
