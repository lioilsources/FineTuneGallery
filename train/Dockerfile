# kohya-ss sd-scripts trainer image (GPU box).
# Known-good combo: torch 2.1.2 + cu121 + sd-scripts v0.8.7 (SDXL-stable).
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ARG SD_SCRIPTS_REF=v0.8.7
RUN git clone https://github.com/kohya-ss/sd-scripts.git && \
    cd sd-scripts && git checkout ${SD_SCRIPTS_REF}

WORKDIR /app/sd-scripts
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir bitsandbytes==0.43.0 lion-pytorch

# Single-GPU accelerate defaults; run.sh passes configs explicitly.
ENV HF_HOME=/cache
ENTRYPOINT ["accelerate", "launch", "--num_cpu_threads_per_process", "4"]
