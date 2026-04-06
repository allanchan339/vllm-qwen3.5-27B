#!/bin/bash

# CUDA PATH SETTINGS
export CUDA_HOME=/usr
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Safe, Speed-Focused Env Vars
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_CUMEM_ENABLE=0
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring

# Optional: Clean cache
rm -rf ~/.cache/flashinfer

source /home/cychan/vllm/.venv_sglang/bin/activate

# FINAL CORRECTED CONFIG - April 2026
# Port 8000 (vLLM standard), --dtype auto for pre-quantized model
python -m sglang.launch_server \
  --model-path Intel/Qwen3.5-27B-int4-AutoRound \
  --served-model-name vllm/Qwen3.5-27B \
  --dtype auto \
  --trust-remote-code \
  --tp 2 \
  --context-length 220000 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 4096 \
  --max-running-requests 8 \
  --kv-cache-dtype fp8_e4m3 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port 8000