#!/bin/bash

# vLLM Benchmark Script
# Tests server performance with concurrent requests

set -e

# Activate virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# Configuration
MODEL="QuantTrio/Qwen3.5-27B-AWQ"
BASE_URL="http://localhost:8000"
ENDPOINT="/v1/chat/completions"
INPUT_LEN=204800
OUTPUT_LEN=512
NUM_PROMPTS=10
REQUEST_RATE=4

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo "❌ Error: vLLM is not installed or not in PATH"
    echo "Install with: pip install vllm"
    exit 1
fi

# Check if server is running
echo "🔍 Checking if server is running at $BASE_URL..."
if ! curl -s "$BASE_URL/models" > /dev/null; then
    echo "❌ Error: Server is not responding at $BASE_URL"
    echo "Make sure vLLM server is started first"
    exit 1
fi
echo "✅ Server is running"

echo ""
echo "🚀 Starting vLLM Benchmark..."
echo "━━━━━━━━ Parameters ="
echo "  Model:          $MODEL"
echo "  Endpoint:       $ENDPOINT"
echo "  Input tokens:   $INPUT_LEN"
echo "  Output tokens:  $OUTPUT_LEN"
echo "  Num prompts:    $NUM_PROMPTS"
echo "  Request rate:   $REQUEST_RATE req/s"
echo ""

# Run benchmark
vllm bench serve \
    --backend openai-chat \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len $INPUT_LEN \
    --random-output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS \
    --request-rate $REQUEST_RATE

echo ""
echo "✅ Benchmark completed!"
