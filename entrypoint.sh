#!/bin/bash
#set -e  # Exit on error
#set -u  # Treat unset variables as errors
#set -o pipefail  # Catch errors in pipelines
set -euo pipefail # combine all 3 above

# 🔧 CRITICAL: Set a complete PATH for appuser (empty PATH is the root cause)
export PATH="/app/.venv/bin:/usr/local/bin:/usr/bin:/bin"

# Also set PYTHONPATH for module resolution
export PYTHONPATH=/app

# 🔎 DEBUG: Verify environment
echo "🔍 检查虚拟环境... (PATH: $PATH)"
echo "🔍 entrypoint.sh 开始 $(date)" >> /app/logs/debug.log
echo "USER=$(id)" >> /app/logs/debug.log
echo "PATH=$PATH" >> /app/logs/debug.log
echo "PYTHONPATH=$PYTHONPATH" >> /app/logs/debug.log

# Verify critical binaries exist (use absolute paths)
VENV_PYTHON="/app/.venv/bin/python"
UV_BIN="/usr/local/bin/uv"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "❌ Python 未找到: $VENV_PYTHON" >&2
  exit 1
fi
if [ ! -x "$UV_BIN" ]; then
  echo "❌ uv 未找到: $UV_BIN" >&2
  exit 1
fi

if [ ! -d "/app/.venv/bin" ]; then
    echo "❌ /app/.venv/bin/ 不存在"
    exit 1
fi

# Check if critical executables exist
for cmd in uv vllm uvicorn; do
    if ! command -v $cmd &>/dev/null; then
        # Try via uv run as fallback
        if ! uv run which $cmd &>/dev/null; then
            echo "❌ 命令 '$cmd' 未找到 (PATH: $PATH)"
            exit 1
        fi
    fi
done
echo "✅ 环境检查通过"

# Cleanup function for graceful shutdown
cleanup() {
    echo "🛑 正在关闭服务..."
    # Stop vLLM if running
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "🔄 停止 vLLM 进程 (PID: $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGTERM SIGINT EXIT  # ✅ Also trap EXIT for safety

# Ensure logs directory is writable by current user
if [ ! -w /app/logs ]; then
  chmod 755 /app/logs 2>/dev/null || true
fi

# Start vLLM in background
echo "🚀 开启 vLLM 服务..."
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
uv run vllm serve /app/models/Qwen/Qwen3___5-9B \
  --port 8012 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 20480 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 16384 \
  --limit-mm-per-prompt '{"image": 1, "video": 1}' \
  --mm-encoder-tp-mode data \
  --media-io-kwargs '{"video": {"num_frames": 64}}' \
  --enable-chunked-prefill \
  --served-model-name "qwen3.5-9b" \
  --trust-remote-code \
  --attention-backend FLASH_ATTN &
VLLM_PID=$!

# Small delay to let vLLM start initializing
sleep 3

# Wait for vLLM to be ready (with timeout)
echo "⏳ 等待 vLLM 健康检查..."
TIMEOUT=500  # 500 seconds max
ELAPSED=0
while ! curl -s -f http://localhost:8012/health >/dev/null 2>&1; do
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "❌ vLLM 启动超时 (${TIMEOUT}s)"
        echo "📋 最后 20 行 vLLM 日志:"
        tail -20 /app/logs/vllm.log 2>/dev/null || echo "(无日志)"
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi
    # Print progress dot every 10 seconds
    if [ $((ELAPSED % 10)) -eq 0 ]; then
        echo -n "⏳"
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done
echo "✅ vLLM 准备就绪"

# Verify vLLM process is still alive
if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "❌ vLLM 进程意外退出"
    exit 1
fi

# Start FastAPI wrapper in foreground
export PYTHONPATH=/app:$PYTHONPATH
echo "🚀 开启 FastAPI 服务..."
# Use absolute paths to avoid PATH issues
if ! "$VENV_PYTHON" -m uvicorn multimodal_summarization:app \
  --host 0.0.0.0 --port 8010 --workers 1 --log-level debug \
  >> /app/logs/fastapi.log 2>&1; then
  echo "❌ FastAPI 启动失败" >&2
  # Also log the error to debug.log for visibility
  tail -20 /app/logs/fastapi.log >> /app/logs/debug.log 2>&1
  exit 1
fi