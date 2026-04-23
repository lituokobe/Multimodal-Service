# Embedding Service Multimodal
In this project, we will deploy Qwen3-VL-Embedding-8B and Qwen3.5-9B for multimodal tasks.

## Qwen3.5-9B
Due to the constraints of Windows (vLLM not supported), the model files and environments are downloaded in the WSL system.

The environment is `vllm_deploy`, deployed with conda. In WSL, after switching to this environment, run the service with below code.

```bash
OMP_NUM_THREADS=1 \
vllm serve /root/vllm_deploy/models/Qwen/Qwen3___5-9B \
  --port 8000 \
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
  --attention-backend FLASH_ATTN
```

## Qwen3-VL-EmbEdding-2B

