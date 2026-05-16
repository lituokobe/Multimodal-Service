# Multimodal LLM Service – Intelligent Image/Video Summarization

> A localized deployment solution based on **Qwen3.5-9B + vLLM + Docker**, delivering high-precision media understanding capabilities for AI-powered short-video projects.

## 📋 Table of Contents
- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Quick Start](#-quick-start)
- [Configuration Guide](#-configuration-guide)
- [API Documentation](#-api-documentation)

---

## ✨ Features

- 🖼️ **Intelligent Image Captioning**: Automatically analyzes text overlays, colors, fonts, and layout in images, generating Chinese descriptions optimized for short-video production.
- 🎬 **Video Segment Understanding**: Splits video footage along the timeline and outputs structured, multi-dimensional descriptions including shot type, scene, subject, action, and mood.
- 🐳 **Docker Containerization**: One-click deployment with GPU passthrough support, enabling seamless cross-environment migration.
- ⚡ **vLLM-Accelerated Inference**: Leverages optimizations like Flash Attention and Chunked Prefill to boost throughput and reduce latency.
- 🔧 **Flexible Configuration**: Dynamically adjustable parameters such as GPU memory utilization, max concurrent sequences, and multimodal input limits.

---
## 💻 System Requirements

| Component       | Minimum Requirement        | Recommended Configuration              |
|----------------|---------------------------|----------------------------------------|
| GPU            | NVIDIA RTX 3090 (24GB)    | RTX 4090 / A100 (40GB+)                |
| CUDA           | 12.1+                     | 12.8                                   |
| Docker         | 24.0+                     | 29.0.1 with nvidia-container-toolkit   |
| GPU Memory     | ≥20GB                     | ≥32GB (for longer videos)              |
| OS             | Linux / WSL2              | Ubuntu 22.04 LTS                       |

> ⚠️ Ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed to enable GPU passthrough in Docker.

---

## 🚀 Quick Start

```bash
git clone http://gogs.km360.cn/lituo/multimodal-service.git

cd multimodal_service

docker compose up
```

---

## ⚙️ Configuration Guide
### Core vLLM Launch Arguments
```bash
OMP_NUM_THREADS=1 vllm serve /root/vllm_deploy/models/Qwen/Qwen3___5-9B \
  --port 8012 \                          # Internal vLLM service port
  --tensor-parallel-size 1 \             # Tensor parallelism (set to 1 for single GPU)
  --gpu-memory-utilization 0.90 \        # Max GPU memory usage to avoid OOM
  --max-model-len 20480 \                # Maximum context length
  --max-num-seqs 1 \                     # Max concurrent sequences (recommended: 1 for video tasks)
  --max-num-batched-tokens 16384 \       # Max tokens per batch
  --limit-mm-per-prompt '{"image":1,"video":1}' \  # Multimodal input limit per request
  --mm-encoder-tp-mode data \            # Multimodal encoder tensor parallelism mode
  --media-io-kwargs '{"video":{"num_frames":64}}' \ # Number of frames extracted from video
  --enable-chunked-prefill \             # Enable chunked prefill for long inputs
  --served-model-name "qwen3.5-9b" \     # Model name identifier
  --trust-remote-code \                  # Required for custom model loading
  --attention-backend FLASH_ATTN         # Use Flash Attention for acceleration
```
### Environment Variables Reference
| Variable Name                | Default Value                        | Description                              |
|----------------------------|------------------------------------|------------------------------------------|
| `MODEL_PATH`               | `/app/models/Qwen/Qwen3___5-9B`    | Local path to the model                  |
| `API_PORT`                 | `8010`                             | External API service port                |
| `VLLM_PORT`                | `8012`                             | Internal vLLM port (not exposed)         |
| `GPU_MEMORY_UTILIZATION`   | `0.90`                             | GPU memory usage ratio (range: 0.7–0.95) |
| `VIDEO_NUM_FRAMES`         | `64`                               | Number of video frames to sample (affects granularity and VRAM usage) |

> 🔍 **Port Notes**: Port `8010` is the public API gateway (for user requests), while `8012` is the internal vLLM backend port. Communication between them occurs via localhost inside the container.

---
## 📡 API Documentation

### 🔍 Health Check
```http
GET http://localhost:8010/health
```
**Response**:
```json
{
  "status": "healthy",
  "model": "Qwen3.5",
  "timestamp": "current timestamp"
}
```
### 🖼️ Image Captioning (requires HTTP URL)
```http
POST http://localhost:8010/summarize_image
Content-Type: application/json

{
  "image_path": "https://example.com/image.jpg"
}
```
**Response**:
```json
{
    "overall_summary": "A Chinese description of the image. Since this project focuses on text-overlay images, the caption includes details about colors, fonts, layout, and suitability for specific short-video content."
}
```
### 🎬 Video Footage Captioning (requires HTTP URL):
```http
POST http://localhost:8010/summarize_footage
Content-Type: application/json

{
  "footage_path": "https://example.com/video.mp4"
}
```
**Response**:
```json
{
    "segments": [
        {
            "start": "segment start time",
            "end": "segment end time",
            "shot_type": "camera shot type",
            "scene": "background setting",
            "subject": "main subject",
            "action": "subject's action",
            "emotion_vibe": "mood or atmosphere",
            "description": "detailed segment description"
        }
    ],
    "overall_summary": "Overall description of the video footage and its suitability for short-video production."
}
```