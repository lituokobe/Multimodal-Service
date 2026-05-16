# 多模态大模型服务 - 图片/视频智能描述

> 基于 Qwen3.5-9B + vLLM + Docker 的本地化部署方案，为 AI 短视频项目提供高精度的素材理解能力。

## 📋 目录
- [功能特性](#-功能特性)
- [环境要求](#-环境要求)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [API 接口文档](#-api-接口文档)

---

## ✨ 功能特性

- 🖼️ **图片智能描述**：自动解析图片中的文字贴图、颜色、字体、排版，并输出适用于短视频制作的中文描述
- 🎬 **视频分段理解**：对视频素材进行时间轴分段，输出镜头类型、场景、主体、动作、氛围等多维度结构化描述
- 🐳 **Docker 容器化**：一键部署，支持 GPU 直通，便于跨环境迁移
- ⚡ **vLLM 加速推理**：支持 Flash Attention、Chunked Prefill 等优化，提升吞吐与响应速度
- 🔧 **灵活配置**：支持显存利用率、最大序列数、多模态输入限制等参数动态调整

---
## 💻 环境要求

| 组件        | 最低要求                | 推荐配置                               |
|------------|------------------------|--------------------------------------|
| GPU        | NVIDIA RTX 3090 (24GB) | RTX 4090 / A100 (40GB+)              |
| CUDA       | 12.1+                  | 12.8                                 |
| Docker     | 24.0+                  | 29.0.1 with nvidia-container-toolkit |
| 显存        | ≥20GB                  | ≥32GB（支持更长视频）                   |
| 系统        | Linux / WSL2           | Ubuntu 22.04 LTS                     |

> ⚠️ 请确保已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 以支持 Docker GPU 直通。

---

## 🚀 快速开始

```bash
git clone http://gogs.km360.cn/lituo/multimodal-service.git

cd multimodal_service

docker compose up
```

---

## ⚙️ 配置说明
### 核心启动参数（vLLM）
```bash
OMP_NUM_THREADS=1 \
vllm serve /root/vllm_deploy/models/Qwen/Qwen3___5-9B \
  --port 8012 \                          # vLLM 内部服务端口
  --tensor-parallel-size 1 \             # 多卡并行数（单卡设为1）
  --gpu-memory-utilization 0.90 \        # 显存占用上限，避免 OOM
  --max-model-len 20480 \                # 最大上下文长度
  --max-num-seqs 1 \                     # 并发序列数（视频任务建议设为1）
  --max-num-batched-tokens 16384 \       # 批处理 token 上限
  --limit-mm-per-prompt '{"image":1,"video":1}' \  # 单请求多模态限制
  --mm-encoder-tp-mode data \            # 多模态编码器并行模式
  --media-io-kwargs '{"video":{"num_frames":64}}' \ # 视频抽帧数
  --enable-chunked-prefill \             # 启用分块预填充，提升长文本效率
  --served-model-name "qwen3.5-9b" \     # 模型标识名
  --trust-remote-code \                  # 信任远程代码（加载自定义模型必需）
  --attention-backend FLASH_ATTN         # 使用 Flash Attention 加速
```
### 环境变量参考
| 变量名                      | 默认值                             | 说明                 |
|--------------------------|---------------------------------|--------------------|
| `MODEL_PATH`             | `/app/models/Qwen/Qwen3___5-9B` | 模型本地路径             |
| `API_PORT`               | `8010`                          | 对外 API 服务端口        |
| `VLLM_PORT`              | `8012`                          | vLLM 内部端口（无需暴露）    |
| `GPU_MEMORY_UTILIZATION` | `0.90`                          | 显存使用比例（0.7~0.95）   |
| `VIDEO_NUM_FRAMES`       | `64`                            | 视频抽帧数量，影响理解粒度与显存占用 |

> 🔍 **端口说明**：`8010` 为 API 网关端口（用户调用），`8012` 为 vLLM 后端端口（内部通信），容器内通过 localhost 转发。

---
## 📡 API 接口文档

### 🔍 健康检查
```http
GET http://localhost:8010/health
```
**响应**：
```json
{
  "status": "healthy",
  "model": "Qwen3.5",
  "timestamp": "当前时间"
}
```
### 🖼️ 图片描述（必须是http网络图片）
```http
POST http://localhost:8010/summarize_image
Content-Type: application/json

{
  "image_path": "https://example.com/image.jpg"
}
```
**响应**：
```json
{
    "overall_summary": "图片的中文描述。本项目中的图片主要是文字贴图，描述中会包含颜色、字体、排版，以及适用于何种短视频制作。"
}
```
### 🎬 视频素材描述（必须是http网络视频素材）：
```http
POST http://localhost:8010/summarize_footage
Content-Type: application/json

{
  "footage_path": "https://example.com/video.mp4"
}
```
**响应**：
```json
{
    "segments": [
        {
            "start": "片段开始时间",
            "end": "片段结束时间",
            "shot_type": "片段拍摄镜头机位",
            "scene": "片段背景",
            "subject": "片段主体",
            "action": "片段主体动作",
            "emotion_vibe": "片段氛围",
            "description": "片段描述"
        }
    ],
    "overall_summary": "整体视频素材描述，适用何种短视频制作。"
}
```

