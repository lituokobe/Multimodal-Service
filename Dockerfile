FROM python:3.11.15-slim

# 🇨🇳 Use Alibaba Cloud mirror for Debian packages
RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 curl ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Install uv to a shared, world-readable location
ADD https://astral.sh/uv/0.11.6/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    rm -rf /root/.local/bin /uv-installer.sh

COPY pyproject.toml uv.lock ./

ENV UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
# 🔄 Add retry + timeout for unstable networks
ENV UV_RETRIES=3
ENV UV_CONNECT_TIMEOUT=30
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Create a non-root user with fixed UID/GID (1008 is me)
RUN useradd -m -u 1008 appuser && chown -R appuser:appuser /app

# Create necessary directories with correct permissions
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app/logs

# Switch to non-root user for security
USER appuser

# Expose the service port
EXPOSE 8010 8012

## Add --progress=plain to see real-time download progress
#docker compose build --progress=plain multimodal_summarization 2>&1 | Select-String "Downloading|Downloaded"