from modelscope import snapshot_download

# Download the summarization model (Qwen3.5-9B) used by multimodal_summarization.py.
# Run once to populate ./models; the container mounts ./models read-only.
model_dir = snapshot_download('Qwen/Qwen3.5-9B', cache_dir=r".\models")
print(f"Model downloaded to: {model_dir}")