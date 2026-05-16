from pathlib import Path

# Get project folder dir
current_file = Path(__file__).resolve()
project_dir = current_file.parent.parent

ENV_PATH = project_dir / ".env"
LOG_PATH = project_dir / "logs"

# MULTIMODAL_EMBEDDING_PATH = project_dir / "models/Qwen/Qwen3-VL-Embedding-2B"

MULTIMODAL_LLM_URL = "http://localhost:8012/v1"
STAGING_DIR = Path("/app/staging")