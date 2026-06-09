import os
import json
import re
import base64
import uuid
import requests
from pathlib import Path
from urllib.parse import urlparse
import cv2
from functionals.logger import multimodal_logger

def is_url(path: str) -> bool:
    """Check if path is a URL."""
    return path.startswith(('http://', 'https://', 'ftp://'))

def download_file_from_url(url: str, staging_dir: Path|str) -> Path|None:
    """Download file from URL to staging directory using urllib (no external deps)."""
    if isinstance(staging_dir, str):
        staging_dir = Path(staging_dir)

    local_path = None  # Initialize to avoid scope issues

    try:
        # Generate unique filename
        parsed = urlparse(url)
        original_filename = Path(parsed.path).name or f"download_{uuid.uuid4().hex[:8]}"

        # Determine extension from URL
        if '.' not in original_filename:
            if 'image' in url.lower() or any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                original_filename += '.jpg'
            elif 'video' in url.lower() or any(url.lower().endswith(ext) for ext in ['.mp4', '.mov', '.avi']):
                original_filename += '.mp4'
            else:
                original_filename += '.bin'

        local_path = staging_dir / original_filename
        multimodal_logger.info(f"📥 从该URL下载: {url}")
        multimodal_logger.info(f"📁 保存到: {local_path}")

        # 🔑 CRITICAL: Add User-Agent header to avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8',
            'Referer': f'https://{parsed.netloc}/',
            'Sec-Fetch-Dest': 'image',
            'Sec-Fetch-Mode': 'no-cors',
            'Sec-Fetch-Site': 'same-origin',
        }

        # Use a Session for cookie persistence if needed
        with requests.Session() as session:
            response = session.get(url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        # Verify download succeeded
        if local_path.stat().st_size == 0:
            raise RuntimeError(f"下载文件为空或缺失: {url}")

        multimodal_logger.info(f"✅ 下载完成: {local_path} ({local_path.stat().st_size} bytes)")
        return local_path

    except Exception as e:
        multimodal_logger.error(f"❌ 下载失败{url}: {e}")
        # Clean up partial file if it exists
        if 'local_path' in locals() and local_path.exists():
            try:
                local_path.unlink()
                multimodal_logger.debug(f"🗑️ 清理失败下载: {local_path}")
            except:
                pass
        return None

def image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

def clean_image_summary(raw_text: str) -> str:
    # 1. Remove any block between <think> and </think> (non-greedy, across lines)
    raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)

    # 2. If </think> still exists (e.g., content before closing tag without opening), keep only what's after the LAST </think>
    if '</think>' in raw_text:
        raw_text = raw_text.split('</think>')[-1]

    # 3. Clean up whitespace: collapse multiple blank lines → single newline, trim trailing spaces per line, strip ends
    raw_text = re.sub(r'\n\s*\n', '\n', raw_text)  # collapse blank lines
    raw_text = '\n'.join(line.rstrip() for line in raw_text.split('\n'))  # trim trailing spaces
    raw_text = raw_text.strip()  # trim overall leading/trailing whitespace

    return raw_text

def video_to_data_url(path: str, max_size_mb: int = 50) -> str:
    """Convert local video to data URL (warning: base64 inflates size ~33%)"""
    size_mb = os.path.getsize(path) / 1024 / 1024
    if size_mb > max_size_mb:
        raise ValueError(f"Video too large ({size_mb:.1f}MB > {max_size_mb}MB). Use remote URL or trim first.")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    multimodal_logger.info(f"已把视频{path}转化为base64。")
    return f"data:video/mp4;base64,{b64}"

def extract_json_block(text: str) -> dict|None:
    """
    :param text: output from multimodal model of interpreting a video footage
    :return: a structured JSON
    """
    # Safety net: strip <think>...</think> blocks in case there is any
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Strategy 1: Try direct json.loads first (if model outputs pure JSON)
    try:
        return json.loads(text.strip())
    except:
        pass

    # Strategy 2: Remove markdown code blocks
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

    # Strategy 3: Find the outermost {...} or [...] block
    # Handles nested braces by counting depth
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    end = -1
    for i, c in enumerate(text[start:], start):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        # 🆕 截断检测：尝试修复不完整的 JSON
        candidate = text[start:] + "}"  # 简单补全
        try:
            import json_repair
            return json_repair.repair_json(candidate, return_objects=True)
        except:
            return None

    candidate = text[start:end]
    try:
        return json.loads(candidate)
    except:
        try:
            import json_repair
            return json_repair.repair_json(candidate, return_objects=True)
        except:
            return None

def get_video_duration(path: str) -> float:
    """get the duration of a video"""
    try:
        cap = cv2.VideoCapture(path)
        duration = round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS), 3)
        cap.release()
        multimodal_logger.info(f"视频{path}时长为{duration}秒。")
    except Exception as e:
        duration = 0.0
        multimodal_logger.error(f"视频{path}无法提取时长: {e}")
    return duration

def format_timestamp(seconds: float) -> str:
    """convert seconds to HH:MM:SS.mmm format"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


if __name__ == "__main__":
     download_file_from_url(
         "https://imgs.design006.com/202204/Design006_8QGaeTCrEK.jpg",
         "E:\Li_Tuo_work\multimodal_service\staging")