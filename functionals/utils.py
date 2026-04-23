import os
import json
import re
import base64
import cv2
from functionals.logger import multimodal_logger

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

def get_video_duration(path: str) -> int:
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
     pass