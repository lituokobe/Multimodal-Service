"""
This file is to use multimodal LLM to understand video footage and image, and output structured description.
"""

# ========= Import dependencies =========
import os
import time
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import torch
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from config.constant_config import MAX_RETRY
from functionals.prompts import IMAGE_SUMMARY_PROMPT, FOOTAGE_SUMMARY_PROMPT
from config.path_config import MULTIMODAL_LLM_URL, STAGING_DIR
from config.schema_config import ImageSummary, FootageSummary, SummarizeImageRequest, SummarizeFootageRequest, \
    APIResponse
from functionals.logger import multimodal_logger
from functionals.utils import video_to_data_url, extract_json_block, get_video_duration, format_timestamp, \
    image_to_data_url, clean_image_summary, is_url, download_file_from_url

# ========= APIs =========
multimodal_llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, clean up on shutdown."""
    global multimodal_llm
    multimodal_logger.info("🔧 正在加载 Qwen3.5 模型...")
    start = time.time()

    multimodal_llm = ChatOpenAI(
        model="qwen3.5-9b",  # Must match the model name vLLM registered
        base_url=MULTIMODAL_LLM_URL,  # Your local vLLM endpoint
        api_key="empty",  # vLLM doesn't require auth for local deployment
        temperature=0,
        max_tokens=8192,  # max output tokens. Note: vLLM uses max_tokens, not max_completion_tokens
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False  # Directly passed to vLLM, following vLLM argument instruction
            }
        }
    )

    multimodal_logger.info(f"✅ Qwen3.5-9B 模型加载成功，耗时{time.time() - start:.2f}秒")
    yield
    # Cleanup: release GPU memory
    if multimodal_llm is not None:
        del multimodal_llm
        torch.cuda.empty_cache()
        multimodal_logger.info("🧹 Qwen3.5-9B 模型 GPU 显存释放")

app = FastAPI(
    title="Multimodal Summarization API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None
)

@app.post("/summarize_image", response_model=APIResponse[ImageSummary])
async def summarize_image(request: SummarizeImageRequest) -> APIResponse[ImageSummary]:
    """Summarize image to natural language text with multimodal_llm."""
    staged_file = None
    try:
        # ------ Check if input is URL or local path --------
        if is_url(request.image_path):
            # Download from URL
            staged_file = download_file_from_url(request.image_path, STAGING_DIR)
        else:
            # Use local file in staging directory
            staged_file = STAGING_DIR / request.image_path

        # ------ Create and validate image path --------
        if not staged_file.exists():
            e_m = f"图片文件在容器的staging路径中不存在: {staged_file}"
            multimodal_logger.error(e_m)
            return APIResponse[ImageSummary].fail(e_m, error_code="FILE_NOT_FOUND")
        if not staged_file.is_file():
            e_m = f"图片文件无效:{staged_file}"
            multimodal_logger.error(e_m)
            return APIResponse[ImageSummary].fail(e_m, error_code="FILE_INVALID")

        # ------ Describe the image --------
        message=HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": image_to_data_url(staged_file)}},
            {"type": "text", "text": IMAGE_SUMMARY_PROMPT}
        ])

        image_summary = None

        for attempt in range(MAX_RETRY):
            try:
                # Invoke model
                start_time = time.time()
                response = multimodal_llm.invoke([message])
                image_summary = response.content

                latency = round(time.time()-start_time,2)
                multimodal_logger.info(f"{staged_file}总结完成，耗时{latency}秒。")

                if not isinstance(image_summary, str):
                    e_m = f"未能总结图片{staged_file}"
                    multimodal_logger.error(e_m)
                    return APIResponse[ImageSummary].fail(e_m , error_code="SUMMARY_INVALID")

            except Exception as e:
                e_m = f"第{attempt + 1}/{MAX_RETRY}次总结图片{staged_file}，报错: {e}"
                multimodal_logger.error(e_m )
                if attempt < (MAX_RETRY-1):
                    await asyncio.sleep(0.1)
                    continue
                return APIResponse[ImageSummary].fail(e_m , error_code="LLM_INFERENCE_FAILED")

        return APIResponse.ok(
            data=ImageSummary(overall_summary=clean_image_summary(image_summary))
        )

    except Exception as e:
        e_m = f"图片描述失败: {e}"
        multimodal_logger.error(e_m)
        return APIResponse[ImageSummary].fail(e_m, error_code="LLM_INTERNAL_ERROR")

    finally:
        # 🔥 Guaranteed cleanup
        if staged_file and Path(staged_file).exists():
            try:
                if os.access(staged_file, os.W_OK):
                    Path(staged_file).unlink()
                    multimodal_logger.debug(f"清理: {staged_file}")
            except Exception as e:
                multimodal_logger.warning(f"{staged_file} 清理失败: {e}")

@app.post("/summarize_footage", response_model=APIResponse[FootageSummary])
async def summarize_footage(request: SummarizeFootageRequest) -> APIResponse[FootageSummary]:
    """Summarize footage to structured output with natural language text with multimodal_llm."""
    staged_file = None
    try:
        # ------ Check if input is URL or local path --------
        if is_url(request.footage_path):
            # Download from URL
            staged_file = download_file_from_url(request.footage_path, STAGING_DIR)
        else:
            # Use local file in staging directory
            staged_file = STAGING_DIR / request.footage_path

        # ------ Create and validate footage path --------
        if (not staged_file) or (not staged_file.exists()):
            e_m = f"视频素材文件在容器的staging路径中不存在: {staged_file}"
            multimodal_logger.error(e_m)
            return APIResponse[FootageSummary].fail(e_m, error_code="FILE_NOT_FOUND")
        if not staged_file.is_file():
            e_m = f"视频素材文件无效:{staged_file}"
            multimodal_logger.error(e_m)
            return APIResponse[FootageSummary].fail(e_m, error_code="FILE_INVALID")

        # ------ prepare the dynamic prompt for the summarization --------
        footage_duration = get_video_duration(staged_file)
        footage_duration_formatted = format_timestamp(footage_duration)

        footage_summary_prompt_formated = FOOTAGE_SUMMARY_PROMPT.format(
            footage_duration=footage_duration,
            footage_duration_formatted=footage_duration_formatted
        )

        # ------ Describe the footage --------
        message=HumanMessage(content=[
            {"type": "video_url", "video_url": {"url": video_to_data_url(staged_file)}},
            {"type": "text", "text": footage_summary_prompt_formated}
        ])

        parsed_json = {}

        for attempt in range(MAX_RETRY):
            try:
                # Invoke model
                start_time = time.time()
                response = multimodal_llm.invoke([message])
                raw_output = response.content
                latency = round(time.time()-start_time,2)

                multimodal_logger.info(f"{staged_file}总结完成，耗时{latency}秒。")

                parsed_json = extract_json_block(raw_output)
                if not parsed_json:
                    e_m = f"未能从{staged_file}的总结内容中提取JSON。raw_output:{raw_output}"
                    multimodal_logger.error(e_m)
                    return APIResponse[FootageSummary].fail(e_m, error_code="SUMMARY_INVALID")

            except Exception as e:
                e_m = f"第{attempt + 1}/{MAX_RETRY}次总结视频{staged_file}，报错: {e}"
                multimodal_logger.error(e_m)
                if attempt < (MAX_RETRY-1):
                    await asyncio.sleep(0.1)
                    continue
                return APIResponse[FootageSummary].fail(e_m, error_code="LLM_INFERENCE_FAILED")

        return APIResponse.ok(
            data=FootageSummary(**parsed_json, duration=footage_duration)
        )

    except Exception as e:
        e_m = f"视频素材描述失败: {e}"
        multimodal_logger.error(e_m)
        return APIResponse[FootageSummary].fail(e_m, error_code="LLM_INTERNAL_ERROR")

    finally:
        # 🔥 Guaranteed cleanup
        if staged_file and Path(staged_file).exists():
            try:
                if os.access(staged_file, os.W_OK):
                    Path(staged_file).unlink()
                    multimodal_logger.debug(f"清理: {staged_file}")
            except Exception as e:
                multimodal_logger.warning(f"{staged_file} 清理失败: {e}")

@app.get("/health")
async def health_check():
    """Production health check endpoint [[30]]."""
    if multimodal_llm is None:
        raise HTTPException(status_code=503, detail="Qwen3.5 模型未加载")
    return {
        "status": "healthy",
        "model": "Qwen3.5",
        "timestamp": datetime.now().isoformat()
    }


# ========= Test =========
if __name__ == "__main__":

    print("Test starts.")

    # video summary
    # result1 = summarize_footage(SummarizeFootageRequest(footage_path=r"http://videovueapi.km360.cn/static/uploads/video/20260602/20260602163335_6a1e955f7a1c9.mp4"))
    # print(result1)

    # # image (design) summary
    # result2 = []
    # for img in [
    #     r"E:\Li_Tuo_work\multimodal_service\images\雄安智能工业展.png",
    #     r"E:\Li_Tuo_work\multimodal_service\images\梅州茶道节.png",
    #     r"E:\Li_Tuo_work\multimodal_service\images\重庆火锅节.png",
    #     r"E:\Li_Tuo_work\multimodal_service\images\钦州母婴展.png",
    #     r"E:\Li_Tuo_work\multimodal_service\images\呼和浩特大型边境商业展.png",
    #     r"E:\Li_Tuo_work\multimodal_service\images\重庆火锅节.png",
    #     r"E:\Li_Tuo_work\multimodal_service\images\保定大学生赛跑.png",
    #     r"E:\Li_Tuo_work\multimodal_service\images\科技感.png",
    #     r"E:\Li_Tuo_work\multimodal_service\images\动感.png",
    #     r"E:\Li_Tuo_work\multimodal_service\images\科技大气.png",
    #     r"E:\Li_Tuo_work\multimodal_service\images\可爱俏皮.png",
    # ]:
    #     res = summarize_image(SummarizeImageRequest(image_path=img))
    #     result2.append(res)
    #
    # print(result2)

# ========= Test with text =========
# text_response = llm.invoke("Type \"I love Qwen3.5\" backwards")
# print(text_response)
# print(text_response.content)

