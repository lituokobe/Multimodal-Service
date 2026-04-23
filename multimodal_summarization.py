# This file is to use multimodal LLM to understand video footage and output structured messages.

# ========= Import dependencies =========
import time
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from config.path_config import MULTIMODAL_LLM_URL
from config.schema_config import FootageSummary
from functionals.logger import multimodal_logger
from functionals.utils import video_to_data_url, extract_json_block, get_video_duration, format_timestamp, \
    image_to_data_url, clean_image_summary

# ========= Initialize LangChain client =========
multimodal_llm = ChatOpenAI(
    model="qwen3.5-9b",  # Must match the model name vLLM registered
    base_url=MULTIMODAL_LLM_URL,  # Your local vLLM endpoint
    api_key="empty",  # vLLM doesn't require auth for local deployment
    temperature=0,
    max_tokens=8192,  # max output tokens. Note: vLLM uses max_tokens, not max_completion_tokens
)

# ========= Prepare the summary prompt =========
IMAGE_SUMMARY_PROMPT = """
你是一名专业的商业图片描述师。请仔细观察输入的图片，用200字以内输出一段清晰、实用的描述，适合用于平面设计或视频制作参考。

【输出要求】
- 请仅在内部进行分步思考，最终输出时只返回200字以内的纯文本描述
- 不要输出任何分析过程、步骤编号、标题、markdown符号（如**、#、-）
- 不要使用空行分隔，各部分用句号或分号自然连接
- 直接开始描述主题，无需前缀
- 用平实、专业的中文输出，避免主观评价（如“很好看”），只描述客观事实与明确指向的用途建议。

【内容要求】
描述需按以下顺序包含：
- 主题：图片的核心内容（人物/风景/品牌/产品/文字图形等）
- 背景与环境：场景、空间感、细节
- 文字信息：如有文字，逐字抄录并说明字体风格（如无，略过）
- 颜色与色调：主色、辅色、整体冷暖/明暗倾向
- 构图与光影：主体位置、视觉焦点、光线方向与质感
- 风格与感觉：视觉风格（如极简、复古、科技感）及情绪基调
- 商业适用场景：适合用于哪类平面设计（海报、广告、包装等）或视频制作（片头、转场、背景、宣传片等），并简述原因

现在，请根据要求描述图片。
"""

FOOTAGE_SUMMARY_PROMPT = """
你是一个专业的视频内容分析员。请仔细观看输入视频，按镜头切换或内容主题明显变化进行分段，并输出严格符合以下JSON格式的分析结果。所有描述必须使用中文，JSON键名使用英文。

【分段规则】
- 以镜头切换、场景变更或视觉主题明显变化为界进行分段。
- 时间戳格式统一为 "HH:MM:SS.mmm"（例如 "00:00:00.000"）。
- 片段时间必须连续，首片段从 "00:00:00.000" 开始。
- 视频总时长为{footage_duration}秒，（格式化后: {footage_duration_formatted}）请确保所有时间戳不超过此值，末片段结束时间等于此值。
- 若视频没有主题变换或极短（<1秒），仅输出1个片段即可。
- 

【字段要求】
segments 数组中每个对象包含：
- "start": 片段起始时间
- "end": 片段结束时间
- "shot_type": 镜头类型（如：俯拍、仰拍、近景、中景、远景、特写、跟拍、摇镜头、固定机位等）
- "scene": 场景描述（室内/室外、天气/光线、背景布局/关键道具），**请尽量包含显著可见的品牌（尽量使用中文）产品/文字，忽略背景中模糊、过小（<画面1/20）、快速闪过的文字**
- "subject": 主体信息（人物/动物/物体/无主体，可补充数量、身份或显著特征）
- "action": 主体动作或事件进展
- "emotion_vibe": 画面传递的情绪、氛围或视觉风格（如：科技感、温馨、紧张、风驰电掣、高级感等）
- "description": 一段连贯的简述，整合上述信息，适合营销素材检索与AI视频生成参考，**请尽量包含显著可见的品牌（尽量使用中文）/产品/文字，忽略背景中模糊、过小（<画面1/20）、快速闪过的文字**（≤100字）

最后附加：
- "overall_summary": 1-2句中文总结全片核心内容，并指出适合的营销方向或产品品类，**请尽量包含显著可见的品牌（尽量使用中文）产品/文字，忽略背景中模糊、过小（<画面1/20）、快速闪过的文字**（≤100字）

【输出要求】
- 仅输出合法JSON，严禁使用Markdown代码块包裹，严禁附加任何解释性文字。
- 确保时间逻辑连贯、描述精炼准确、键名与示例完全一致。
- 示例结构：
{{
  "segments": [
    {{
      "start": "00:00:00.000",
      "end": "00:00:03.500",
      "shot_type": "俯拍",
      "scene": "室外沙滩，正午强光，布局开阔",
      "subject": "多名中学生",
      "action": "激烈进行沙滩排球比赛",
      "emotion_vibe": "青春活力、激情",
      "description": "俯拍正午沙滩，一群中学生激烈打排球，画面充满青春张力。"
    }},
    {{
      "start": "00:00:03.500",
      "end": "00:00:08.200",
      "shot_type": "近景跟拍",
      "scene": "沙滩与棕榈树交界，光线柔和",
      "subject": "一名擦汗的男生",
      "action": "暂停休息，喘息擦汗",
      "emotion_vibe": "疲惫但专注",
      "description": "近景捕捉一名男生擦汗休息，突出运动后的真实质感。"
    }}
  ],
  "overall_summary": "沙滩排球少年与休息特写交替，适合运动饮料、户外服饰或快消品广告素材。"
}}
"""

# ========= Define the functions =========
def summarize_image(image_path: str,
                    multimodal_llm:ChatOpenAI = multimodal_llm,
                    max_retries: int = 2) -> str|None:
    """
    Summarize image to natural language text with multimodal_llm.
    :param image_path: the path of image
    :param multimodal_llm: multimodal_llm to understand the video, by default Qwen3.5
    :param max_retries: maximum number of retries
    :return: string of summary or nothing if error happens
    """
    message=HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
        {"type": "text", "text": IMAGE_SUMMARY_PROMPT}
    ])

    for attempt in range(max_retries):
        try:
            # Invoke model
            start_time = time.time()
            response = multimodal_llm.invoke([message])
            image_summary = response.content
            latency = round(time.time()-start_time,2)

            multimodal_logger.info(f"{image_path}总结完成，耗时{latency}秒。")

            if not isinstance(image_summary, str):
                multimodal_logger.error(f"未能总结图片{image_path}")
                return None
            # Validate against schema
            return clean_image_summary(image_summary)

        except Exception as e:
            multimodal_logger.error(f"第{attempt + 1}/{max_retries}次总结图片{image_path}，报错: {e}")
            if attempt < max_retries:
                time.sleep(0.1)
                continue
            return None

def summarize_footage(footage_path: str,
                      multimodal_llm:ChatOpenAI = multimodal_llm,
                      footage_summary_prompt: str = FOOTAGE_SUMMARY_PROMPT,
                      max_retries: int = 2) -> FootageSummary|None:
    """
    Summarize video footage to natural language text by multimodal_llm.
    :param footage_path: the path of video footage
    :param multimodal_llm: multimodal_llm to understand the video, by default Qwen3.5
    :param footage_summary_prompt: instruction to output
    :param max_retries: maximum number of retries
    :return: the data in the format of FootageSummary or nothing if error happens
    """
    # Prepare the prompt message
    footage_data_url = video_to_data_url(footage_path)
    footage_duration = get_video_duration(footage_path)
    footage_duration_formatted = format_timestamp(footage_duration)

    footage_summary_prompt_formated = footage_summary_prompt.format(
        footage_duration=footage_duration,
        footage_duration_formatted=footage_duration_formatted
    )

    message=HumanMessage(content=[
        {"type": "video_url", "video_url": {"url": footage_data_url}},
        {"type": "text", "text": footage_summary_prompt_formated}
    ])

    for attempt in range(max_retries):
        try:
            # Invoke model
            start_time = time.time()
            response = multimodal_llm.invoke([message])
            raw_output = response.content
            latency = round(time.time()-start_time,2)

            multimodal_logger.info(f"{footage_path}总结完成，耗时{latency}秒。")

            parsed_json = extract_json_block(raw_output)
            if not parsed_json:
                multimodal_logger.error(f"未能从{footage_path}总结内容中提取JSON")
                return None
            # Validate against schema
            return FootageSummary(**parsed_json)

        except ValidationError as e:
            multimodal_logger.error(f"第{attempt + 1}/{max_retries}次总结视频{footage_path}，schema矫正报错: {e}")
            if attempt<max_retries:
                time.sleep(0.1)
                continue
        except Exception as e:
            multimodal_logger.error(f"第{attempt + 1}/{max_retries}次总结视频{footage_path}，报错: {e}")
            if attempt < max_retries:
                time.sleep(0.1)
                continue
            return None

# ========= Test =========
if __name__ == "__main__":
    # result1 = summarize_footage(r"E:\Li_Tuo_work\multimodal_service\video_footage\229275_tiny.mp4")
    # print(result1)

    result2 = summarize_image(r"/\images\first_frame_20260302_145412.png")
    print(result2)

# ========= Test with text =========
# text_response = llm.invoke("Type \"I love Qwen3.5\" backwards")
# print(text_response)
# print(text_response.content)

