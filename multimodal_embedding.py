# This file is to use multimodal embedding model to convert text/image/video footage to vectors.

# ========= Import dependencies =========
from typing import Literal
import time
import torch
from functionals.qwen3_vl_embedding import Qwen3VLEmbedder # official wrapper from Qwen with small adaptation
from functionals.logger import multimodal_logger

# ========= Initialize the Qwen3VLEmbedder model =========
embedding_model = Qwen3VLEmbedder(model_name_or_path="./models/Qwen.Qwen3-VL-Embedding-2B")

# ========= Define the function =========
def embed_data(data_type: Literal["text", "image", "video"],
               data: str, # string data, or the data path for image/video footage
               embedding_model:Qwen3VLEmbedder = embedding_model,
               max_retries: int = 2) -> torch.Tensor|None:
    """
    Embed data (text, image, video) into vectors one by one, this will prevent separate max-length for each video
    :param data_type: text, image, or video. Audio files (music) can not be embedded, their description will be embedded instead
    :param data: text data, or path of image/video
    :param embedding_model: Multimodal embedding model, Qwen3VLEmbedder by defaul
    :param max_retries: maximum number of retries
    :return: embedded vectors as torch Tensors, or nothing if error happens
    """
    if data_type not in ["text", "image", "video"]:
        multimodal_logger.error(f"{data}的数据类型只能是'text','image','video'")
        return None

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = embedding_model.process(
                [
                    {data_type: data}
                ]
            )
            latency = round(time.time() - start_time, 2)
            multimodal_logger.info(f"'{data}'嵌入完成，耗时{latency}秒。")

            # Return the final embedding
            embedding = response[0]
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding)
            if embedding.device.type != "cpu":
                embedding = embedding.cpu()

            return embedding  # 🔑 Return raw bfloat16 tensor
        except Exception as e:
            multimodal_logger.error(f"第{attempt + 1}/{max_retries}次嵌入{data}，报错: {e}")
            if attempt < max_retries:
                time.sleep(0.1)
                continue
    return None

# ========= Test =========
if __name__ == "__main__":
    # result1 = embed_data(
    #     "text", "小孩子一个人在游乐场"
    # )
    # print(result1)
    #
    # result2 = embed_data(
    #     "image", r"E:\Li_Tuo_work\multimodal_service\images\boy_cfg1.0.png"
    # )
    # print(result2)
    #
    # result3 = embed_data(
    #     "video", r"E:\Li_Tuo_work\multimodal_service\video_footage\event2.mp4"
    # )
    # print(result3)
    #
    # print((result1*result2).sum())
    # print((result1*result3).sum())
    # print((result2*result3).sum())

    # ========= Test with similarity score =========
    queries = [
        # {"text": "热闹的会展现场有很多人在看车"},
        # {"text": "一个男孩在玩两个恐龙玩具"},
        # {"text": "一个小男孩在游乐场一个人玩一个恐龙玩具"},
        # {"text": "几个女生在机场"},
        {"text": "一种传统的中国风风格"},
        {"text": "突出青春的活力"},
        {"text": "家庭温馨的感觉"},
        {"text": "高端大气上档次"},
        {"text": "突出科技感"},
        {"text": "一种动感的感觉"},
        {"text": "可爱俏皮的的感觉"},
    ]

    # Define a list of document texts and images
    documents = [
        # {"text": "一群人在打闹"},
        # {"text": "一个男孩在滑梯前玩很多恐龙玩具"},
        # {"text": "四个女性在接机大厅举着牌子等高先生出现"},
        # {"text": "车展上热潮涌动，很多新车发布"}
        {"text": '画面采用极简主义风格，背景为纯白。主标题使用深灰蓝色的粗体无衬线字，字号最大，位于顶部居中。下方日期与地点信息使用同色系细体字，字号较小。底部两行深蓝色标语字体加粗，字号中等。整体排版垂直居中，留白充足，视觉重心集中在上方。色彩以冷色调的蓝灰色为主，搭配白色背景，对比鲜明。视觉调性专业、稳重且具有科技感。这种设计适合用于科技展会、商务会议或企业宣传类短视频的封面，能清晰传达信息并营造高端大气的氛围。'},
        {"text":'画面采用纯白背景，视觉中心为垂直居中的排版结构。顶部与底部文字采用圆润的描边字体，颜色为浅棕色；中间核心区域使用米黄色矩形块衬托深墨色行书大字，形成强烈的视觉层级。色彩体系以米黄、浅棕、墨黑为主，整体呈现温暖古朴的大地色系。构图上留白极多，元素分布疏朗，营造出宁静典雅的氛围。这种设计具有浓厚的中式传统韵味，适合用于茶道、非遗文化、国风生活美学等主题的短视频封面或宣传海报，能有效传达文化传承的庄重感与艺术气息。'},
        {"text": '画面采用垂直居中对齐排版，主标题使用深褐色粗体字并带有浅色描边，字号最大且位于顶部，视觉冲击力强。副标题置于浅米黄色矩形背景块内，字体较细。下方文字采用空心描边风格，字号逐行递减。色彩体系以深褐色、浅米黄、浅橙色为主，整体呈现暖色调大地色系。构图上文字集中在画面上半部分，下方留有大量空白。视觉调性偏向市井烟火气与热闹氛围，风格接地气且具食欲感。适合用于美食探店、节日促销或生活类短视频的封面设计，能有效传达热闹与美味的信息。'},
        {"text": '画面采用垂直居中对齐排版，主标题使用带有黄色描边的粗体卡通字体，字号最大，视觉冲击力强。副标题置于橙色矩形色块内，使用白色细体，形成鲜明对比。下方文字采用亮绿色和深灰色描边字体，字号依次递减。色彩体系以暖橙色和亮绿色为主，搭配黑白灰，整体明快高饱和。构图上元素集中在上方，下方大面积留白。视觉调性活泼、温馨、亲民。适合用于亲子、母婴或家庭生活类的短视频封面制作，因其色彩温暖且字体具有亲和力，能有效吸引家庭受众。'},
        {"text": '画面采用极简构图，大面积白色背景衬托上方元素。字体方面，顶部使用黄色描边白底圆体，中间主体为极粗黑体大字置于黄色矩形色块中，底部为白描黑体小字，形成强烈的字号对比与层级感。色彩体系以高饱和度的柠檬黄与纯黑为主，辅以白色，对比度极高。视觉调性醒目直接，具有强烈的促销与活动预告氛围。这种设计适合用于商业促销、展会预告或新闻资讯类短视频封面，利用高对比度色彩和粗犷字体在信息流中快速抓取用户注意力，强调核心信息。'},
        {"text": '画面采用垂直居中对齐排版，主标题使用深褐色粗体字并带有浅色描边，字号最大且位于顶部，视觉冲击力强。副标题置于浅米黄色矩形背景块内，字体较细。下方文字采用空心描边风格，字号逐行递减。色彩体系以深褐色、浅米黄、浅橙色为主，整体呈现暖色调大地色系。构图上文字集中在画面上半部分，下方留有大量空白。视觉调性偏向市井烟火气与热闹氛围，风格接地气且具食欲感。适合用于美食探店、节日促销或生活类短视频的封面设计，能有效传达热闹与美味的信息。'},
        {"text": '画面采用极简白底背景，视觉重心集中在上方。主标题使用加粗黑体，带有白色描边，字号最大，极具视觉冲击力。副标题置于芥末黄色矩形色块之上，使用白色细体字。下方辅助信息采用黄色描边字体，整体排版居中对齐，层级分明。色彩以黑、白、芥末黄为主，对比强烈。整体风格偏向运动海报，传递出活力与激情。适合用于体育赛事、校园活动或健身类短视频的封面设计，能有效吸引注意力并传达动感氛围。'},
        {"text": '字体采用特粗无衬线黑体，主标题字号巨大且带有立体描边效果，与下方细长的说明文字形成强烈粗细对比，层级分明。色彩以深蓝紫渐变为主基调，辅以高饱和度的霓虹红、白、青色，营造出强烈的冷暖对比与科技感。构图上采用垂直流式布局，背景垂直光束引导视线自上而下，底部半透明色块承载核心信息，留白节奏紧凑。视觉调性呈现现代动感与促销氛围，具有明显的赛博朋克或霓虹风格。适合用于房地产推广、电商大促或活动招募类短视频，因其高对比度和强视觉冲击力能有效吸引用户注意力并快速传递核心利益点。'},
        {"text": '字体采用粗体立体无衬线字，主标题字号巨大且带白色描边，正文为细体紫色，排版紧凑。色彩以粉紫黄暖色调渐变为主，搭配深紫与亮白，明度较高。构图上Logo居顶，标题横幅居中，下方圆角矩形承载信息，视觉动线清晰垂直。视觉调性活泼动感，具有强烈的促销与活动氛围。适配房地产促销或商业活动类短视频，因色彩吸睛且信息层级分明，能有效引导用户关注。'},
        {"text": '标题采用大号书法风格字体，笔画粗犷且带有青白渐变，极具视觉冲击力；正文使用细瘦无衬线字体，排列整齐。色彩以深蓝色为主背景，搭配亮青色与白色高光，营造冷峻的科技氛围。构图上，顶部标题居中，中间穿插类似HUD的几何边框与发光线条，背景辅以放射状光效，引导视线向下流动。整体风格呈现强烈的科技感与未来感，情绪基调专业且充满活力。这种设计非常适合用于科技类、互联网行业或企业发布会的短视频封面，能够迅速建立专业、前沿的品牌形象。'},
        {"text": '该设计采用竖屏构图，顶部为醒目的标题区域，使用粗体卡通风格字体，字号对比强烈，第二行文字带有黄色高亮背景，形成视觉焦点。色彩体系以深蓝色为主调，搭配鲜艳的黄色和橙色作为辅助色，冷暖对比鲜明，视觉冲击力强。中间留有大面积灰色区域用于放置视频画面，底部辅以几何图形装饰和说明文字。整体风格活泼、动感且年轻化，适合用于生活类、娱乐类或快节奏的短视频制作，能有效吸引年轻受众的注意力。'},


        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\boy_cfg1.0.png"}, # boy playing 1 dinosaur
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\boy_cfg0.0.png"}, # boy playing 2 dinosaurs
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\cn_text_cfg0.0.png"}, # girls at the airport
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\shoes_cfg1.0.png"},
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\雄安智能工业展.png"},
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\梅州茶道节.png"},
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\重庆火锅节.png"},
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\钦州母婴展.png"},
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\呼和浩特大型边境商业展.png"},
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\重庆火锅节.png"},
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\保定大学生赛跑.png"},
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\科技感.png"},
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\动感.png"},
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\科技大气.png"},
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\可爱俏皮.png"},

        # {"video": r"E:\Li_Tuo_work\embedding_service_multimodal\video_footage\event2.mp4"},
        # {"video": r"E:\Li_Tuo_work\embedding_service_multimodal\video_footage\253998_medium.mp4"}, # Daddy carrying daughter
        # {"video": r"E:\Li_Tuo_work\embedding_service_multimodal\video_footage\229275_tiny.mp4"} # 2 girls sitting
    ]
    # Process the inputs to get embeddings
    query_embeddings = torch.stack([embed_data(k, v) for q in queries for k, v in q.items()])
    document_embeddings = torch.stack([embed_data(k, v) for d in documents for k, v in d.items()])
    # Compute similarity scores between query embeddings and document embeddings
    similarity_scores = (query_embeddings @ document_embeddings.T)

    # Print out the similarity scores in a list format
    print(similarity_scores.tolist())

    # # =========  test with text =========
    # test_embedding = embedding_model.process(
    #     [{"text":"展会让您心情愉悦"}]
    # )
    # print(test_embedding)