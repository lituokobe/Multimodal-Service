# This file is to use multimodal embedding model to convert text/image/video footage to vectors.

# ========= Import dependencies =========
from typing import Literal
import time
import torch
from functionals.qwen3_vl_embedding import Qwen3VLEmbedder # official wrapper from Qwen with small adaptation
from config.path_config import MULTIMODAL_EMBEDDING_PATH
from functionals.logger import multimodal_logger

# ========= Initialize the Qwen3VLEmbedder model =========
embedding_model = Qwen3VLEmbedder(model_name_or_path=MULTIMODAL_EMBEDDING_PATH)

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
        {"text": "热闹的会展现场有很多人在看车"},
        {"text": "一个男孩在玩两个恐龙玩具"},
        {"text": "一个小男孩在游乐场一个人玩一个恐龙玩具"},
        {"text": "几个女生在机场"}
    ]

    # Define a list of document texts and images
    documents = [
        # {"text": "一群人在打闹"},
        # {"text": "一个男孩在滑梯前玩很多恐龙玩具"},
        # {"text": "四个女性在接机大厅举着牌子等高先生出现"},
        # {"text": "车展上热潮涌动，很多新车发布"}

        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\boy_cfg1.0.png"}, # boy playing 1 dinosaur
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\boy_cfg0.0.png"}, # boy playing 2 dinosaurs
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\cn_text_cfg0.0.png"}, # girls at the airport
        # {"image": r"E:\Li_Tuo_work\multimodal_service\images\shoes_cfg1.0.png"},

        {"video": r"E:\Li_Tuo_work\embedding_service_multimodal\video_footage\event2.mp4"},
        {"video": r"E:\Li_Tuo_work\embedding_service_multimodal\video_footage\253998_medium.mp4"}, # Daddy carrying daughter
        {"video": r"E:\Li_Tuo_work\embedding_service_multimodal\video_footage\229275_tiny.mp4"} # 2 girls sitting
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