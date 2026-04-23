from pydantic import BaseModel, Field

class FootageSegment(BaseModel):
    """
    Metadata of one segment of video footage.
    """
    start: str
    end: str
    shot_type: str
    scene: str
    subject: str
    action: str
    emotion_vibe: str
    description: str = Field(max_length=100)

class FootageSummary(BaseModel):
    """
    Summary of a video footage, including the info of all the segments and an overall summary.
    """
    segments: list[FootageSegment]
    overall_summary: str = Field(max_length=100)