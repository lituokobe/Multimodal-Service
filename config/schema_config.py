from typing import Generic, TypeVar
from pydantic import BaseModel, Field

class SummarizeImageRequest(BaseModel):
    image_path: str = Field(..., description="Path to image file")

class ImageSummary(BaseModel):
    """Response model for image summarization."""
    overall_summary: str = Field(..., description="Natural language summary of the image")

class SummarizeFootageRequest(BaseModel):
    footage_path: str = Field(..., description="Path to video footage file")

class FootageSegment(BaseModel):
    """
    Metadata of one segment of video footage.
    """
    start: str = Field(default="")
    end: str = Field(default="")
    shot_type: str = Field(default="")
    scene: str = Field(default="")
    subject: str = Field(default="")
    action: str = Field(default="")
    emotion_vibe: str = Field(default="")
    description: str = Field(default="", max_length=100)

class FootageSummary(BaseModel):
    """
    Summary of a video footage, including the info of all the segments and an overall summary.
    """
    segments: list[FootageSegment] = Field(default=[])
    overall_summary: str = Field(default="", max_length=100)

# API response schema
T = TypeVar("T")
class APIResponse(BaseModel, Generic[T]):
    """Unified API response wrapper"""
    success: bool = Field(..., description="Whether the request succeeded")
    data: T|None = Field(default=None, description="Response payload on success")
    error: str|None = Field(default=None, description="Error message on failure")
    error_code: str|None = Field(default=None, description="Machine-readable error code")

    @classmethod
    def ok(cls, data: T) -> "APIResponse[T]":
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str, error_code: str = "UNKNOWN") -> "APIResponse[T]":
        return cls(success=False, error=error, error_code=error_code)
