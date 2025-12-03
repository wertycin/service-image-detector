from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Point(BaseModel):
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class BoundingBox(BaseModel):
    label: str = Field(..., description="Class label of the detected object")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    top_left: Point = Field(..., description="Top-left corner of the bounding box")
    bottom_right: Point = Field(..., description="Bottom-right corner of the bounding box")


class DetectionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    url: Optional[str] = Field(None, description="URL of the image to process")
    image_bytes: Optional[str] = Field(
        None, description="Base64-encoded image bytes", alias="bytes"
    )

    @model_validator(mode="after")
    def validate_input(self) -> "DetectionRequest":
        if self.url is None and self.image_bytes is None:
            raise ValueError("Either 'url' or 'bytes' must be provided")
        if self.url is not None and self.image_bytes is not None:
            raise ValueError("Only one of 'url' or 'bytes' should be provided, not both")
        return self


class DetectionResponse(BaseModel):
    bboxes: list[BoundingBox] = Field(
        default_factory=list, description="List of detected bounding boxes"
    )


class DetectionWithImageResponse(BaseModel):
    bboxes: list[BoundingBox] = Field(
        default_factory=list, description="List of detected bounding boxes"
    )
    image: str = Field(..., description="Base64-encoded image with drawn bounding boxes")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
