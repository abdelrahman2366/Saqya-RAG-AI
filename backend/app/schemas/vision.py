"""Request/response models of the /vision routes (plant-disease detection from a photo)."""

from pydantic import BaseModel, Field

from app.schemas.query import QueryResponse


class PredictionOut(BaseModel):
    label: str = Field(..., description="The model's class label, e.g. Tomato___Late_blight")
    confidence: float = Field(..., ge=0, le=1)
    crop: str = ""
    disease: str = Field("", description="Empty when the label is a healthy plant")
    healthy: bool = False


class DiagnosisResponse(BaseModel):
    predictions: list[PredictionOut] = Field(default_factory=list, description="Top labels, most likely first")
    top: PredictionOut | None = None
    confident: bool = Field(False, description="top.confidence >= VISION_MIN_CONFIDENCE")
    message: str = Field("", description="What to tell the farmer (always says it is a suggestion, not a diagnosis)")
    advice: QueryResponse | None = Field(None, description="Treatment advice from the bulletins, when confident")
    model: str = ""
    latency_ms: int = 0


class VisionStatus(BaseModel):
    enabled: bool
    model: str | None = None
    loaded: bool = False
    failed: str | None = None
    min_confidence: float
