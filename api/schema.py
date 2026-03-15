"""api/schema.py — Pydantic models."""
from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int = Field(..., example=42)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)


class RecommendedItem(BaseModel):
    item_id: int
    original_id: str = Field(..., description="Original itemid from Retailrocket dataset")
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendedItem]


class HealthResponse(BaseModel):
    status: str = "ok"
    n_users: int
    n_items: int
    deepfm_loaded: bool
