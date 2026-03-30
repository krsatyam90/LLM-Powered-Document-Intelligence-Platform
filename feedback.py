"""
src/api/routes/feedback.py
POST /api/v1/feedback  – stores thumbs-up/down + optional comment.
Drives the RLHF-style weekly relevance improvement loop.
"""

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.feedback.models import FeedbackRecord
from src.feedback.db import get_db

router = APIRouter()


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5)   # 1–5 stars (1-2 → negative signal)
    comment: str | None = None
    session_id: str | None = None


class FeedbackResponse(BaseModel):
    id: int
    recorded_at: datetime


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    record = FeedbackRecord(
        query=req.query,
        answer=req.answer,
        rating=req.rating,
        comment=req.comment,
        session_id=req.session_id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return FeedbackResponse(id=record.id, recorded_at=record.created_at)
