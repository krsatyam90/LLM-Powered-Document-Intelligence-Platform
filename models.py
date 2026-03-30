"""src/feedback/models.py"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    query      = Column(Text, nullable=False)
    answer     = Column(Text, nullable=False)
    rating     = Column(Integer, nullable=False)   # 1-5
    comment    = Column(Text, nullable=True)
    session_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
