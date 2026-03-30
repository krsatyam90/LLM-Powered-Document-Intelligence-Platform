"""
src/feedback/rlhf_loop.py
Weekly job: pull low-rated answers from DB → generate improved
preference pairs → fine-tune the adapter (or log for DPO pipeline).

Run via: python -m src.feedback.rlhf_loop
"""

import json
from datetime import datetime, timedelta

import structlog
from sqlalchemy import select

from src.feedback.db import SessionLocal
from src.feedback.models import FeedbackRecord
from src.utils.config import settings

log = structlog.get_logger(__name__)

NEGATIVE_THRESHOLD = 2   # ratings ≤ 2 flagged for improvement


def collect_negatives(days: int = 7) -> list[dict]:
    cutoff = datetime.utcnow() - timedelta(days=days)
    db = SessionLocal()
    rows = db.execute(
        select(FeedbackRecord)
        .where(FeedbackRecord.rating <= NEGATIVE_THRESHOLD)
        .where(FeedbackRecord.created_at >= cutoff)
    ).scalars().all()
    db.close()
    return [{"query": r.query, "answer": r.answer, "rating": r.rating} for r in rows]


def build_preference_dataset(negatives: list[dict], output_path: str) -> int:
    """
    Stub: pairs each negative answer with a placeholder 'chosen' answer.
    In production, 'chosen' is retrieved from human annotators or a
    stronger model (GPT-4-class teacher).
    """
    pairs = []
    for item in negatives:
        pairs.append({
            "prompt": item["query"],
            "rejected": item["answer"],
            "chosen": "__PLACEHOLDER__",   # replaced by annotation pipeline
        })

    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)

    return len(pairs)


def run():
    log.info("rlhf_loop_start")
    negatives = collect_negatives(days=7)
    log.info("negatives_collected", count=len(negatives))

    if not negatives:
        log.info("nothing_to_improve")
        return

    out = f"data/preference_{datetime.utcnow().strftime('%Y%m%d')}.json"
    n = build_preference_dataset(negatives, out)
    log.info("preference_dataset_written", path=out, pairs=n)
    # TODO: trigger fine-tuning job on SageMaker / local LoRA trainer


if __name__ == "__main__":
    run()
