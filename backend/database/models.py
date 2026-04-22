import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum, Float, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CallCategory(str, enum.Enum):
    health = "health"
    agriculture = "agriculture"
    education = "education"
    general = "general"
    unclear = "unclear"


class CallOutcome(str, enum.Enum):
    success = "success"
    error = "error"
    fallback = "fallback"


class CallLog(Base):
    __tablename__ = "call_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Africa's Talking session ID — links logs to a specific call
    call_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    # SHA-256 of the caller's phone number — enables repeat-caller analytics without
    # storing the number itself (privacy requirement)
    caller_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[Optional[CallCategory]] = mapped_column(
        Enum(CallCategory), nullable=True
    )
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    outcome: Mapped[CallOutcome] = mapped_column(Enum(CallOutcome), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Phase 3: faster-whisper transcription metadata
    transcription_succeeded: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    transcription_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    no_speech_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_log_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
