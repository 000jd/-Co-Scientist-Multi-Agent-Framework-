"""
Database schema and models.
"""

import json
from typing import Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import JSON, String, Float, Integer, Boolean, DateTime
from datetime import datetime, timezone

def _now():
    return datetime.now(timezone.utc)

Base = declarative_base()

class HypothesisModel(Base):
    __tablename__ = "hypotheses"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    domain_type: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, index=True)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=True, index=True)
    candidates_json: Mapped[dict] = mapped_column(JSON)
    data: Mapped[dict] = mapped_column(JSON, nullable=True) # Full backup
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

class PipelineStateModel(Base):
    __tablename__ = "pipeline_state"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    current_cycle: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String)
    state_data: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

class DatabaseManager:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.async_session = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save_hypothesis(self, hypothesis: Any) -> None:
        async with self.async_session() as session:
            model = HypothesisModel(
                id=hypothesis.id,
                domain_type="generic",
                status=hypothesis.status.value,
                candidates_json=[c.model_dump(mode='json') for c in hypothesis.candidates],
                data=hypothesis.model_dump(mode='json'),
            )
            await session.merge(model)  # Upsert
            await session.commit()

# We can instantiate global instances after config is loaded
AsyncSessionLocal = None
