from fastapi import APIRouter

from database.connection import ping_db

router = APIRouter(tags=["system"])


@router.get("/health")
async def health_check():
    db_ok = await ping_db()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
    }
