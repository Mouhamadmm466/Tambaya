from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings

# SQLAlchemy async requires the asyncpg driver; normalize the URL scheme regardless
# of what is stored in .env (postgresql:// or postgres://)
_url = settings.database_url
for _prefix in ("postgresql://", "postgres://"):
    if _url.startswith(_prefix):
        _url = "postgresql+asyncpg" + _url[len(_prefix) - 3:]
        break

engine = create_async_engine(
    _url,
    echo=settings.app_env == "development",
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def ping_db() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
