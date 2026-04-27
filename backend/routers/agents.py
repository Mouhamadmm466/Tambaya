from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.router_agent import router_agent

router = APIRouter(prefix="/api/agents", tags=["agents"])


class RouteRequest(BaseModel):
    text: str


@router.post("/route")
async def route_text(req: RouteRequest):
    """Dev endpoint: test routing on any Hausa text.

    Returns the classified category and the raw LLM response for inspection.
    Remove or restrict before production.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    result = await router_agent.classify(req.text)
    return {
        "category": result.category.value,
        "raw_response": result.raw_response,
    }
