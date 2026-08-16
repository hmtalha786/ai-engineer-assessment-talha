from fastapi import APIRouter, Request

from app.models import AskRequest, AskResponse


router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Pass a validated question to the application orchestrator."""
    return await request.app.state.services.orchestrator.ask(payload.question)
