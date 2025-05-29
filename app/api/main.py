from fastapi import APIRouter

from app.api.routes import accent

api_router = APIRouter()
api_router.include_router(accent.router)
