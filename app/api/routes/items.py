from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/items", tags=["items"])


@router.get("/", response_model=str)
def read_items() -> Any:
    """
    Retrieve items.
    """

    return "Works"
