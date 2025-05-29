import os
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse

from app.utils import predict_accent, process_url_for_audio

router = APIRouter(prefix="/accent", tags=["accent"])
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")


@router.post("/detect_accent")
async def detect_accent_api(url: Annotated[str, Form()]):
    output_dir = UPLOAD_FOLDER
    audio_path_or_error = await process_url_for_audio(url, output_dir)

    if isinstance(audio_path_or_error, str) and "Error" in audio_path_or_error:
        raise HTTPException(status_code=500, detail=audio_path_or_error)

    audio_path = audio_path_or_error
    prediction_result = await predict_accent(audio_path)
    return JSONResponse(content=prediction_result)
