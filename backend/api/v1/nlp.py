"""NLP Continuous Sign Language Translation API Router."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from core_engine.nlp.gloss_to_sentence import GlossToSentenceTranslator

router = APIRouter(tags=["NLP Translation"])

translator = GlossToSentenceTranslator()


class GlossTranslateRequest(BaseModel):
    glosses: List[str] = Field(..., description="Array of isolated BdSL sign glosses (e.g. ['আমি', 'ভাত', 'খাওয়া'])")


class StreamTokenRequest(BaseModel):
    token: str = Field(..., description="Single-frame sign prediction token")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model prediction confidence score")
    timestamp: Optional[float] = Field(None, description="Prediction timestamp in epoch seconds")


class TranslationResponse(BaseModel):
    raw_glosses: List[str]
    translated_text: str
    translated_en: Optional[str] = None
    confidence: float
    is_final: bool = True
    boundary_triggered: Optional[bool] = None


@router.post("/translate", response_model=TranslationResponse)
async def translate_gloss_sequence(request: GlossTranslateRequest):
    """Translates a sequence of isolated sign language glosses into natural, grammatically correct Bengali."""
    try:
        result = translator.translate(request.glosses)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error translating gloss sequence: {str(e)}"
        )


@router.post("/debounce-stream", response_model=TranslationResponse)
async def debounce_stream_token(request: StreamTokenRequest):
    """Ingests a real-time token stream and returns active debounced translation state."""
    try:
        result = translator.process_stream(
            sign_slug=request.token,
            confidence=request.confidence,
            timestamp=request.timestamp
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing stream token: {str(e)}"
        )


@router.post("/reset")
async def reset_stream():
    """Resets the internal streaming buffer and debouncer state."""
    translator.reset()
    return {"status": "success", "message": "Debouncer and translation buffers reset."}
