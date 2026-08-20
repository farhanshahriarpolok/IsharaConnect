"""FastAPI REST routes for IsharaConnect API."""

import json
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

from backend.services.staging_service import staging_service

router = APIRouter(prefix="/api/v1")


class SignProposal(BaseModel):
    user_id: str
    bangla: str
    english: str
    category: str
    samples: list[list[list[float]]] = Field(default_factory=list, description="List of 30x126 landmark matrices")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check API and inference service status."""
    return {
        "status": "online",
        "service": "IsharaConnect API",
        "version": "1.0.0"
    }


@router.get("/dictionary")
async def get_dictionary() -> Dict[str, Any]:
    """Returns all supported BdSL signs from labels.json."""
    labels_path = Path("dataset/labels.json")
    if not labels_path.exists():
        raise HTTPException(status_code=404, detail="Dictionary not found")
        
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dictionary: {str(e)}")


@router.post("/signs/propose")
async def propose_sign(proposal: SignProposal) -> Dict[str, Any]:
    """Staging endpoint for users submitting new signs."""
    submission_id = staging_service.submit_proposal(
        label_bn=proposal.bangla,
        label_en=proposal.english,
        contributor=proposal.user_id,
        samples=proposal.samples
    )
        
    return {
        "status": "success",
        "proposal_id": submission_id,
        "message": "Sign proposal submitted for review."
    }
