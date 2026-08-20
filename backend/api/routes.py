"""FastAPI REST routes for IsharaConnect API."""

import json
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1")


class SignProposal(BaseModel):
    user_id: str
    bangla: str
    english: str
    category: str


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
    pending_dir = Path("dataset/pending")
    pending_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple JSON save for staging (in a real app, this would be a DB insert)
    import time
    proposal_id = f"prop_{int(time.time())}_{proposal.user_id}"
    
    file_path = pending_dir / f"{proposal_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(proposal.model_dump(), f, indent=2)
        
    return {
        "status": "success",
        "proposal_id": proposal_id,
        "message": "Sign proposal submitted for review."
    }
