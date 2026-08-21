"""Admin REST routes for IsharaConnect API."""

import logging
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from backend.auth.dependencies import require_role
from pydantic import BaseModel

from backend.services.staging_service import staging_service

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_role(["SUPER_ADMIN", "ADMIN", "LINGUISTIC_REVIEWER"]))]
)
logger = logging.getLogger(__name__)


@router.get("/pending")
async def list_pending_submissions() -> Dict[str, List[Dict]]:
    """List all pending sign submissions."""
    results = staging_service.list_pending()
    return {"pending_submissions": results}


@router.post("/approve/{submission_id}")
async def approve_submission(submission_id: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Approve a submission and trigger retraining worker."""
    submission = staging_service.get_submission(submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
        
    if submission.get("status") != "PENDING":
        raise HTTPException(status_code=400, detail="Submission is not in PENDING state")
        
    # Mark as APPROVED
    success = staging_service.update_status(submission_id, "APPROVED")
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update status")
        
    # Dispatch retraining task in background
    # In a full distributed system, this would be a Celery/Redis queue task
    import subprocess
    import sys
    from pathlib import Path
    
    def trigger_retraining():
        script_path = Path("scripts/retrain_worker.py").resolve()
        logger.info("Triggering background retraining for submission %s", submission_id)
        try:
            # Run in background process so we don't block asyncio loop or risk memory leaks in main process
            subprocess.Popen([sys.executable, str(script_path), "--submission", submission_id])
        except Exception as e:
            logger.error("Failed to start retraining worker: %s", e)

    background_tasks.add_task(trigger_retraining)
    
    return {
        "status": "success",
        "message": f"Submission {submission_id} approved. Retraining pipeline initiated."
    }


@router.post("/reject/{submission_id}")
async def reject_submission(submission_id: str) -> Dict[str, Any]:
    """Reject a submission and mark it for cleanup."""
    submission = staging_service.get_submission(submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
        
    if submission.get("status") != "PENDING":
        raise HTTPException(status_code=400, detail="Submission is not in PENDING state")
        
    success = staging_service.reject_submission(submission_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update status")
        
    return {
        "status": "success",
        "message": f"Submission {submission_id} rejected."
    }
