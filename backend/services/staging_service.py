"""Staging Service for managing new sign proposals."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class StagingService:
    """Manages the lifecycle of new sign proposals (submission, approval, rejection)."""

    def __init__(self, storage_dir: str = "dataset/pending_submissions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def submit_proposal(self, label_bn: str, label_en: str, contributor: str, samples: List[List[List[float]]]) -> str:
        """Store a new pending submission."""
        timestamp = int(datetime.now().timestamp())
        submission_id = f"sub_{timestamp}_{contributor.replace(' ', '_')}"
        
        payload = {
            "submission_id": submission_id,
            "label_bn": label_bn,
            "label_en": label_en,
            "contributor": contributor,
            "samples": samples,
            "status": "PENDING",
            "created_at": datetime.now().isoformat()
        }
        
        file_path = self.storage_dir / f"{submission_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            
        logger.info("Submission %s saved successfully.", submission_id)
        return submission_id

    def list_pending(self) -> List[Dict]:
        """List all pending submissions without loading full landmark arrays."""
        results = []
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("status") == "PENDING":
                        # Exclude large samples array for summary
                        summary = {
                            "submission_id": data.get("submission_id"),
                            "label_bn": data.get("label_bn"),
                            "label_en": data.get("label_en"),
                            "contributor": data.get("contributor"),
                            "created_at": data.get("created_at"),
                            "num_samples": len(data.get("samples", []))
                        }
                        results.append(summary)
            except Exception as e:
                logger.error("Failed to read submission %s: %s", file_path, e)
                
        # Sort by oldest first
        results.sort(key=lambda x: x["created_at"])
        return results

    def get_submission(self, submission_id: str) -> Optional[Dict]:
        """Retrieve full submission data."""
        file_path = self.storage_dir / f"{submission_id}.json"
        if not file_path.exists():
            return None
            
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_status(self, submission_id: str, new_status: str) -> bool:
        """Update the status of a submission."""
        file_path = self.storage_dir / f"{submission_id}.json"
        if not file_path.exists():
            return False
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        data["status"] = new_status
        data["updated_at"] = datetime.now().isoformat()
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        return True

    def reject_submission(self, submission_id: str) -> bool:
        """Reject and optionally clean up a submission."""
        # For audit purposes, we just mark as REJECTED
        return self.update_status(submission_id, "REJECTED")

staging_service = StagingService()
