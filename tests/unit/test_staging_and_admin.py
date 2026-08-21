"""Unit tests for Staging Service and Admin workflows."""

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.staging_service import StagingService
from backend.auth.dependencies import get_current_user
from backend.database.models import User

# Use a test directory for staging
TEST_STAGING_DIR = "dataset/test_pending_submissions"


@pytest.fixture
def staging_service():
    """Fixture providing a fresh StagingService pointing to a test dir."""
    service = StagingService(storage_dir=TEST_STAGING_DIR)
    yield service
    # Cleanup after test
    if Path(TEST_STAGING_DIR).exists():
        shutil.rmtree(TEST_STAGING_DIR)


@pytest.fixture
def test_client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_submit_and_list_proposal(staging_service):
    """Test submitting a proposal and listing it."""
    # Submit
    sub_id = staging_service.submit_proposal(
        label_bn="বাংলা", 
        label_en="Bangla", 
        contributor="Test User", 
        samples=[[[0.1]*126]*30]*5 # 5 samples of 30 frames of 126 features
    )
    
    assert sub_id.startswith("sub_")
    
    # Retrieve full
    sub = staging_service.get_submission(sub_id)
    assert sub is not None
    assert sub["label_bn"] == "বাংলা"
    assert sub["status"] == "PENDING"
    
    # List pending
    pending = staging_service.list_pending()
    assert len(pending) == 1
    assert pending[0]["submission_id"] == sub_id


def test_approve_and_reject_flow(staging_service):
    """Test updating status to APPROVED and REJECTED."""
    sub_id = staging_service.submit_proposal(
        label_bn="ধন্যবাদ", 
        label_en="Thank you", 
        contributor="Contributor 2", 
        samples=[[[0.2]*126]*30]*5
    )
    
    # Approve
    success = staging_service.update_status(sub_id, "APPROVED")
    assert success is True
    assert staging_service.get_submission(sub_id)["status"] == "APPROVED"
    
    # Should not appear in pending list anymore
    pending = staging_service.list_pending()
    assert len(pending) == 0
    
    # Test rejection (status becomes REJECTED)
    rejected_success = staging_service.reject_submission(sub_id)
    assert rejected_success is True
    assert staging_service.get_submission(sub_id)["status"] == "REJECTED"


@patch("backend.api.admin_routes.staging_service")
def test_admin_approve_endpoint(mock_service, test_client):
    """Test the admin approve endpoint triggers correctly."""
    # Setup mock
    mock_service.get_submission.return_value = {"status": "PENDING"}
    mock_service.update_status.return_value = True
    
    mock_admin = User(id="admin_1", email="admin@ishara.local", role="SUPER_ADMIN")
    app.dependency_overrides[get_current_user] = lambda: mock_admin

    try:
        # Note: background tasks run immediately in TestClient
        response = test_client.post("/api/v1/admin/approve/sub_123")
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        mock_service.update_status.assert_called_once_with("sub_123", "APPROVED")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
