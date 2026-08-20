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
    assert len(sub["samples"]) == 5
    
    # List pending (should exclude full samples array)
    pending_list = staging_service.list_pending()
    assert len(pending_list) == 1
    assert pending_list[0]["submission_id"] == sub_id
    assert "samples" not in pending_list[0]
    assert pending_list[0]["num_samples"] == 5


def test_reject_proposal(staging_service):
    """Test rejecting a proposal updates its status."""
    sub_id = staging_service.submit_proposal("test", "test", "user", [])
    
    success = staging_service.reject_submission(sub_id)
    assert success is True
    
    # Verify status changed
    sub = staging_service.get_submission(sub_id)
    assert sub["status"] == "REJECTED"
    
    # List pending should now be empty
    pending_list = staging_service.list_pending()
    assert len(pending_list) == 0


@patch("backend.api.admin_routes.staging_service")
def test_admin_approve_endpoint(mock_service, test_client):
    """Test the admin approve endpoint triggers correctly."""
    # Setup mock
    mock_service.get_submission.return_value = {"status": "PENDING"}
    mock_service.update_status.return_value = True
    
    # Note: background tasks run immediately in TestClient
    response = test_client.post("/api/v1/admin/approve/sub_123")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    mock_service.update_status.assert_called_once_with("sub_123", "APPROVED")
