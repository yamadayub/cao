"""Unit tests for generation API endpoints."""

import base64
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Mock Supabase before importing app
@pytest.fixture(autouse=True)
def mock_supabase():
    """Mock Supabase client for all tests."""
    with patch("app.services.supabase_client.get_supabase_client") as mock:
        # Create mock client
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def client(mock_supabase):
    """Create test client."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def sample_base64_image():
    """Create a minimal valid PNG image as base64."""
    # 1x1 pixel white PNG
    png_data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
        0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
        0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,  # IEND chunk
        0x44, 0xAE, 0x42, 0x60, 0x82,
    ])
    return base64.b64encode(png_data).decode("utf-8")


class TestCreateGenerationJob:
    """Tests for POST /api/v1/simulations/generate"""

    def test_create_morph_job_success(self, client, mock_supabase, sample_base64_image):
        """Test creating a morph job successfully."""
        # Mock the database insert
        mock_supabase.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{
                "id": "test-job-id",
                "mode": "morph",
                "status": "queued",
                "progress": 0,
                "strength": 0.5,
                "parts": [],
                "created_at": "2026-01-26T00:00:00+00:00",
                "started_at": None,
                "completed_at": None,
                "error": None,
                "result_image_path": None,
            }]
        )

        response = client.post(
            "/api/v1/simulations/generate",
            json={
                "base_image": sample_base64_image,
                "target_image": sample_base64_image,
                "mode": "morph",
                "strength": 0.5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["job_id"] == "test-job-id"
        assert data["data"]["status"] == "queued"

    def test_create_parts_job_success(self, client, mock_supabase, sample_base64_image):
        """Test creating a parts job successfully."""
        mock_supabase.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{
                "id": "test-job-id",
                "mode": "parts",
                "status": "queued",
                "progress": 0,
                "strength": 0.7,
                "parts": ["eyes", "nose"],
                "created_at": "2026-01-26T00:00:00+00:00",
                "started_at": None,
                "completed_at": None,
                "error": None,
                "result_image_path": None,
            }]
        )

        response = client.post(
            "/api/v1/simulations/generate",
            json={
                "base_image": sample_base64_image,
                "target_image": sample_base64_image,
                "mode": "parts",
                "parts": ["eyes", "nose"],
                "strength": 0.7,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "queued"

    def test_create_parts_job_without_parts_fails(self, client, sample_base64_image):
        """Test that parts mode requires parts list."""
        response = client.post(
            "/api/v1/simulations/generate",
            json={
                "base_image": sample_base64_image,
                "target_image": sample_base64_image,
                "mode": "parts",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "parts" in data["error"]["message"].lower()


class TestGetJobStatus:
    """Tests for GET /api/v1/simulations/generate/{job_id}"""

    def test_get_queued_job(self, client, mock_supabase):
        """Test getting a queued job status."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{
                "id": "test-job-id",
                "mode": "morph",
                "status": "queued",
                "progress": 0,
                "strength": 0.5,
                "parts": [],
                "created_at": "2026-01-26T00:00:00+00:00",
                "started_at": None,
                "completed_at": None,
                "error": None,
                "result_image_path": None,
            }]
        )

        response = client.get("/api/v1/simulations/generate/test-job-id")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "queued"
        assert data["data"]["progress"] == 0

    def test_get_running_job(self, client, mock_supabase):
        """Test getting a running job status."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{
                "id": "test-job-id",
                "mode": "morph",
                "status": "running",
                "progress": 50,
                "strength": 0.5,
                "parts": [],
                "created_at": "2026-01-26T00:00:00+00:00",
                "started_at": "2026-01-26T00:01:00+00:00",
                "completed_at": None,
                "error": None,
                "result_image_path": None,
            }]
        )

        response = client.get("/api/v1/simulations/generate/test-job-id")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "running"
        assert data["data"]["progress"] == 50

    def test_get_succeeded_job(self, client, mock_supabase, sample_base64_image):
        """Test getting a succeeded job with result URL."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{
                "id": "test-job-id",
                "mode": "morph",
                "status": "succeeded",
                "progress": 100,
                "strength": 0.5,
                "parts": [],
                "created_at": "2026-01-26T00:00:00+00:00",
                "started_at": "2026-01-26T00:01:00+00:00",
                "completed_at": "2026-01-26T00:02:00+00:00",
                "error": None,
                "result_image_path": sample_base64_image,
            }]
        )

        response = client.get("/api/v1/simulations/generate/test-job-id")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "succeeded"
        assert data["data"]["progress"] == 100
        assert data["data"]["result_image_url"] is not None

    def test_get_failed_job(self, client, mock_supabase):
        """Test getting a failed job with error message."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{
                "id": "test-job-id",
                "mode": "morph",
                "status": "failed",
                "progress": 20,
                "strength": 0.5,
                "parts": [],
                "created_at": "2026-01-26T00:00:00+00:00",
                "started_at": "2026-01-26T00:01:00+00:00",
                "completed_at": "2026-01-26T00:02:00+00:00",
                "error": "No face detected in base image",
                "result_image_path": None,
            }]
        )

        response = client.get("/api/v1/simulations/generate/test-job-id")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "failed"
        assert data["data"]["error"] == "No face detected in base image"

    def test_get_nonexistent_job(self, client, mock_supabase):
        """Test getting a job that doesn't exist."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[]
        )

        response = client.get("/api/v1/simulations/generate/nonexistent-id")

        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "NOT_FOUND"


class TestGetJobResult:
    """Tests for GET /api/v1/simulations/generate/{job_id}/result"""

    def test_get_result_success(self, client, mock_supabase, sample_base64_image):
        """Test getting result of succeeded job."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{
                "id": "test-job-id",
                "mode": "morph",
                "status": "succeeded",
                "progress": 100,
                "strength": 0.5,
                "parts": [],
                "created_at": "2026-01-26T00:00:00+00:00",
                "started_at": "2026-01-26T00:01:00+00:00",
                "completed_at": "2026-01-26T00:02:00+00:00",
                "error": None,
                "result_image_path": sample_base64_image,
            }]
        )

        response = client.get("/api/v1/simulations/generate/test-job-id/result")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["image"] == sample_base64_image
        assert data["data"]["format"] == "png"
        assert data["data"]["mode"] == "morph"

    def test_get_result_job_not_complete(self, client, mock_supabase):
        """Test getting result of non-complete job."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{
                "id": "test-job-id",
                "mode": "morph",
                "status": "running",
                "progress": 50,
                "strength": 0.5,
                "parts": [],
                "created_at": "2026-01-26T00:00:00+00:00",
                "started_at": "2026-01-26T00:01:00+00:00",
                "completed_at": None,
                "error": None,
                "result_image_path": None,
            }]
        )

        response = client.get("/api/v1/simulations/generate/test-job-id/result")

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "JOB_FAILED"

    def test_get_result_nonexistent_job(self, client, mock_supabase):
        """Test getting result of nonexistent job."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[]
        )

        response = client.get("/api/v1/simulations/generate/nonexistent-id/result")

        assert response.status_code == 404


class TestImageValidation:
    """Tests for image validation in face detection."""

    @pytest.mark.skip(reason="Requires real face detection service")
    def test_face_not_detected_error(self, client, mock_supabase, sample_base64_image):
        """Test error when face is not detected."""
        # This would require mocking the face detection service
        # which is complex due to MediaPipe initialization
        pass
