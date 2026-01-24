"""Tests for health check endpoint."""

import pytest


class TestHealthCheck:
    """Test cases for GET /health endpoint."""

    def test_health_returns_ok_status(self, client):
        """Health check should return success with ok status."""
        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "ok"

    def test_health_returns_version(self, client):
        """Health check should include API version."""
        response = client.get("/health")

        data = response.json()
        assert "version" in data["data"]
        assert data["data"]["version"]  # Not empty

    def test_health_returns_timestamp(self, client):
        """Health check should include timestamp."""
        response = client.get("/health")

        data = response.json()
        assert "timestamp" in data["data"]
        assert data["data"]["timestamp"]  # Not empty

    def test_health_response_format(self, client):
        """Health check should follow standard response format."""
        response = client.get("/health")

        data = response.json()

        # Check top-level structure
        assert "success" in data
        assert "data" in data

        # Check data structure
        assert "status" in data["data"]
        assert "version" in data["data"]
        assert "timestamp" in data["data"]


class TestRootEndpoint:
    """Test cases for GET / endpoint."""

    def test_root_returns_api_info(self, client):
        """Root endpoint should return API information."""
        response = client.get("/")

        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "Cao API"
