"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

# Register pytest-asyncio plugin
pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: marks tests as slow")


@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio as the async backend."""
    return "asyncio"


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def api_base_url():
    """Base URL for API endpoints."""
    return "/api/v1"
