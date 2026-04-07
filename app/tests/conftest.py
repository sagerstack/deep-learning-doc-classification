import pytest
from fastapi.testclient import TestClient

from app.src.main import app


@pytest.fixture
def client():
    return TestClient(app)
