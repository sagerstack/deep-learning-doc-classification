import io
import os
import sys
from pathlib import Path

# Force CPU for tests to avoid MPS segfaults in torch_geometric kNN
os.environ["FORCE_DEVICE"] = "cpu"

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.src.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def small_test_image():
    img = Image.new("L", (400, 500), color=180)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf
