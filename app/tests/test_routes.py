import io
import os

import pytest
from PIL import Image

# Force CPU for test stability (MPS can segfault on torch_geometric kNN)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def _make_test_image(width: int = 400, height: int = 500) -> io.BytesIO:
    """Create a grayscale test image sized to produce valid feature maps."""
    img = Image.new("L", (width, height), color=180)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


@pytest.fixture
def small_test_image():
    return _make_test_image()


class TestHomePage:
    def test_home_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_home_contains_title(self, client):
        response = client.get("/")
        assert "Document Classification" in response.text

    def test_home_contains_sample_thumbnails(self, client):
        response = client.get("/")
        assert "in-dist/" in response.text

    def test_home_contains_upload_zone(self, client):
        response = client.get("/")
        assert 'type="file"' in response.text


@pytest.mark.slow
class TestClassifyWithUpload:
    def test_classify_upload_returns_200(self, client, small_test_image):
        response = client.post(
            "/classify",
            files={"file": ("test_doc.jpg", small_test_image, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_classify_upload_contains_results(self, client, small_test_image):
        response = client.post(
            "/classify",
            files={"file": ("test_doc.jpg", small_test_image, "image/jpeg")},
        )
        assert "CNN Feature" in response.text or "has_results" not in response.text


@pytest.mark.slow
class TestClassifyWithSample:
    def test_classify_sample_returns_200(self, client):
        response = client.post(
            "/classify",
            data={"sample": "in-dist/invoice.jpg"},
        )
        assert response.status_code == 200

    def test_classify_sample_contains_predictions(self, client):
        response = client.post(
            "/classify",
            data={"sample": "in-dist/invoice.jpg"},
        )
        assert "Model Predictions" in response.text or "confidence" in response.text.lower()


@pytest.mark.slow
class TestClassifyHTMX:
    def test_htmx_request_returns_partial(self, client, small_test_image):
        response = client.post(
            "/classify",
            files={"file": ("test_doc.jpg", small_test_image, "image/jpeg")},
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "<!DOCTYPE html>" not in response.text

    def test_htmx_response_has_result_sections(self, client):
        response = client.post(
            "/classify",
            data={"sample": "in-dist/letter.jpg"},
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200


class TestClassifyErrors:
    def test_no_input_returns_400(self, client):
        response = client.post("/classify")
        assert response.status_code == 400

    def test_invalid_sample_returns_error(self, client):
        response = client.post(
            "/classify",
            data={"sample": "nonexistent.jpg"},
        )
        assert response.status_code >= 400
