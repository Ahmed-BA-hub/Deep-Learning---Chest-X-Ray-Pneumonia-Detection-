import pytest
from fastapi.testclient import TestClient
from app import app
from PIL import Image
import io
import json
import os

client = TestClient(app)

# ---- Test Health ----
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert data["model"] == "EfficientNet-B0"

# ---- Test Root ----
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data

# ---- Test Metrics ----
def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert data["metrics"]["accuracy"] > 0.90
    assert data["metrics"]["auc_roc"] > 0.90

# ---- Test Predict with valid image ----
def test_predict_valid_image():
    img = Image.new("RGB", (224, 224), color="gray")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post("/predict", files={"file": ("test.jpg", buffer, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["NORMAL", "PNEUMONIA"]
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 100
    assert "probabilities" in data
    assert "gradcam_image" in data

# ---- Test Predict with different sizes ----
def test_predict_different_sizes():
    for size in [(100, 100), (500, 500), (1024, 768)]:
        img = Image.new("RGB", size, color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        response = client.post("/predict", files={"file": ("test.jpg", buffer, "image/jpeg")})
        assert response.status_code == 200

# ---- Test Predict with PNG ----
def test_predict_png():
    img = Image.new("RGB", (224, 224), color="gray")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    response = client.post("/predict", files={"file": ("test.png", buffer, "image/png")})
    assert response.status_code == 200

# ---- Test Config Exists ----
def test_config_exists():
    assert os.path.exists("model_config.json")
    with open("model_config.json", "r") as f:
        config = json.load(f)
    assert config["model_name"] == "EfficientNet-B0"
    assert config["num_classes"] == 2
    assert config["center_crop_ratio"] == 0.7

# ---- Test Model File Exists ----
def test_model_exists():
    assert os.path.exists("best_model_crop.pt")

# ---- Test Gradcam is valid base64 ----
def test_gradcam_base64():
    img = Image.new("RGB", (224, 224), color="gray")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    response = client.post("/predict", files={"file": ("test.jpg", buffer, "image/jpeg")})
    data = response.json()
    import base64
    decoded = base64.b64decode(data["gradcam_image"])
    assert len(decoded) > 0
    gradcam_img = Image.open(io.BytesIO(decoded))
    assert gradcam_img.size[0] > 0