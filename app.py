from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import json
import base64

# ---- App Setup ----
app = FastAPI(
    title="Pneumonia Detection API",
    description="API de détection de pneumonie par analyse d'images radiographiques (Rayons X)",
    version="1.0.0",
    contact={"name": "Ahmed Ben Attia Khiari & Achref Ghorbel"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Model Loading ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("model_config.json", "r") as f:
    config = json.load(f)

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, config["num_classes"])
model.load_state_dict(torch.load("best_model_crop.pt", map_location=device))
model = model.to(device)
model.eval()

# ---- Transforms ----
class DynamicCenterCrop:
    def __init__(self, ratio=0.7):
        self.ratio = ratio
    def __call__(self, img):
        w, h = img.size
        new_w = int(w * self.ratio)
        new_h = int(h * self.ratio)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return img.crop((left, top, left + new_w, top + new_h))

inference_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    DynamicCenterCrop(config["center_crop_ratio"]),
    transforms.Resize((config["input_size"], config["input_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=config["normalization"]["mean"],
                         std=config["normalization"]["std"])
])

raw_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    DynamicCenterCrop(config["center_crop_ratio"]),
    transforms.Resize((config["input_size"], config["input_size"])),
])

# ---- Grad-CAM ----
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, input, output):
        self.activations = output.detach()

    def _bwd(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, output

grad_cam = GradCAM(model, model.features[-1])

def generate_gradcam_image(img: Image.Image) -> str:
    """Generate Grad-CAM overlay and return as base64 string."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    tensor = inference_transform(img).unsqueeze(0).to(device)
    heatmap, _ = grad_cam.generate(tensor)

    raw_img = np.array(raw_transform(img))

    colormap = cm.jet(heatmap)[:, :, :3]
    colormap = (colormap * 255).astype(np.uint8)

    if raw_img.ndim == 2:
        raw_img = np.stack([raw_img]*3, axis=-1)

    overlay = (0.6 * raw_img + 0.4 * colormap).astype(np.uint8)

    overlay_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ---- Routes ----
@app.get("/")
async def root():
    return {
        "message": "Pneumonia Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload X-ray image for prediction",
            "/metrics": "GET - Model performance metrics",
            "/health": "GET - API health check"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "device": str(device), "model": config["model_name"]}

@app.get("/metrics")
async def metrics():
    return {
        "model": config["model_name"],
        "metrics": config["metrics"],
        "training": config["training"]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image (JPEG, PNG)")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image")

    # Predict
    tensor = inference_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)

    pred_idx = probs.argmax(dim=1).item()
    confidence = probs[0][pred_idx].item()

    # Grad-CAM
    gradcam_b64 = generate_gradcam_image(img)

    return JSONResponse(content={
        "prediction": config["class_names"][pred_idx],
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "NORMAL": round(probs[0][0].item() * 100, 2),
            "PNEUMONIA": round(probs[0][1].item() * 100, 2)
        },
        "gradcam_image": gradcam_b64
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)