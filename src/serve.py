from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src.model import get_model


app = FastAPI(title="MLOps PyTorch Model Serving")

MODEL_PATH = Path("checkpoints/classifier_v1.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

inference_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ]
)


def load_model():
    if not MODEL_PATH.exists():
        return None

    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint not loaded",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint not loaded",
        )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        ) from exc

    tensor = inference_transform(pil_image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "class_index": predicted_index,
        "probabilities": {
            CLASS_NAMES[i]: round(float(probabilities[i]), 6)
            for i in range(len(CLASS_NAMES))
        },
    }