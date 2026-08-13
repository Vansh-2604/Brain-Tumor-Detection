import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
import os

print("PREDICT.PY IMPORTED", flush=True)

device = torch.device("cpu")

# Class names
classes = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor Detected",
    "Pituitary Tumor"
]

# Create model
model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    4
)

# Load trained weights
model.load_state_dict(
    torch.load(
        "model/brain_tumor_model.pth",
        map_location=device
    )
)
print("MODEL WEIGHTS LOADED", flush=True)

model.eval()
print("MODEL READY", flush=True)

# Same transform used during testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_path):
    
    start = time.time()
    print("ENTERED predict_image", flush=True)
    print("Image path:", image_path, flush=True)

    print("File exists:", os.path.exists(image_path), flush=True)
    if os.path.exists(image_path):
        print("File size:", os.path.getsize(image_path), "bytes", flush=True)

    print("BEFORE PIL OPEN", flush=True)

    image = Image.open(image_path).convert("RGB")
    print("RGB CONVERSION SUCCESS", flush=True)

    image = transform(image)
    print("Transform done:", time.time() - start, flush=True)

    image = image.unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        print("Model inference done:", time.time() - start, flush=True) 

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )
        print("Prediction finished:", time.time() - start, flush=True)

    return (
        classes[predicted.item()],
        round(confidence.item() * 100, 2)
    )