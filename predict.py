import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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

model.eval()

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

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    return (
        classes[predicted.item()],
        round(confidence.item() * 100, 2)
    )