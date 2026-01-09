from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os

class_names = [
    'Front_Breakage',
    'Front_Crushed',
    'Front_Normal',
    'Rear_Breakage',
    'Rear_Crushed',
    'Rear_Normal'
]

class CarClassifierResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet50(weights='DEFAULT')

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace FC layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, 6)
        )

    def forward(self, x):
        return self.model(x)


train_model = None

def predict(image_path):
    global train_model

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0)

    if train_model is None:
        train_model = CarClassifierResNet()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "model", "saved_model.pth")
        train_model.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        )
        train_model.eval()

    with torch.no_grad():
        output = train_model(image_tensor)
        _, predicted_class = torch.max(output, 1)

    return class_names[predicted_class.item()]
