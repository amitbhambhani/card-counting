import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=52):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 128x128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 64x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 32x32

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 16x16
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict(image_path):
    model_path="src/card_model.pth"

    # Used ChatGPT to get the command that checks if a path is valid using the os library (Reference #1)
    if not os.path.exists(image_path):
        print(f"Error: image not found: {image_path}")
        return

    # Used ChatGPT to get the command that checks if a path is valid using the os library (Reference #1)
    if not os.path.exists(model_path):
        print(f"Error: model not found: {model_path}")
        return

    # Load image
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(num_classes=52)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        output = model(img_tensor.to(device))
        _, predicted = output.max(1)

    # Convert class index to folder name (01–52)
    label_num = predicted.item() + 1

    return label_num


# ===================== REFERENCES ====================#
# Code/logic derived from references will be marked with (Reference #)
#
# 1. ChatGPT (OpenAI)