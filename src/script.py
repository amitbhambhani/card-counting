import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

#Model from ../card.ipynb
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
    # Resize each image to 256x256
    transforms.Resize((256, 256)),
    # Create a matrix representation of the image
    transforms.ToTensor(),
])

# Takes a path to an image as a parameter. Returns the model output given the image. 
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
    image = Image.open(image_path).convert("RGB")  #convert image to rgb scale and transform it, so it the model can use it
    img_tensor = transform(image).unsqueeze(0)  # add batch dimension

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU or CPU

    model = SimpleCNN(num_classes=52) # initalizes model 
    model.load_state_dict(torch.load(model_path, map_location=device)) # Load training from model_path into device and the model
    model.to(device) # Move model into the device
    model.eval() # Set model to evaluation mode

    # Predict
    with torch.no_grad(): # doesn't allow gradient calculation
        output = model(img_tensor.to(device)) #model predictions
        _, predicted = output.max(1) #class that the model predicts

    # Convert class index to folder name (01–52)
    label_num = predicted.item() + 1

    return label_num


# ===================== REFERENCES ====================#
# Code/logic derived from references will be marked with (Reference #)
#
# 1. ChatGPT (OpenAI)