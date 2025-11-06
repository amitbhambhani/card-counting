import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 37 * 37, 512)
        self.fc2 = nn.Linear(512, 52)  # 52 output classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 37 * 37)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # no sigmoid or softmax we need the number logits for CrossEntropyLoss
        return x
# logit is a raw prediction number from the model, when there are 52 possible classes,
# you get 52 logits, one for each class. The higher the logit, the more the model thinks
# that class is the correct one. These logits are then converted to probabilities 
# using softmax internally in CrossEntropyLoss.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#uses nvidia cuda to move processing to the gpu
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)