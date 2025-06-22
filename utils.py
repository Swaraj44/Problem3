# utils.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

def load_model(path="mnist_cnn.pth", device="cpu"):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def get_digit_images(digit, count=5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    filtered = [img for img, label in dataset if label == digit]
    images = random.sample(filtered, count)

    return images
