import torch
import torch.nn as nn
import torch.optim as optim
import PIL
import cv2


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

class ID_CNN(nn.Module):
    """Returns a mapping of the multiplicative noise that is divided from the image to provide a clear image."""
    def __init__(self):
        super(ID_CNN, self).__init__()

        self.convL1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3 ,padding=1)
        self.convL2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 ,padding=1)

        self.BatchNorm = nn.BatchNorm2d()

        self.relu = torch.F.relu()

        self.convL8 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3 ,padding=1)


    def forward(self, x):
        x = self.relu(self.convL1(x)) # Layer_1

        x = self.relu(self.BatchNorm(self.convL2(x))) # Layer_2 
        x = self.relu(self.BatchNorm(self.convL2(x))) # Layer_3
        x = self.relu(self.BatchNorm(self.convL2(x))) # Layer_4
        x = self.relu(self.BatchNorm(self.convL2(x))) # Layer_5
        x = self.relu(self.BatchNorm(self.convL2(x))) # Layer_6
        x = self.relu(self.BatchNorm(self.convL2(x))) # Layer_7

        x = self.relu(self.convL8(x)) # Layer_8

        return x

def training():
    pass
