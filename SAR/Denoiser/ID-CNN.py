import torch
import torch.nn as nn
import torch.optim as optim
import PIL
import cv2
import tqdm as tqdm
from torchvision import transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from helpers import euclidean_TV_loss



class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir_path, gtruth_dir_path, img_transforms):

        self.noisy_dir = noisy_dir_path
        self.gtruth_dir = gtruth_dir_path
        self.transform = img_transforms


        self.clean_image = sorted([f for f in os.listdir(self.gtruth_dir) if f.endwith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.noisy_image = sorted([f for f in os.listdir(self.noisy_dir) if f.endwith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

        
        assert len(self.clean_image) == len(noisy_image), "The data sizes dont match"

    def __len__(self):
        return len(noisy_image)

    def __getitem__(self, idx):
        noisy_path = self.noisy_dir / self.noisy_image[idx]
        clean_path = self.gtruth_dir / self.clean_image[idx]
        
        noisy_img = Image.open(noisy_path)
        clean_img = Image.open(clean_path)


        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        
        return noisy_img, clean_img



class ID_CNN(nn.Module):
    """Returns a mapping of the multiplicative noise that is divided from the image to provide a clear image."""
    def __init__(self):
        super(ID_CNN, self).__init__()

        self.convL1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3 ,padding=1)
        self.convL2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 ,padding=1)
        self.convL3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 ,padding=1)
        self.convL4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 ,padding=1)
        self.convL5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 ,padding=1)
        self.convL6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 ,padding=1)
        self.convL7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 ,padding=1)
        self.convL8 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3 ,padding=1)

        self.BatchNorm2 = nn.BatchNorm2d(64)
        self.BatchNorm3 = nn.BatchNorm2d(64)
        self.BatchNorm4 = nn.BatchNorm2d(64)
        self.BatchNorm5 = nn.BatchNorm2d(64)
        self.BatchNorm6 = nn.BatchNorm2d(64)
        self.BatchNorm7 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.convL1(x)) # Layer_1

        x = self.relu(self.BatchNorm2(self.convL2(x))) # Layer_2 
        x = self.relu(self.BatchNorm3(self.convL3(x))) # Layer_3
        x = self.relu(self.BatchNorm4(self.convL4(x))) # Layer_4
        x = self.relu(self.BatchNorm5(self.convL5(x))) # Layer_5
        x = self.relu(self.BatchNorm6(self.convL6(x))) # Layer_6
        x = self.relu(self.BatchNorm7(self.convL7(x))) # Layer_7

        mask = self.relu(self.convL8(x)) # Layer_8
        mask = torch.clamp(mask, min=0.01)
        return mask


def training(epochs, optimizer, train_dataset, val_dataset, model, device='cuda'):
    model
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in tqdm.tqdm(range(epochs)):
        model.train()
        


if __name__ == "__main__":
    EPOCHS = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")

    train_noisy_dir = "Dataset/SAR despeckling filters dataset/Main folder/Noisy"
    train_gtruth_dir = "Dataset/SAR despeckling filters dataset/Main folder/GTruth"
    val_noisy_dir = "Dataset/SAR despeckling filters dataset/Main folder/Noisy_val"
    val_gtruth_dir = "Dataset/SAR despeckling filters dataset/Main folder/GTruth_val"

    img_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.Normalize(),
        transforms.ToTensor()
    ])

    train_dataset = DenoisingDataset(
        noisy_dir_path=train_noisy_dir,
        gtruth_dir_path=train_gtruth_dir,
        img_transforms=img_transforms
    )
    
    val_dataset = DenoisingDataset(
        noisy_dir_path=val_noisy_dir,
        gtruth_dir_path=val_gtruth_dir,
        img_transforms=img_transforms
    )


    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=8, num_workers=4, shuffle=False)

    model_idcnn = ID_CNN()

    training(
        epochs=EPOCHS,
        train_dataset=train_loader, 
        val_dataset=val_loader, 
        model=model_idcnn, 
        device=device
    )



