import torch
import torch.nn as nn
import torch.optim as optim
import PIL.Image as Image
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


        self.clean_image = sorted([f for f in os.listdir(self.gtruth_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.noisy_image = sorted([f for f in os.listdir(self.noisy_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

        
        assert len(self.clean_image) == len(self.noisy_image), "The data sizes dont match"

    def __len__(self):
        return len(self.noisy_image)

    def __getitem__(self, idx):
        noisy_path = f"{self.noisy_dir}/{self.noisy_image[idx]}"
        clean_path = f"{self.gtruth_dir}/{self.clean_image[idx]}"
        
        noisy_img = Image.open(noisy_path).convert('L')
        clean_img = Image.open(clean_path).convert('L')


        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        
        return noisy_img, clean_img

class Inception(nn.Module): # will have to complete
    def __init__(self, in_channels, out_channels):
        super(Inception, self).__init__()

        branch_channels = out_channels//3

        self.kernel_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU()
        )

        self.kernel_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=branch_channels, out_channels=branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU()
        )
        
        self.pooling = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU()
        )

    def forward(self, x):
        branch_1 = self.kernel_1x1(x)
        branch_2 = self.kernel_3x3(x)
        branch_3 = self.pooling(x)

        return torch.cat([branch_1, branch_2, branch_3], dim=1)

class ID_CNN(nn.Module):
    """Returns a mapping of the multiplicative noise that is divided from the image to provide a clear image."""
    def __init__(self):
        super(ID_CNN, self).__init__()

        self.convL1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3 ,padding=1)

        self.inception1 = Inception(64, 64)
        self.inception2 = Inception(64, 64)
        self.inception3 = Inception(64, 64)

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

        # maybe add in googles inception modules to this so that pixel-localized, region-localized, and section localized are obtained.

    def forward(self, x):

        x = self.relu(self.convL1(x)) # Layer_1
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.relu(self.BatchNorm2(self.convL2(x))) # Layer_2 
        x = self.relu(self.BatchNorm3(self.convL3(x))) # Layer_3
        x = self.relu(self.BatchNorm4(self.convL4(x))) # Layer_4
        x = self.relu(self.BatchNorm5(self.convL5(x))) # Layer_5
        x = self.relu(self.BatchNorm6(self.convL6(x))) # Layer_6
        x = self.relu(self.BatchNorm7(self.convL7(x))) # Layer_7

        mask = self.relu(self.convL8(x)) # Layer_8
        mask = torch.clamp(mask, min=0.01)
        return mask


def training(epochs, train_dataset, val_dataset, model, device='cuda'):

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min', patience=5)
    best_val_loss = float('inf')


    for epoch in tqdm.tqdm(range(epochs)):
        model.train()
        train_loss = 0.0

        for noisy, clean in train_dataset:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            y_pred = model(noisy)
            output = noisy/(y_pred + 1e-8)
            loss = euclidean_TV_loss(y_pred=output, y_ground=clean)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_dataset)

        
        # Validation code
        if (epoch+1)%2 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for noisy, clean in val_dataset:
                    noisy, clean = noisy.to(device), clean.to(device)

                    y_pred = model(noisy)
                    output = noisy/(y_pred + 1e-8)
                    loss = euclidean_TV_loss(y_pred=output, y_ground=clean)
                
                    val_loss += loss.item()

            val_loss /= len(val_dataset)
            scheduler.step(val_loss) # keeps track of val loss progression and changes lr based on that
            
            print(f'Epoch [{epoch+1}/{epochs}] '
                f'Train Loss: {train_loss:.4f} '
                f'Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'SAR/models/best_denoising_model.pth')
                print(f'Model saved with val loss: {val_loss:.4f}')
            

if __name__ == "__main__":
    EPOCHS = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")

    train_noisy_dir = "Dataset/SAR_despeckling_filters_Dataset/Main folder/Noisy"
    train_gtruth_dir = "Dataset/SAR_despeckling_filters_Dataset/Main folder/GTruth"
    val_noisy_dir = "Dataset/SAR_despeckling_filters_Dataset/Main folder/Noisy_val"
    val_gtruth_dir = "Dataset/SAR_despeckling_filters_Dataset/Main folder/GTruth_val"

    img_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Grayscale(1),
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


    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=8, shuffle=False)

    print("Dataset loaded successfully... ")

    model_idcnn = ID_CNN()
    
    print("Starting training... ")


    training(
        epochs=EPOCHS,
        train_dataset=train_loader, 
        val_dataset=val_loader, 
        model=model_idcnn, 
        device=device
    )

    print("Completed Training...")


# if weight decay is present, use AdamW