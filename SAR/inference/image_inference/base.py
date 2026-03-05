import torch
import torch.nn as nn
import numpy as np
import PIL.Image as Image
import cv2
import os
from torchvision import transforms

# ── CONFIG ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "SAR/models/idcnn_base.pth"
NOISY_DIR    = "Dataset/SAR_despeckling_filters_Dataset/Main folder/Noisy_val"
GTRUTH_DIR   = "Dataset/SAR_despeckling_filters_Dataset/Main folder/GTruth_val"
OUTPUT_DIR   = "SAR/visual_output"
RESIZE       = (512, 512)
NUM_SAMPLES  = 5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


class ID_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convL1     = nn.Conv2d(1,  64, 3, padding=1)
        self.convL2     = nn.Conv2d(64, 64, 3, padding=1)
        self.convL3     = nn.Conv2d(64, 64, 3, padding=1)
        self.convL4     = nn.Conv2d(64, 64, 3, padding=1)
        self.convL5     = nn.Conv2d(64, 64, 3, padding=1)
        self.convL6     = nn.Conv2d(64, 64, 3, padding=1)
        self.convL7     = nn.Conv2d(64, 64, 3, padding=1)
        self.convL8     = nn.Conv2d(64,  1, 3, padding=1)
        self.BatchNorm2 = nn.BatchNorm2d(64)
        self.BatchNorm3 = nn.BatchNorm2d(64)
        self.BatchNorm4 = nn.BatchNorm2d(64)
        self.BatchNorm5 = nn.BatchNorm2d(64)
        self.BatchNorm6 = nn.BatchNorm2d(64)
        self.BatchNorm7 = nn.BatchNorm2d(64)
        self.relu       = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.convL1(x))
        x = self.relu(self.BatchNorm2(self.convL2(x)))
        x = self.relu(self.BatchNorm3(self.convL3(x)))
        x = self.relu(self.BatchNorm4(self.convL4(x)))
        x = self.relu(self.BatchNorm5(self.convL5(x)))
        x = self.relu(self.BatchNorm6(self.convL6(x)))
        x = self.relu(self.BatchNorm7(self.convL7(x)))
        return torch.clamp(self.relu(self.convL8(x)), min=0.01)


def to_uint8(tensor):
    return (tensor.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def visualise():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = ID_CNN().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])

    SUPPORTED = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    files = sorted(f for f in os.listdir(NOISY_DIR)
                   if os.path.splitext(f)[1].lower() in SUPPORTED)[:NUM_SAMPLES]

    for fname in files:
        noisy_path  = os.path.join(NOISY_DIR,  fname)
        gtruth_path = os.path.join(GTRUTH_DIR, fname)

        x = tfm(Image.open(noisy_path).convert('L')).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = torch.clamp(x / (model(x) + 1e-8), 0, 1)

        noisy_np    = to_uint8(x)
        denoised_np = to_uint8(out)
        gt_np       = np.array(Image.open(gtruth_path).convert('L').resize(
                          (RESIZE[1], RESIZE[0]), Image.BILINEAR), dtype=np.uint8)

        diff = cv2.applyColorMap(
            cv2.normalize(np.abs(denoised_np.astype(np.int16) - gt_np.astype(np.int16))
                          .astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX),
            cv2.COLORMAP_HOT)

        row = cv2.hconcat([
            cv2.cvtColor(noisy_np,    cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(denoised_np, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(gt_np,       cv2.COLOR_GRAY2BGR),
            diff,
        ])

        for i, label in enumerate(["Noisy", "Denoised", "Ground Truth", "Diff"]):
            cv2.putText(row, label, (i * RESIZE[1] + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 100), 2)

        out_path = os.path.join(OUTPUT_DIR, f"vis_{os.path.splitext(fname)[0]}.png")
        cv2.imwrite(out_path, row)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    print(f"Running on: {DEVICE}\n")
    visualise()