import torch
import torch.nn as nn
import numpy as np
import PIL.Image as Image
import os
from torchvision import transforms

# ── CONFIG ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "SAR/models/model_base/idcnn_inception_reduced.pth"
INPUT_DIR    = "Dataset/SAR_Color_Dataset/train/sar_images"
OUTPUT_DIR   = "Dataset/SAR_Color_Dataset/train/denoised_sar_images"
RESIZE       = (512, 512)   # Set to None to keep original size
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

print(f"running on {DEVICE}")

class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        bc1 = out_channels // 3
        bc2 = out_channels // 3
        bc3 = out_channels - (bc1 + bc2)
        self.kernel_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, bc1, 1), nn.BatchNorm2d(bc1), nn.ReLU())
        self.kernel_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, bc2, 1), nn.BatchNorm2d(bc2), nn.ReLU(),
            nn.Conv2d(bc2, bc2, 3, padding=1), nn.BatchNorm2d(bc2), nn.ReLU())
        self.pooling = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, bc3, 1), nn.BatchNorm2d(bc3), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.kernel_1x1(x), self.kernel_3x3(x), self.pooling(x)], dim=1)


class ID_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convL1     = nn.Conv2d(1,  64, 3, padding=1)
        self.inception1 = Inception(64, 64)
        self.inception2 = Inception(64, 64)
        self.inception3 = Inception(64, 64)
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
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.relu(self.BatchNorm2(self.convL2(x)))
        x = self.relu(self.BatchNorm3(self.convL3(x)))
        x = self.relu(self.BatchNorm4(self.convL4(x)))
        x = self.relu(self.BatchNorm5(self.convL5(x)))
        x = self.relu(self.BatchNorm6(self.convL6(x)))
        x = self.relu(self.BatchNorm7(self.convL7(x)))
        return torch.clamp(self.relu(self.convL8(x)), min=0.01)


def run_inference(
    input_dir:   str = INPUT_DIR,
    output_dir:  str = OUTPUT_DIR,
    weights_path: str = WEIGHTS_PATH,
    resize:      tuple = RESIZE,
    device:      str = DEVICE,
):
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = ID_CNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Model loaded from: {weights_path}")
    print(f"Running on: {device}\n")

    tfm_list = []
    if resize:
        tfm_list.append(transforms.Resize(resize))
    tfm_list += [transforms.Grayscale(1), transforms.ToTensor()]
    tfm = transforms.Compose(tfm_list)

    SUPPORTED = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    files = sorted(f for f in os.listdir(input_dir)
                   if os.path.splitext(f)[1].lower() in SUPPORTED)

    if not files:
        print(f"No supported images found in: {input_dir}")
        return

    print(f"Found {len(files)} image(s). Processing...\n")

    for i, fname in enumerate(files, 1):
        input_path  = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        # Load & preprocess
        img = Image.open(input_path).convert('L')
        original_size = img.size  # (W, H)
        x = tfm(img).unsqueeze(0).to(device)

        # Inference: model predicts noise/speckle map; divide to despeckle
        with torch.no_grad():
            out = torch.clamp(x / (model(x) + 1e-8), 0, 1)

        # Convert to uint8
        denoised_np = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)
        result_img  = Image.fromarray(denoised_np, mode='L')

        # Restore original size if resize was applied
        if resize:
            result_img = result_img.resize(original_size, Image.BILINEAR)

        result_img.save(output_path)
        print(f"[{i}/{len(files)}] {fname} -> {output_path}")

    print(f"\nDone. Despeckled images saved to: {output_dir}")


if __name__ == "__main__":
    run_inference()