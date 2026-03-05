import torch
import torch.nn as nn
import numpy as np
import PIL.Image as Image
import cv2
import os
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ── CONFIG ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "SAR/models/best_denoising_model_Inception_reduced.pth"
NOISY_DIR    = "Dataset/SAR_despeckling_filters_Dataset/Main folder/Noisy_val"
GTRUTH_DIR   = "Dataset/SAR_despeckling_filters_Dataset/Main folder/GTruth_val"
RESIZE       = (512, 512)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


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


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_enl(img: np.ndarray) -> float:
    h, w = img.shape
    cy, cx = h // 2, w // 2
    ch, cw = max(1, h // 10), max(1, w // 10)
    patch = img[cy-ch:cy+ch, cx-cw:cx+cw].astype(np.float64)
    mu, sigma = patch.mean(), patch.std()
    return (mu / sigma) ** 2 if sigma > 0 else float('inf')


def compute_epi(noisy: np.ndarray, denoised: np.ndarray) -> float:
    sobel = lambda img: cv2.Sobel(img.astype(np.float64), cv2.CV_64F, 1, 0) + \
                        cv2.Sobel(img.astype(np.float64), cv2.CV_64F, 0, 1)
    e_n, e_d = sobel(noisy).flatten(), sobel(denoised).flatten()
    return float(np.corrcoef(e_n, e_d)[0, 1]) if np.std(e_n) * np.std(e_d) > 0 else 0.0


# ── Eval loop ─────────────────────────────────────────────────────────────────

def evaluate():
    model = ID_CNN().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    tfm = transforms.Compose([
        *([transforms.Resize(RESIZE)] if RESIZE else []),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])

    SUPPORTED = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    files = sorted(f for f in os.listdir(NOISY_DIR)
                   if os.path.splitext(f)[1].lower() in SUPPORTED)

    results = []
    print(f"{'File':<30} {'PSNR':>7} {'SSIM':>7} {'RMSE':>7} {'ENL':>8} {'EPI':>7}")
    print("-" * 70)

    for fname in files:
        noisy_path  = os.path.join(NOISY_DIR,  fname)
        gtruth_path = os.path.join(GTRUTH_DIR, fname)
        if not os.path.exists(gtruth_path):
            print(f"  [skip] no ground truth for {fname}")
            continue

        x = tfm(Image.open(noisy_path).convert('L')).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = torch.clamp(x / (model(x) + 1e-8), 0, 1)

        denoised = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)
        noisy_np = (x.squeeze().cpu().numpy()  * 255).astype(np.uint8)

        gt = Image.open(gtruth_path).convert('L')
        if RESIZE:
            gt = gt.resize((RESIZE[1], RESIZE[0]), Image.BILINEAR)
        gt_np = np.array(gt, dtype=np.uint8)

        p    = psnr(gt_np, denoised, data_range=255)
        s    = ssim(gt_np, denoised, data_range=255)
        rmse = np.sqrt(np.mean((gt_np.astype(np.float64) - denoised.astype(np.float64)) ** 2))
        enl  = compute_enl(denoised)
        epi  = compute_epi(noisy_np, denoised)

        results.append((p, s, rmse, enl, epi))
        print(f"{fname:<30} {p:>7.3f} {s:>7.4f} {rmse:>7.3f} {enl:>8.2f} {epi:>7.4f}")

    if results:
        arr = np.array(results)
        print("-" * 70)
        print(f"{'MEAN':<30} {arr[:,0].mean():>7.3f} {arr[:,1].mean():>7.4f} "
              f"{arr[:,2].mean():>7.3f} {arr[:,3].mean():>8.2f} {arr[:,4].mean():>7.4f}")


if __name__ == "__main__":
    print(f"Running on: {DEVICE}\n")
    evaluate()