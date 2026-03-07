import torch
import torch.nn as nn
import numpy as np
import PIL.Image as Image
import cv2
import os
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_msssim import ssim as pt_ssim

# ── CONFIG ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH  = "SAR/models/model_base/idcnn_base.pth"
NOISY_DIR     = "Dataset/SAR_despeckling_filters_Dataset/Main folder/Noisy_val"
GTRUTH_DIR    = "Dataset/SAR_despeckling_filters_Dataset/Main folder/GTruth_val"
RESIZE        = (512, 512)   # set None to keep original size
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


class ID_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convL1 = nn.Conv2d(1,  64, 3, padding=1)
        self.convL2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convL3 = nn.Conv2d(64, 64, 3, padding=1)
        self.convL4 = nn.Conv2d(64, 64, 3, padding=1)
        self.convL5 = nn.Conv2d(64, 64, 3, padding=1)
        self.convL6 = nn.Conv2d(64, 64, 3, padding=1)
        self.convL7 = nn.Conv2d(64, 64, 3, padding=1)
        self.convL8 = nn.Conv2d(64,  1, 3, padding=1)
        self.BatchNorm2 = nn.BatchNorm2d(64)
        self.BatchNorm3 = nn.BatchNorm2d(64)
        self.BatchNorm4 = nn.BatchNorm2d(64)
        self.BatchNorm5 = nn.BatchNorm2d(64)
        self.BatchNorm6 = nn.BatchNorm2d(64)
        self.BatchNorm7 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.convL1(x))
        x = self.relu(self.BatchNorm2(self.convL2(x)))
        x = self.relu(self.BatchNorm3(self.convL3(x)))
        x = self.relu(self.BatchNorm4(self.convL4(x)))
        x = self.relu(self.BatchNorm5(self.convL5(x)))
        x = self.relu(self.BatchNorm6(self.convL6(x)))
        x = self.relu(self.BatchNorm7(self.convL7(x)))
        return torch.clamp(self.relu(self.convL8(x)), min=0.01)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_enl(img: np.ndarray) -> float:
    """ENL on a flat central crop (10% of image)."""
    h, w = img.shape
    cy, cx = h // 2, w // 2
    ch, cw = max(1, h // 10), max(1, w // 10)
    patch = img[cy-ch:cy+ch, cx-cw:cx+cw].astype(np.float64)
    mu, sigma = patch.mean(), patch.std()
    return (mu / sigma) ** 2 if sigma > 0 else float('inf')


def compute_epi(noisy: np.ndarray, denoised: np.ndarray) -> float:
    """Edge Preservation Index: correlation of Sobel edge maps."""
    sobel = lambda img: cv2.Sobel(img.astype(np.float64), cv2.CV_64F, 1, 0) + \
                        cv2.Sobel(img.astype(np.float64), cv2.CV_64F, 0, 1)
    e_n, e_d = sobel(noisy).flatten(), sobel(denoised).flatten()
    denom = np.std(e_n) * np.std(e_d)
    return float(np.corrcoef(e_n, e_d)[0, 1]) if denom > 0 else 0.0

def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_t   = torch.tensor(gt   / 255.0).float().unsqueeze(0).unsqueeze(0)
    pred_t = torch.tensor(pred / 255.0).float().unsqueeze(0).unsqueeze(0)
    return pt_ssim(gt_t, pred_t, data_range=1.0, size_average=True).item()


# ── Main eval loop ────────────────────────────────────────────────────────────

def evaluate():
    model = ID_CNN().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    tfm = transforms.Compose([
        *([ transforms.Resize(RESIZE)] if RESIZE else []),
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

        # Inference
        x = tfm(Image.open(noisy_path).convert('L')).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = torch.clamp(x / (model(x) + 1e-8), 0, 1)

        denoised = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)
        noisy_np = (x.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # Load ground truth at same size
        gt = Image.open(gtruth_path).convert('L')
        if RESIZE:
            gt = gt.resize((RESIZE[1], RESIZE[0]), Image.BILINEAR)
        gt_np = np.array(gt, dtype=np.uint8)

        # Compute metrics
        p    = psnr(gt_np, denoised, data_range=255)
        s    = compute_ssim(gt_np, denoised)
        rmse = np.sqrt(np.mean((gt_np.astype(np.float64) - denoised.astype(np.float64))**2))
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
