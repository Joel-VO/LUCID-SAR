import os
import numpy as np
from PIL import Image
from pathlib import Path
 
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from skimage.color import lab2rgb
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Config  ← edit these
# ─────────────────────────────────────────────
INPUT_PATH          = "Dataset/SAR_despeckling_filters_Dataset/Main folder/Noisy_val/5120_2560.tiff"
DESPECKLE_WEIGHTS   = "SAR/models/denoiser/idcnn_inception_reduced.pth"
COLORIZER_WEIGHTS   = "SAR/models/Colorizer/generator_final.pt"
OUTPUT_PATH         = None        # None = auto-name next to input file
DESPECKLE_RESIZE    = (512, 512)  # resize before despeckling; None = keep original
COLORIZER_SIZE      = 256
SHOW                = True
# ─────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Despeckler — ID-CNN
# ─────────────────────────────────────────────
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
 
 
def despeckle(img_path, weights_path, resize, device):
    model = ID_CNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
 
    tfm_list = []
    if resize:
        tfm_list.append(transforms.Resize(resize))
    tfm_list += [transforms.Grayscale(1), transforms.ToTensor()]
    tfm = transforms.Compose(tfm_list)
 
    img = Image.open(img_path).convert('L')
    original_size = img.size   # (W, H)
    x = tfm(img).unsqueeze(0).to(device)
 
    with torch.no_grad():
        out = torch.clamp(x / (model(x) + 1e-8), 0, 1)
 
    denoised = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)
    result   = Image.fromarray(denoised, mode='L')
    if resize:
        result = result.resize(original_size, Image.BILINEAR)
    return result
 
 
# ─────────────────────────────────────────────
# Colorizer — ResNet18 U-Net
# ─────────────────────────────────────────────
def build_generator(size, device):
    body  = create_body(resnet18(), pretrained=False, n_in=1, cut=-2)
    net_G = DynamicUnet(body, n_out=2, img_size=(size, size),
                        self_attention=True, act_cls=nn.ReLU)
    return net_G.to(device)
 
 
def lab_to_rgb(L, ab):
    """L (1,H,W) [-1,1], ab (2,H,W) [-1,1] → RGB uint8 (H,W,3)."""
    L_lab = (L.numpy()[0] + 1.0) * 127.5 / 255.0 * 100.0
    ab_np = ab.numpy() * 110.0
    Lab   = np.stack([L_lab, ab_np[0], ab_np[1]], axis=-1)
    return (np.clip(lab2rgb(Lab), 0, 1) * 255).astype("uint8")
 
 
def colorize(denoised_pil, weights_path, size, device):
    net_G = build_generator(size, device)
    net_G.load_state_dict(torch.load(weights_path, map_location=device))
    net_G.eval()
 
    original_size = denoised_pil.size   # (W, H)
    sar_resized   = transforms.functional.resize(denoised_pil, (size, size))
 
    L   = (np.array(sar_resized, dtype="float32")[:, :, np.newaxis] / 127.5) - 1.0
    L_t = torch.from_numpy(L).permute(2, 0, 1).unsqueeze(0).to(device)
 
    with torch.no_grad():
        ab_t = net_G(L_t).squeeze(0).cpu()
 
    rgb = lab_to_rgb(L_t.squeeze(0).cpu(), ab_t)
    return Image.fromarray(rgb).resize(original_size, Image.BILINEAR)
 
 
# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────
def run_pipeline(input_path, despeckle_weights, colorizer_weights,
                 output_path=None, despeckle_resize=(512, 512),
                 colorizer_size=256):
 
    print(f"Input:      {input_path}")
    print(f"Running on: {DEVICE}\n")
 
    print("Step 1/2: Despeckling...")
    denoised = despeckle(input_path, despeckle_weights, despeckle_resize, DEVICE)
 
    print("Step 2/2: Colorizing...")
    colorized = colorize(denoised, colorizer_weights, colorizer_size, DEVICE)
 

    stem = Path(input_path).stem
    if output_path is None:
        output_path = stem + "_colorized.png"
    denoised_path   = stem + "_denoised.png"
    comparison_path = stem + "_comparison.png"
 
    # Save denoised and colorized
    denoised.save(denoised_path)
    colorized.save(output_path)
    print(f"Saved denoised:   {denoised_path}")
    print(f"Saved colorized:  {output_path}")
 
    # Build and save comparison plot (no display)
    raw_pil = Image.open(input_path).convert('L')
    raw     = np.array(raw_pil, dtype="float32")
    raw     = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-8) * 255).astype("uint8")
 
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(raw,       cmap="gray"); axes[0].set_title("Raw SAR")
    axes[1].imshow(denoised,  cmap="gray"); axes[1].set_title("Despeckled")
    axes[2].imshow(colorized);              axes[2].set_title("Colorized")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=120)
    plt.close()
    print(f"Saved comparison: {comparison_path}")
 
    return colorized, denoised, comparison_path # take this for frontend
 
 
if __name__ == "__main__":
    run_pipeline(
        input_path        = INPUT_PATH,
        despeckle_weights = DESPECKLE_WEIGHTS,
        colorizer_weights = COLORIZER_WEIGHTS,
        output_path       = OUTPUT_PATH,
        despeckle_resize  = DESPECKLE_RESIZE,
        colorizer_size    = COLORIZER_SIZE,
    )
