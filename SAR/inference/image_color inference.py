"""
Inference — SAR Image Colorization
Just set INPUT_PATH and run: python inference.py
"""

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
import tifffile
import matplotlib.pyplot as plt

# ── Set these ─────────────────────────────────
INPUT_PATH   = "Dataset/SAR_Color_Dataset/train/denoised_sar_images/TrainArea_014.tif"
WEIGHTS_PATH = "SAR/models/Colorizer/generator_final.pt"
OUTPUT_PATH  = None   # e.g. "result.png", or None to auto-name
# ──────────────────────────────────────────────

SIZE   = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_generator(n_input=1, n_output=2, size=SIZE):
    body = create_body(resnet18(), pretrained=False, n_in=n_input, cut=-2)
    return DynamicUnet(body, n_out=n_output, img_size=(size, size),
                       self_attention=True, act_cls=nn.ReLU)


def load_sar(img_path):
    arr = tifffile.imread(img_path).astype("float32")
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    arr = arr.astype("uint8")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    img = transforms.functional.resize(Image.fromarray(arr, mode="L"), (SIZE, SIZE))
    L   = (np.array(img, dtype="float32")[:, :, np.newaxis] / 127.5) - 1.0
    return torch.from_numpy(L).permute(2, 0, 1).unsqueeze(0), arr


def lab_to_rgb(L_t, ab_t):
    L_np  = (L_t.numpy()[0] + 1.0) * 127.5 / 255.0 * 100.0
    ab_np = ab_t.numpy() * 110.0
    Lab   = np.stack([L_np, ab_np[0], ab_np[1]], axis=-1)
    return (np.clip(lab2rgb(Lab), 0, 1) * 255).astype("uint8")


def colorize(img_path, weights_path, output_path=None):
    net_G = build_generator().to(DEVICE)
    net_G.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    net_G.eval()

    L_t, sar_display = load_sar(img_path)
    with torch.no_grad():
        ab_t = net_G(L_t.to(DEVICE)).squeeze(0).cpu()

    rgb = lab_to_rgb(L_t.squeeze(0), ab_t)

    if output_path is None:
        output_path = Path(img_path).stem + "_colorized.png"
    Image.fromarray(rgb).save(output_path)
    print(f"Saved: {output_path}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(sar_display, cmap="gray"); axes[0].set_title("Input (SAR)"); axes[0].axis("off")
    axes[1].imshow(rgb);                      axes[1].set_title("Colorized");   axes[1].axis("off")
    plt.tight_layout()
    plt.show()


colorize(INPUT_PATH, WEIGHTS_PATH, OUTPUT_PATH)