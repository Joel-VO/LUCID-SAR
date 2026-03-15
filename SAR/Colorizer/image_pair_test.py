"""
Pair Checker — visually verify SAR ↔ colour image alignment
Usage: python check_pairs.py
"""

import glob
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
COLOR_DIR = Path("Dataset/SAR_Color_Dataset/train/rgb_images")
SAR_DIR   = Path("Dataset/SAR_Color_Dataset/train/denoised_sar_images")
N_PREVIEW = 6   # number of pairs to show
# ────────────────────────────────────────────────────────────────────────────


def to_uint8(arr):
    arr = arr.astype("float32")
    return ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype("uint8")


def load_for_display(path):
    arr = to_uint8(tifffile.imread(path))
    if arr.ndim == 2:
        return arr, "gray"
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr, None


def main():
    color_paths = sorted(glob.glob(str(COLOR_DIR / "*.tif")) +
                         glob.glob(str(COLOR_DIR / "*.tiff")))
    sar_paths   = sorted(glob.glob(str(SAR_DIR   / "*.tif")) +
                         glob.glob(str(SAR_DIR   / "*.tiff")))

    print(f"Found {len(sar_paths)} SAR images, {len(color_paths)} colour images.\n")

    if len(sar_paths) != len(color_paths):
        print("WARNING: counts don't match — check folder contents.")

    # Print all pairs
    print(f"{'SAR filename':<40}  {'Colour filename'}")
    print("-" * 80)
    for s, c in zip(sar_paths, color_paths):
        match = "✓" if Path(s).stem == Path(c).stem else "✗ MISMATCH"
        print(f"{Path(s).name:<40}  {Path(c).name}  {match}")

    # Visual preview
    n = min(N_PREVIEW, len(sar_paths))
    fig, axes = plt.subplots(n, 2, figsize=(7, n * 3.5))
    if n == 1:
        axes = [axes]

    for i in range(n):
        sar_arr,   sar_cmap   = load_for_display(sar_paths[i])
        color_arr, color_cmap = load_for_display(color_paths[i])

        # SAR: show first band as grayscale
        if sar_arr.ndim == 3:
            sar_arr = sar_arr[:, :, 0]

        axes[i][0].imshow(sar_arr, cmap="gray")
        axes[i][0].set_title(f"SAR\n{Path(sar_paths[i]).name}", fontsize=8)
        axes[i][0].axis("off")

        axes[i][1].imshow(color_arr, cmap=color_cmap)
        axes[i][1].set_title(f"Colour\n{Path(color_paths[i]).name}", fontsize=8)
        axes[i][1].axis("off")

    plt.suptitle("Pair Check — SAR (left) should match Colour (right)", fontsize=10)
    plt.tight_layout()
    plt.savefig("pair_check.png", dpi=120)
    plt.show()
    print("\nSaved: pair_check.png")


if __name__ == "__main__":
    main()