import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from fastai.vision.all import *
from pathlib import Path
import random


# ── Config ────────────────────────────────────────────────────────────────────
SAR_PATH   = Path("Dataset/SAR_Color_Dataset/test/sar_images")
OPT_PATH   = Path("Dataset/SAR_Color_Dataset/test/rgb_images")
MODEL_PATH = Path("models/sar_colorizer_unet_resnet34.pth")
NUM_SAMPLES = 6          # images to compare
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Rebuild the exact same DataBlock / learner you trained with ───────────────
def get_optical(fn): return OPT_PATH / fn.name

sar_colorizer = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), ImageBlock),
    get_items=get_image_files,
    get_y=get_optical,
    splitter=RandomSplitter(0.1),
    item_tfms=Resize(512),
    batch_tfms=Normalize.from_stats([0.5], [0.5])
)
dls = sar_colorizer.dataloaders(SAR_PATH, bs=1, num_workers=0)


# ── Re-create the model architecture (must match training) ────────────────────
from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn

class SARWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)

encoder = resnet34(weights=ResNet34_Weights.DEFAULT)
encoder = nn.Sequential(*list(encoder.children())[:-2])
unet    = DynamicUnet(encoder, n_out=3, img_size=(512, 512), norm_type=None)
model   = SARWrapper(unet)


# ── Load weights & attach to learner ─────────────────────────────────────────
learn = Learner(dls, model, loss_func=L1LossFlat())
learn.load(MODEL_PATH.stem)          # fastai strips the .pth automatically
learn.model.eval()


# ── Helper: denormalise a tensor → numpy HWC uint8 ───────────────────────────
def to_img(tensor, grayscale=False):
    """Convert a normalised tensor (C,H,W) → uint8 numpy array."""
    t = tensor.detach().cpu().float()
    t = (t * 0.5 + 0.5).clamp(0, 1)          # undo Normalize([0.5],[0.5])
    t = t.permute(1, 2, 0).numpy()
    if grayscale:
        t = t[:, :, 0]                         # drop channel dim for grey
    return (t * 255).astype(np.uint8)


# ── Pick random test images ───────────────────────────────────────────────────
all_files = get_image_files(SAR_PATH)
samples   = random.sample(list(all_files), min(NUM_SAMPLES, len(all_files)))


# ── Run inference & collect results ──────────────────────────────────────────
results = []
with torch.no_grad():
    for sar_fp in samples:
        # --- load & preprocess SAR (grayscale) ---
        img   = PILImageBW.create(sar_fp)
        tfm   = Pipeline([Resize(512), ToTensor()])
        inp   = tfm(img)                       # (1, H, W)  0-1 range
        inp_n = (inp - 0.5) / 0.5             # normalise to match training

        # --- predict ---
        pred = learn.model(inp_n.unsqueeze(0).to(DEVICE))   # (1,3,H,W)
        pred = pred.squeeze(0)                               # (3,H,W)

        # --- load ground-truth RGB ---
        gt_fp  = get_optical(sar_fp)
        gt_img = PILImage.create(gt_fp)
        gt_tfm = Pipeline([Resize(512), ToTensor()])
        gt     = gt_tfm(gt_img)               # (3,H,W)  0-1 range
        gt_n   = (gt - 0.5) / 0.5

        results.append({
            "name": sar_fp.stem,
            "sar":  to_img(inp_n[None].squeeze(0), grayscale=True),    # HxW
            "pred": to_img(pred),                                        # HxWx3
            "gt":   to_img(gt_n),                                        # HxWx3
        })


# ── Plot ──────────────────────────────────────────────────────────────────────
n = len(results)
fig = plt.figure(figsize=(13, 4.5 * n), facecolor="#0d0d0d")
fig.suptitle("SAR Colorisation — Input / Predicted / Ground Truth",
             fontsize=16, color="white", y=1.01, fontweight="bold")

gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.35, wspace=0.05)
col_titles = ["SAR Input (greyscale)", "Predicted (RGB)", "Ground Truth (RGB)"]

for row, r in enumerate(results):
    imgs   = [r["sar"], r["pred"], r["gt"]]
    cmaps  = ["gray", None, None]

    for col, (im, cmap, title) in enumerate(zip(imgs, cmaps, col_titles)):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(im, cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        if row == 0:
            ax.set_title(title, color="white", fontsize=11, pad=8)
        if col == 0:
            ax.set_ylabel(r["name"], color="#aaa", fontsize=8,
                          rotation=0, labelpad=60, va="center")

plt.savefig("sar_comparison.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("✓ Saved → sar_comparison.png")


# ── Per-image MAE ─────────────────────────────────────────────────────────────
print(f"\n{'Image':<30} {'MAE':>8}")
print("─" * 40)
for r in results:
    pred_f = r["pred"].astype(float)
    gt_f   = r["gt"].astype(float)
    mae    = np.abs(pred_f - gt_f).mean()
    print(f"{r['name']:<30} {mae:>8.2f}")