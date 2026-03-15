"""
Image Colorization with Pretrained ResNet18 U-Net + PatchGAN
============================================================
Based on: "Colorizing black & white images with U-Net and conditional GAN"
by Moein Shariatnia — reimplemented as clean, runnable Python.

This is the BEST model from the tutorial:
  - Generator : U-Net with pretrained ResNet18 encoder (fastai DynamicUnet)
  - Discriminator: PatchGAN (70x70 receptive field)
  - Training  : 2-stage — L1 pretrain first, then full GAN fine-tuning

Requirements:
    pip install torch torchvision fastai Pillow numpy matplotlib scikit-image tqdm tifffile
"""

# ─────────────────────────────────────────────
# 1.  Imports
# ─────────────────────────────────────────────
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet

from skimage.color import rgb2lab, lab2rgb
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────────
# 2.  Config
# ─────────────────────────────────────────────
SIZE            = 256    # image size
BATCH_SIZE      = 16
PRETRAIN_EPOCHS = 20     # Stage 1: generator pretrain with L1 only
GAN_EPOCHS      = 20     # Stage 2: full adversarial training
LAMBDA_L1       = 100.0  # weight for L1 loss inside GAN training
LR_G            = 2e-4
LR_D            = 2e-4
BETA1           = 0.5
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# 3.  Dataset
# ─────────────────────────────────────────────
class ColorizationDataset(Dataset):
    """
    Paired dataset:
        sar_paths   : denoised SAR images  → model input  (grayscale L channel, [-1,1])
        color_paths : ground truth colour  → target output (ab channels, [-1,1])
    """
    def __init__(self, sar_paths, color_paths, size=SIZE, augment=True):
        assert len(sar_paths) == len(color_paths)
        self.sar_paths   = sar_paths
        self.color_paths = color_paths
        self.size        = size
        self.augment     = augment

    def __len__(self):
        return len(self.sar_paths)

    def __getitem__(self, idx):
        # Load .tif files via tifffile, then convert to PIL for resize/flip
        sar_arr   = tifffile.imread(self.sar_paths[idx])
        color_arr = tifffile.imread(self.color_paths[idx])

        def to_uint8(arr):
            arr = arr.astype("float32")
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            return arr.astype("uint8")

        sar_arr   = to_uint8(sar_arr)
        color_arr = to_uint8(color_arr)

        # Ensure SAR is 2-D grayscale, colour is H×W×3
        if sar_arr.ndim == 3:
            sar_arr = sar_arr[:, :, 0]           # take first band
        if color_arr.ndim == 2:
            color_arr = np.stack([color_arr] * 3, axis=-1)
        elif color_arr.shape[2] > 3:
            color_arr = color_arr[:, :, :3]      # drop alpha / extra bands

        sar   = Image.fromarray(sar_arr,   mode="L")
        color = Image.fromarray(color_arr, mode="RGB")

        if self.augment and np.random.rand() > 0.5:
            sar   = transforms.functional.hflip(sar)
            color = transforms.functional.hflip(color)

        sar   = transforms.functional.resize(sar,   (self.size, self.size))
        color = transforms.functional.resize(color, (self.size, self.size))

        # SAR → L channel: [0,255] → [-1,1]
        L = np.array(sar, dtype="float32")[:, :, np.newaxis]
        L = (L / 127.5) - 1.0

        # colour image → ab channels normalised to [-1,1]
        ab = rgb2lab(np.array(color)).astype("float32")[:, :, 1:3] / 110.0

        L  = torch.from_numpy(L ).permute(2, 0, 1)   # (1, H, W)
        ab = torch.from_numpy(ab).permute(2, 0, 1)   # (2, H, W)
        return L, ab


def make_dataloaders(train_sar, train_color, val_sar, val_color,
                     batch_size=BATCH_SIZE):
    train_ds = ColorizationDataset(train_sar, train_color, augment=True)
    val_ds   = ColorizationDataset(val_sar,   val_color,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
    return train_dl, val_dl


# ─────────────────────────────────────────────
# 4.  Generator — ResNet18 U-Net (fastai)
# ─────────────────────────────────────────────
def build_generator(n_input=1, n_output=2, size=SIZE):
    """
    Uses fastai's DynamicUnet wrapped around a pretrained ResNet18 body.
    The pretrained encoder weights are kept intact — only the decoder/head
    is randomly initialised (done inside DynamicUnet).
    """
    body = create_body(resnet18(), pretrained=True, n_in=n_input, cut=-2)
    net  = DynamicUnet(body, n_out=n_output, img_size=(size, size),
                       self_attention=True, act_cls=nn.ReLU)
    return net


# ─────────────────────────────────────────────
# 5.  Discriminator — PatchGAN
# ─────────────────────────────────────────────
class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.
    Input: concatenation of L (1-ch) and ab (2-ch) → 3 channels total.
    Output: grid of real/fake scores.
    """
    def __init__(self, in_channels=3, num_filters=64, n_down=3):
        super().__init__()
        layers = [self._block(in_channels, num_filters, norm=False)]
        in_ch  = num_filters
        for i in range(1, n_down):
            out_ch = in_ch * 2
            stride = 1 if i == n_down - 1 else 2
            layers.append(self._block(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    @staticmethod
    def _block(in_c, out_c, stride=2, norm=True):
        block = [nn.Conv2d(in_c, out_c, kernel_size=4,
                           stride=stride, padding=1, bias=not norm)]
        if norm:
            block.append(nn.BatchNorm2d(out_c))
        block.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*block)

    def forward(self, x):
        return self.model(x)


# ─────────────────────────────────────────────
# 6.  GAN Loss helper
# ─────────────────────────────────────────────
class GANLoss(nn.Module):
    def __init__(self, gan_mode="vanilla", real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss() if gan_mode == "vanilla" else nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        labels = self.real_label if target_is_real else self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        return self.loss(preds, labels)


# ─────────────────────────────────────────────
# 7.  Weight initialisation (discriminator only)
# ─────────────────────────────────────────────
def init_weights(net, init="norm", gain=0.02):
    """Initialise weights of a network. Do NOT apply to the pretrained generator."""
    def init_fn(m):
        cls = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in cls or "Linear" in cls):
            if init == "norm":
                nn.init.normal_(m.weight, 0.0, gain)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif init == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif "BatchNorm2d" in cls:
            nn.init.normal_(m.weight, 1.0, gain)
            nn.init.constant_(m.bias, 0.0)
    net.apply(init_fn)
    return net


# ─────────────────────────────────────────────
# 8.  Full ColorizationModel (wraps G + D)
# ─────────────────────────────────────────────
class ColorizationModel:
    def __init__(self, net_G=None, lr_G=LR_G, lr_D=LR_D,
                 beta1=BETA1, lambda_L1=LAMBDA_L1):
        self.device    = DEVICE
        self.lambda_L1 = lambda_L1

        # Generator — pretrained encoder weights are preserved as-is
        if net_G is None:
            self.net_G = build_generator().to(self.device)
        else:
            self.net_G = net_G.to(self.device)

        # Discriminator — fully randomly initialised
        self.net_D = init_weights(
            PatchDiscriminator(in_channels=3).to(self.device))

        # Losses
        self.GANcriterion = GANLoss(gan_mode="vanilla").to(self.device)
        self.L1criterion  = nn.L1Loss()

        # Optimisers
        self.opt_G = optim.Adam(self.net_G.parameters(),
                                lr=lr_G, betas=(beta1, 0.999))
        self.opt_D = optim.Adam(self.net_D.parameters(),
                                lr=lr_D, betas=(beta1, 0.999))

    def set_input(self, data):
        self.L  = data[0].to(self.device)
        self.ab = data[1].to(self.device)

    def forward(self):
        self.fake_ab = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_ab], dim=1)
        real_image = torch.cat([self.L, self.ab],      dim=1)
        fake_preds = self.net_D(fake_image.detach())
        real_preds = self.net_D(real_image)
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D      = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_ab], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1  = self.L1criterion(self.fake_ab, self.ab) * self.lambda_L1
        self.loss_G     = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        self.net_G.train()
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()


# ─────────────────────────────────────────────
# 9.  Logging helper
# ─────────────────────────────────────────────
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.count = self.sum = 0
    def update(self, val, n=1):
        self.count += n
        self.sum   += val * n
    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


# ─────────────────────────────────────────────
# 10. Stage 1 — Pretrain generator with L1 only
# ─────────────────────────────────────────────
def pretrain_generator(net_G, train_dl, optimizer, criterion,
                       epochs=PRETRAIN_EPOCHS):
    net_G.train()
    for epoch in range(epochs):
        meter = AverageMeter()
        for L, ab in tqdm(train_dl, desc=f"Pretrain {epoch+1}/{epochs}"):
            L, ab = L.to(DEVICE), ab.to(DEVICE)
            preds = net_G(L)
            loss  = criterion(preds, ab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            meter.update(loss.item(), n=L.size(0))
        print(f"  Epoch {epoch+1:>3} | L1 loss: {meter.avg:.5f}")


# ─────────────────────────────────────────────
# 11. Stage 2 — Full GAN training
# ─────────────────────────────────────────────
def train_gan(model, train_dl, epochs=GAN_EPOCHS):
    for epoch in range(epochs):
        meters = {k: AverageMeter() for k in
                  ["loss_D_fake", "loss_D_real", "loss_D",
                   "loss_G_GAN",  "loss_G_L1",   "loss_G"]}
        t0 = time.time()
        for data in tqdm(train_dl, desc=f"GAN {epoch+1}/{epochs}"):
            model.set_input(data)
            model.optimize()
            bs = data[0].size(0)
            for k in meters:
                meters[k].update(getattr(model, k).item(), n=bs)
        print(
            f"Epoch {epoch+1:>3} | "
            f"D: {meters['loss_D'].avg:.4f}  "
            f"G: {meters['loss_G'].avg:.4f}  "
            f"G_L1: {meters['loss_G_L1'].avg:.4f}  "
            f"G_GAN: {meters['loss_G_GAN'].avg:.4f}  "
            f"({time.time()-t0:.0f}s)"
        )


# ─────────────────────────────────────────────
# 12. Visualisation
# ─────────────────────────────────────────────
def lab_to_rgb(L, ab):
    """
    Convert L (1,H,W) and ab (2,H,W) tensors → RGB numpy (H,W,3).
    L is in [-1,1] (SAR pixel normalisation), ab is in [-1,1].
    """
    L_np  = (L.cpu().numpy()[0] + 1.0) * 127.5        # (H,W), [0,255]
    # Re-scale to Lab L range [0,100] for lab2rgb
    L_lab = L_np / 255.0 * 100.0                       # (H,W), [0,100]
    ab_np = ab.cpu().numpy() * 110.0                   # (2,H,W)
    Lab   = np.stack([L_lab, ab_np[0], ab_np[1]], axis=-1)  # (H,W,3)
    return (np.clip(lab2rgb(Lab), 0, 1) * 255).astype("uint8")


def visualize_predictions(model, val_dl, n=5):
    model.net_G.eval()
    data              = next(iter(val_dl))
    L_batch, ab_batch = data[0][:n].to(DEVICE), data[1][:n]

    with torch.no_grad():
        fake_ab = model.net_G(L_batch).cpu()

    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
    for i in range(n):
        L_i  = L_batch[i].cpu()
        real = lab_to_rgb(L_i, ab_batch[i])
        fake = lab_to_rgb(L_i, fake_ab[i])
        gray = ((L_i.numpy()[0] + 1.0) * 127.5).astype("uint8")

        axes[i, 0].imshow(gray, cmap="gray"); axes[i, 0].set_title("Input (SAR)")
        axes[i, 1].imshow(fake);              axes[i, 1].set_title("Colorized")
        axes[i, 2].imshow(real);              axes[i, 2].set_title("Ground truth")
        for ax in axes[i]: ax.axis("off")

    plt.tight_layout()
    plt.savefig("colorization_results.png", dpi=120)
    plt.show()
    print("Saved: colorization_results.png")


# ─────────────────────────────────────────────
# 13. Main
# ─────────────────────────────────────────────
def main():
    COLOR_DIR = Path("Dataset/SAR_Color_Dataset/train/rgb_images")   # ← ground truth colour images
    SAR_DIR   = Path("Dataset/SAR_Color_Dataset/train/denoised_sar_images")  # ← denoised SAR input images

    for d in (COLOR_DIR, SAR_DIR):
        if not d.exists():
            raise FileNotFoundError(f"Folder not found: {d}")

    color_paths = sorted(glob.glob(str(COLOR_DIR / "*.tif")) +
                         glob.glob(str(COLOR_DIR / "*.tiff")))
    sar_paths   = sorted(glob.glob(str(SAR_DIR   / "*.tif")) +
                         glob.glob(str(SAR_DIR   / "*.tiff")))

    assert len(color_paths) == len(sar_paths), (
        f"Mismatch: {len(color_paths)} colour images vs {len(sar_paths)} SAR images. "
        "Files are matched by sort order — ensure filenames correspond."
    )

    np.random.seed(42)
    indices              = np.random.permutation(len(color_paths))
    split                = int(0.9 * len(indices))
    train_idx, val_idx   = indices[:split], indices[split:]

    train_color = [color_paths[i] for i in train_idx]
    train_sar   = [sar_paths[i]   for i in train_idx]
    val_color   = [color_paths[i] for i in val_idx]
    val_sar     = [sar_paths[i]   for i in val_idx]

    print(f"Train: {len(train_color)}  Val: {len(val_color)}")

    train_dl, val_dl = make_dataloaders(train_sar, train_color, val_sar, val_color)

    # ── Stage 1: Pretrain generator (L1 only) ──
    print("\n=== Stage 1: Generator pretraining (L1 loss only) ===")
    net_G     = build_generator().to(DEVICE)
    opt_G_pre = optim.Adam(net_G.parameters(), lr=1e-4)
    pretrain_generator(net_G, train_dl, opt_G_pre, nn.L1Loss(),
                       epochs=PRETRAIN_EPOCHS)
    torch.save(net_G.state_dict(), "generator_pretrained.pt")
    print("Saved: generator_pretrained.pt")

    # ── Stage 2: Full GAN training ──
    print("\n=== Stage 2: Full GAN (adversarial) training ===")
    model = ColorizationModel(net_G=net_G)
    train_gan(model, train_dl, epochs=GAN_EPOCHS)
    torch.save(model.net_G.state_dict(), "generator_final.pt")
    torch.save(model.net_D.state_dict(), "discriminator_final.pt")
    print("Saved: generator_final.pt  discriminator_final.pt")

    # ── Visualise ──
    print("\n=== Visualising results ===")
    visualize_predictions(model, val_dl, n=5)


# ─────────────────────────────────────────────
# 14. Inference helper (load saved model)
# ─────────────────────────────────────────────
def colorize_image(img_path, weights_path="generator_final.pt", size=SIZE):
    """
    Colorize a single SAR .tif image using a saved generator.
    Returns: PIL RGB image.
    """
    net_G = build_generator().to(DEVICE)
    net_G.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    net_G.eval()

    arr = tifffile.imread(img_path)
    arr = arr.astype("float32")
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    arr = arr.astype("uint8")
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    img = Image.fromarray(arr, mode="L")
    img = transforms.functional.resize(img, (size, size))

    L   = np.array(img, dtype="float32")[:, :, np.newaxis]
    L   = (L / 127.5) - 1.0
    L_t = torch.from_numpy(L).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        ab_t = net_G(L_t).squeeze(0)

    rgb = lab_to_rgb(L_t.squeeze(0).cpu(), ab_t.cpu())
    return Image.fromarray(rgb)


if __name__ == "__main__":
    main()