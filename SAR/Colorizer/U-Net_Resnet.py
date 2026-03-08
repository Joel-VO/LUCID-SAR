# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

from fastai.vision.all import *
import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import vgg16
import torch.nn.functional as F

sar_path = Path("Dataset/SAR_Color_Dataset/train/sar_images")
opt_path = Path("Dataset/SAR_Color_Dataset/train/rgb_images")

def get_optical(fn):
    return opt_path / fn.name

# Fix: use grayscale-appropriate normalization
sar_colorizer = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), ImageBlock),
    get_items=get_image_files,
    get_y=get_optical,
    splitter=RandomSplitter(0.1),
    item_tfms=Resize(512),
    batch_tfms=[
        *aug_transforms(flip_vert=False, max_rotate=10, max_zoom=1.1),
        Normalize.from_stats([0.5], [0.5])  # grayscale stats
    ]
)

dls = sar_colorizer.dataloaders(sar_path, bs=4, num_workers=4)


class SARWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)


# Perceptual loss using VGG16 features
class PerceptualLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        # Ensure RGB for VGG
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        vgg_pred = self.vgg(pred)
        vgg_target = self.vgg(target)
        perceptual_loss = self.l1(vgg_pred, vgg_target)
        return l1_loss + self.weight * perceptual_loss


encoder = resnet34(weights=ResNet34_Weights.DEFAULT)
encoder = nn.Sequential(*list(encoder.children())[:-2])

unet = DynamicUnet(
    encoder,
    n_out=3,
    img_size=(512, 512),
    norm_type=None
)

model = SARWrapper(unet)
loss_func = PerceptualLoss(weight=0.1)

learn = Learner(
    dls,
    model,
    loss_func=loss_func,
    metrics=[mae]
)

# Use fp16 for speed/memory, fit_one_cycle for better convergence
learn.to_fp16()
learn.fit_one_cycle(20, lr_max=1e-4)

learn.save("models/sar_colorizer_unet_resnet34")