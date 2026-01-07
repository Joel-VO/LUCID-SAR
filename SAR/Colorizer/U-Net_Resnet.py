from fastai.vision.all import *
import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

sar_path = Path("Dataset/SAR_Color Dataset/train/sar_images")
opt_path = Path("Dataset/SAR_Color Dataset/train/rgb_images")

def get_optical(fn):
    return opt_path/fn.name

sar_colorizer = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), ImageBlock),
    get_items=get_image_files,
    get_y=get_optical,
    splitter=RandomSplitter(0.1),
    item_tfms=Resize(512),
    batch_tfms=Normalize.from_stats(*imagenet_stats)
)

dls = sar_colorizer.dataloaders(sar_path, bs=2, num_workers=0) # worker count = 0 only for python 3.14, lower versions can have more

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

unet = DynamicUnet(
    encoder,
    n_out=3,
    img_size=(512, 512),
    norm_type=None
)

model = SARWrapper(unet)

loss_func = L1LossFlat()

learn = Learner(
    dls,
    model,
    loss_func=loss_func,
    metrics=[mae]
)

learn.fine_tune(20, base_lr=1e-4)

# learn.save("sar_colorizer_resnet34")