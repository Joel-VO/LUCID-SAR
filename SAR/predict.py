import torch
from fastai.vision.all import *

path = Path('data')
dls = get_dls(path, bs=8, size=256)
model = create_unet(resnet34, n_out=3, img_size=(256,256))
learn = Learner(dls, model, loss_func=CombinedLoss())

# Load weights
learn.load('sar_colorization')

# Inference
learn.predict(PILImageBW.create('test_sar.png'))