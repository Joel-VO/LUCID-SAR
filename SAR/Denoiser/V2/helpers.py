import torch
import torch.nn as nn
from pytorch_msssim import ssim


def euclidean_TV_loss(y_pred, y_ground, lambda_tv = 0.002):
    mse = nn.MSELoss()
    euclidean = mse(y_pred, y_ground)

    dx = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
    dy = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    tv_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
    # tv_loss = torch.mean(torch.sqrt(dx**2 + dy**2 + 1e-8)) # modified the tv loss equation for batch compute, else normally just take sum

    loss = euclidean + lambda_tv*tv_loss
    return loss


def euclidean_TV_SSIM_loss(y_pred, y_ground, lambda_tv=0.01, lambda_ssim=0.1):

    l1 = nn.L1Loss()(y_pred, y_ground)

    # TV loss
    dx = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
    dy = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    tv = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))


    ssim_loss = 1 - ssim(y_pred, y_ground, data_range=1.0, size_average=True)

    TV_loss = (lambda_tv * tv)
    SSIM_loss = (lambda_ssim * ssim_loss)
    total_loss = l1 + TV_loss+SSIM_loss

    return total_loss