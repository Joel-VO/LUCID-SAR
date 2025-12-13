import torch
import torch.nn as nn


def euclidean_TV_loss(y_pred, y_ground, lambda_tv = 0.002):
    mse = nn.MSELoss()
    euclidean = mse(y_pred, y_ground)

    dx = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
    dy = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    tv_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
    # tv_loss = torch.mean(torch.sqrt(dx**2 + dy**2 + 1e-8)) # modified the tv loss equation for batch compute, else normally just take sum

    loss = euclidean + lambda_tv*tv_loss
    return loss
