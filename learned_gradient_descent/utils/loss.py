import torch


def weighted_MSE_loss(input, target, weight):
    b = input.shape[0]
    return (weight.flatten(start_dim=1) *
            (input.reshape(b, -1) - target.reshape(b, -1))**2).mean()


def weighted_L1_loss(input, target, weight):
    b = input.shape[0]
    return (weight.flatten(start_dim=1) *
            torch.abs(input.reshape(b, -1) - target.reshape(b, -1))).mean()
