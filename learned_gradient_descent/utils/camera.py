import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PinholeCamera(nn.Module):
    def __init__(self, fx, fy, cx, cy, R, t):
        super().__init__()

        K = np.asarray([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]], dtype=np.float32)
        Rt = np.asarray([[R[0, 0], R[0, 1], R[0, 2], t[0]],
                         [R[1, 0], R[1, 1], R[1, 2], t[1]],
                         [R[2, 0], R[2, 1], R[2, 2], t[2]]], dtype=np.float32)
        P = K @ Rt
        self.register_buffer("P", torch.from_numpy(P), persistent=False)

    def forward(self, x):
        x = F.pad(x, (0, 1), "constant", 1)
        proj = torch.einsum("bij,kj->bik", x, self.P)
        proj[:, :, 0] *= -1
        return proj[:, :, :2] / proj[:, :, 2, None]
