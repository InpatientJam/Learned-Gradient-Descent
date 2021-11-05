import torch
import torch.nn as nn
import hydra
import numpy as np

SMPL_TO_COCO = [24, 12, 17, 19, 21, 16, 18,
                20, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28]
H36M_TO_SPIN = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]


class SMPLJointMapper(nn.Module):
    def __init__(self):
        super().__init__()

        # load predefined regressor
        path = hydra.utils.to_absolute_path('data/J_regressor_h36m.npy')
        regressor = np.load(path).transpose(1, 0).astype(np.float32)
        self.register_buffer("regressor_h36m", torch.from_numpy(
            regressor), persistent=False)

    def forward(self, joints, vertices, output_format):
        """
        Args:
            joints (B, J, 3)
            vertices (B, V, 3)
            output_format (str)

        Returns:
            joints (B, J, 3)
        """
        output_format = output_format.upper()
        if output_format == "3DPW":
            return joints[:, :24]
        elif output_format == "COCO18":
            return joints[:, SMPL_TO_COCO]
        elif output_format == "SPIN":
            joints = torch.einsum(
                "bij, ik->bkj", [vertices, self.regressor_h36m])
            return joints[:, H36M_TO_SPIN]
        else:
            raise ValueError(output_format)
