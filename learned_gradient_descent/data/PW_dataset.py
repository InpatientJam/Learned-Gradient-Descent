import hydra
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PWDataset(Dataset):
    def __init__(self, opt):
        self.dataset = np.load(hydra.utils.to_absolute_path(opt.dataset_path))

        # force loading npz data to memory
        self.dataset = {k: self.dataset[k] for k in self.dataset}
        if hasattr(opt, "skip"):
            for k, v in self.dataset.items():
                self.dataset[k] = v[::opt.skip]

    def __getitem__(self, index):
        joint3d = self.dataset["poses3d"][index] * 1000
        joint2d = self.dataset["poses2d"][index][:, :2]
        confidence = self.dataset["poses2d"][index][:, 2]

        smpl_thetas = self.dataset["smpl_pose"][index]
        smpl_betas = self.dataset["smpl_shape"][index]

        return {
            'betas': smpl_betas,
            'thetas': smpl_thetas,
            'joint2d': joint2d,
            'joint3d': joint3d,
            'confidence': confidence,
        }

    def __len__(self):
        """Return the total number of images."""
        return self.dataset["smpl_shape"].shape[0]
