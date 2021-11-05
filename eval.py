import os
import glob

import hydra
import torch
import pytorch_lightning as pl

from learned_gradient_descent.data import create_dataset
from learned_gradient_descent.models.baseline import LGDModel


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, testset):
        super().__init__()
        self.testset = testset

    def test_dataloader(self):
        return self.testset


@hydra.main(config_path="configs", config_name="eval")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    testset = create_dataset(opt.data.test)
    datamodule = CustomDataModule(testset)

    checkpoints = sorted(glob.glob("ckpts/*.ckpt"), key=os.path.getmtime)
    print("checkpoints", checkpoints)
    model = LGDModel.load_from_checkpoint(checkpoints[-1], strict=False, opt=opt.model)
    model.eval()
    trainer = pl.Trainer(gpus=1, accelerator="gpu")
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
