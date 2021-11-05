from typing import Dict, Any
from itertools import chain
import numpy as np
import hydra

import torch
import torch.optim as optim
import pytorch_lightning as pl
from smplx import SMPL

from ..utils.loss import weighted_L1_loss
from ..utils.camera import PinholeCamera
from ..utils.pose_util import compute_MPJPE, normalize_p2d
from ..utils.network import MLPBlock
from ..utils.smpl import SMPLJointMapper


class LGDModel(pl.LightningModule, pl.core.hooks.CheckpointHooks):
    def __init__(self, opt):
        super().__init__()
        self.automatic_optimization = False

        # Constants
        NUM_JOINT = 18
        NUM_BETA = 10
        NUM_THETA = 24 * 3
        NUM_FEATURE = NUM_JOINT * 2 + NUM_THETA * 2
        NUM_HIDDEN = 1024
        self.NUM_STAGES = 3
        self.opt = opt

        # network
        self.training_modules = [
            'pose_upd_net', 'pose_init_net', 'shape_upd_net', 'shape_init_net'
        ]
        self.pose_init_net = MLPBlock(NUM_JOINT * 2, NUM_THETA, NUM_HIDDEN)
        self.pose_upd_net = MLPBlock(NUM_FEATURE, NUM_THETA, NUM_HIDDEN)
        self.shape_init_net = MLPBlock(NUM_JOINT * 2, NUM_BETA, NUM_HIDDEN)
        self.shape_upd_net = MLPBlock(NUM_FEATURE, NUM_BETA, NUM_HIDDEN)

        # other submodules
        self.camera = PinholeCamera(
            fx=256, fy=256, cx=0, cy=0, R=np.eye(3), t=(0, 0, -6))
        self.body_model = SMPL(model_path=hydra.utils.to_absolute_path(
            'data/smplx_models/smpl'), gender="male")
        self.joint_mapper = SMPLJointMapper()
        self.criterionLoss = torch.nn.L1Loss()

        self.confidence_threshold = 0.01

        self.confidence_dist = torch.distributions.Bernoulli(0.8)
        self.joint_weights = torch.ones(1, 18, 3).to("cuda")
        self.joint_weights[:, [4, 7, 10, 13]] = 8
        self.joint_weights[:, [3, 6, 9, 12]] = 4
        self.joint_weights[:, [2, 5, 8, 11]] = 2

    def forward(self, batch):
        torch.set_grad_enabled(True)

        x = batch["joint2d"].flatten(start_dim=1)
        confidence = batch["confidence"]
        batch_size = len(x)

        thetas_history = []
        betas_history = []
        joint2d_history = []
        joint3d_history = []
        for i in range(self.NUM_STAGES + 1):
            if i == 0:
                thetas = self.pose_init_net(x)
                betas = self.shape_init_net(x)
            else:
                feature = torch.cat(
                    [x, thetas, thetas.grad.detach() * batch_size], dim=1)
                thetas = thetas + self.pose_upd_net(feature) * 0.1
                betas = betas + self.shape_upd_net(feature) * 0.1

            # ask torch to save grad for intermediate vars
            thetas.retain_grad()
            joint3d = self.get_joint3d(thetas, betas)
            joint2d = self.camera(joint3d)
            joint2d = normalize_p2d(joint2d,
                                    confidence,
                                    self.confidence_threshold)

            if i < self.NUM_STAGES:
                reprojection_error = weighted_L1_loss(
                    joint2d, x,
                    confidence.unsqueeze(-1).expand(-1, -1, 2))
                reprojection_error.backward(retain_graph=True)

            thetas_history.append(thetas)
            betas_history.append(betas)
            joint2d_history.append(joint2d)
            joint3d_history.append(joint3d)
        return {
            "betas_history": betas_history,
            "thetas_history": thetas_history,
            "joint2d_history": joint2d_history,
            "joint3d_history": joint3d_history
        }

    def configure_optimizers(self):
        params = chain(*[
            getattr(self, module).parameters()
            for module in self.training_modules
        ])
        self.optimizer = optim.Adam(params, lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=1,
                                                   gamma=0.8)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """remove SMPL parameters"""
        keys = [key for key in checkpoint["state_dict"]
                if key.startswith("body_model")]
        for key in keys:
            del checkpoint["state_dict"][key]
        return super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """remove SMPL parameters"""
        keys = [key for key in checkpoint["state_dict"]
                if key.startswith("body_model")]
        for key in keys:
            del checkpoint["state_dict"][key]
        return super().on_save_checkpoint(checkpoint)

    def training_step(self, batch, batch_idx, **kwargs):
        # data augmentation
        joint3d = self.get_joint3d(batch["thetas"], betas=batch["betas"])
        joint2d = self.camera(joint3d)
        confidence = self.confidence_dist.sample(
            joint2d.shape[:2]).to(self.device).float()
        joint2d = normalize_p2d(joint2d,
                                confidence,
                                threshold=self.confidence_threshold)
        joint2d[confidence < self.confidence_threshold, :] = 0.5
        batch["joint2d"] = joint2d
        batch["joint3d"] = joint3d
        batch["confidence"] = confidence

        # training
        predicts = self.forward(batch)
        losses = self.compute_loss(predicts, batch)

        self.optimizer.zero_grad()
        losses["loss_total"].backward()
        self.optimizer.step()

        for k, v in losses.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False)

        return {
            k: v.detach() for (k, v) in losses.items()
        }

    def training_epoch_end(self, *args, **kwargs):
        self.scheduler.step()

    def validation_step(self, batch, batch_idx):
        batch["joint2d"] = normalize_p2d(
            batch["joint2d"],
            batch["confidence"],
            threshold=self.confidence_threshold
        )
        batch["confidence"] = (batch["confidence"] >=
                               self.confidence_threshold).float()

        outputs = self.forward(batch)
        predicts = {
            "betas": outputs["betas_history"][-1],
            "thetas": outputs["thetas_history"][-1],
        }
        losses = self.compute_valid_MPJPE(predicts, batch)
        return {
            k: v.detach() for (k, v) in losses.items()
        }

    def validation_epoch_end(self, outputs):
        total_loss_val = {}
        for output in outputs:
            for k, v in output.items():
                if k not in total_loss_val:
                    total_loss_val[k] = [v]
                else:
                    total_loss_val[k].append(v)
        for k, v in total_loss_val.items():
            self.log(f"val/{k}", torch.cat(v).mean())

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        total_loss_test = {}
        for output in outputs:
            for k, v in output.items():
                if k not in total_loss_test:
                    total_loss_test[k] = [v]
                else:
                    total_loss_test[k].append(v)
        for k, v in total_loss_test.items():
            self.log(f"test/{k}", torch.cat(v).mean())

    def compute_loss(self, predicts, targets):
        steps = len(predicts["betas_history"])

        loss_joint2d_total = 0
        loss_joint3d_total = 0
        loss_thetas_total = 0
        loss_betas_total = 0
        for step in range(steps):
            thetas_pred = predicts["thetas_history"][step]
            betas_pred = predicts["betas_history"][step]
            joint2d_pred = predicts["joint2d_history"][step]
            joint3d_pred = predicts["joint3d_history"][step]

            loss_joint2d_total += weighted_L1_loss(
                joint2d_pred, targets["joint2d"], self.joint_weights[:, :, :2])
            loss_joint3d_total += weighted_L1_loss(joint3d_pred,
                                                   targets["joint3d"],
                                                   self.joint_weights)
            loss_thetas_total += self.criterionLoss(thetas_pred,
                                                    targets["thetas"])
            loss_betas_total += self.criterionLoss(betas_pred,
                                                   targets["betas"])

        loss_total = (loss_joint2d_total * 0.3 + loss_joint3d_total * 0.3 +
                      loss_thetas_total * 1.0 + loss_betas_total * 1.0) / steps
        return {
            "loss_betas": loss_betas_total / steps,
            "loss_thetas": loss_thetas_total / steps,
            "loss_joint2d": loss_joint2d_total / steps,
            "loss_joint3d": loss_joint3d_total / steps,
            "loss_total": loss_total,
        }

    def compute_valid_MPJPE(self, x, y):
        # 3DPW
        joint3d_pr = self.get_joint3d(x["thetas"], x["betas"], protocol='3DPW')
        err_3dpw = compute_MPJPE(joint3d_pr, y["joint3d"])

        # SPIN
        joint3d_pr = self.get_joint3d(x["thetas"], x["betas"], protocol='SPIN')
        joint3d_gt = self.get_joint3d(y["thetas"], y["betas"], protocol='SPIN')
        err_spin = compute_MPJPE(joint3d_pr, joint3d_gt * 1000)
        return {
            "MPJPE_aligned": torch.from_numpy(np.asarray(err_3dpw)),
            "MPJPE_aligned_spin": torch.from_numpy(np.asarray(err_spin)),
        }

    def get_joint3d(self, theta, betas, protocol="COCO18"):
        smpl_output = self.body_model.forward(betas=betas,
                                              body_pose=theta[:, 3:],
                                              global_orient=theta[:, :3])
        joint3d = self.joint_mapper(smpl_output.joints,
                                    smpl_output.vertices,
                                    output_format=protocol)
        return joint3d
