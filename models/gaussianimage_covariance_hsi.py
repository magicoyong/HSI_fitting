from functools import lru_cache
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gsplat.project_gaussians_2d_covariance import project_gaussians_2d_covariance
from gsplat.rasterize_sum_gabor import rasterize_gabor_sum

from hsi_utils import spectral_smoothness_loss, total_variation_loss
from optimizer import Adan

from .gaussianimage_covariance import GaussianImage_Covariance


ChannelGroup = Tuple[int, int, int]


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    clamped = torch.clamp(values, min=1e-6)
    return torch.log(torch.expm1(clamped))


class GaussianImage_Covariance_HSI(GaussianImage_Covariance):
    GAUSSIAN_GROUP_NAMES = {"xyz", "f_dc", "cov2d", "gabor_freqs", "gabor_weights"}

    def __init__(self, loss_type="L2", **kwargs):
        self.rank = int(kwargs["rank"])
        self.spectral_channels = int(kwargs["C"])
        self.opt_type = kwargs.get("opt_type", "adam")
        self.lora_rank = int(kwargs.get("lora_rank", kwargs.get("calib_rank", 2)))
        self.max_lora_scale = float(kwargs.get("lora_alpha", kwargs.get("gamma", 0.1)))
        self.freeze_endmember = bool(
            kwargs.get("freeze_endmember", kwargs.get("freeze_endmember_calibration", False))
        )
        super().__init__(loss_type=loss_type, **kwargs)

        endmember_init = kwargs.get("E0", kwargs.get("E"))
        if endmember_init is None:
            raise ValueError("GaussianImage_Covariance_HSI requires an initial endmember matrix E0")

        self.register_buffer(
            "E0",
            torch.as_tensor(endmember_init, dtype=torch.float32, device=self.device).contiguous(),
        )

        abundance_seed = self._prepare_abundance_seed(kwargs.get("A0"))
        if abundance_seed is not None:
            self.register_buffer("abundance_seed", abundance_seed)
        else:
            self.abundance_seed = None

        feature_init = self._build_initial_feature_logits()
        self._features_dc = nn.Parameter(feature_init)

        self.lora_U = nn.Parameter(torch.empty(self.rank, self.lora_rank, device=self.device))
        self.lora_V = nn.Parameter(torch.empty(self.lora_rank, self.spectral_channels, device=self.device))
        self.lora_scale_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device))
        nn.init.kaiming_uniform_(self.lora_U, a=np.sqrt(5.0))
        nn.init.kaiming_uniform_(self.lora_V, a=np.sqrt(5.0))
        self.lora_V.data.mul_(1e-3)

        if self.freeze_endmember:
            self.lora_U.requires_grad_(False)
            self.lora_V.requires_grad_(False)
            self.lora_scale_logit.requires_grad_(False)

        self.register_buffer("abundance_background3", torch.zeros(3, device=self.device))
        self.register_buffer("abundance_background4", torch.zeros(4, device=self.device))
        self.channel_groups = self._build_channel_groups(self.rank)

        self.training_setup(lr=kwargs["lr"], update_optimizer=True, quantize=False)

    def _prepare_abundance_seed(self, abundance_init: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        if abundance_init is None:
            return None
        abundance = np.asarray(abundance_init, dtype=np.float32)
        if abundance.ndim == 2:
            if abundance.shape[0] != self.H * self.W:
                raise ValueError(
                    f"Expected A0 shape (H*W, R)=({self.H * self.W}, {self.rank}), got {abundance.shape}"
                )
            abundance = abundance.reshape(self.H, self.W, self.rank)
        elif abundance.ndim == 3:
            if abundance.shape[-1] != self.rank:
                if abundance.shape[0] == self.rank:
                    abundance = np.moveaxis(abundance, 0, -1)
                else:
                    raise ValueError(f"Unsupported A0 shape: {abundance.shape}")
        else:
            raise ValueError(f"Unsupported A0 shape: {abundance.shape}")

        return torch.from_numpy(np.ascontiguousarray(abundance)).to(self.device)

    def _build_initial_feature_logits(self) -> torch.Tensor:
        sampled_abundance = self.sample_abundance_features(self.get_xyz.detach())
        if sampled_abundance is None:
            sampled_abundance = torch.full(
                (self.init_num_points, self.rank), 0.05, dtype=torch.float32, device=self.device
            )
        return _inverse_softplus(sampled_abundance)

    @staticmethod
    @lru_cache(maxsize=None)
    def _plan_group_sizes(num_channels: int) -> Tuple[Tuple[int, int], ...]:
        if num_channels == 0:
            return tuple()

        candidates: List[Tuple[int, int, Tuple[Tuple[int, int], ...]]] = []
        if num_channels >= 3:
            tail = GaussianImage_Covariance_HSI._plan_group_sizes(num_channels - 3)
            candidates.append((0, 1 + len(tail), ((3, 3),) + tail))
        else:
            candidates.append((3 - num_channels, 1, ((num_channels, 3),)))

        if num_channels >= 4:
            tail = GaussianImage_Covariance_HSI._plan_group_sizes(num_channels - 4)
            candidates.append((0, 1 + len(tail), ((4, 4),) + tail))
        else:
            candidates.append((4 - num_channels, 1, ((num_channels, 4),)))

        best = min(candidates, key=lambda item: (item[0], item[1]))
        return best[2]

    def _build_channel_groups(self, num_channels: int) -> List[ChannelGroup]:
        groups = []
        start = 0
        for actual_channels, kernel_channels in self._plan_group_sizes(num_channels):
            groups.append((start, start + actual_channels, kernel_channels))
            start += actual_channels
        return groups

    @property
    def get_features(self):
        return F.softplus(self._features_dc)

    def abundance_values_to_logits(self, values: torch.Tensor) -> torch.Tensor:
        return _inverse_softplus(values)

    def sample_abundance_features(self, xy: torch.Tensor) -> Optional[torch.Tensor]:
        if getattr(self, "abundance_seed", None) is None:
            return None
        xy_int = torch.round(xy).long()
        x = torch.clamp(xy_int[:, 0], 0, self.W - 1)
        y = torch.clamp(xy_int[:, 1], 0, self.H - 1)
        return self.abundance_seed[y, x].contiguous()

    def get_calibrated_endmember(self) -> torch.Tensor:
        if self.freeze_endmember:
            return torch.clamp(self.E0, min=1e-6)
        delta = torch.tanh(self.lora_U @ self.lora_V)
        scale = self.max_lora_scale * torch.tanh(self.lora_scale_logit)
        return torch.clamp(self.E0 + scale * delta, min=1e-6)

    def get_endmember_delta_norm(self) -> float:
        with torch.no_grad():
            if self.freeze_endmember:
                return 0.0
            delta = torch.tanh(self.lora_U @ self.lora_V)
            scale = self.max_lora_scale * torch.tanh(self.lora_scale_logit)
            return float((scale * delta).norm().item())

    def training_setup(self, lr, update_optimizer=False, quantize=False):
        if not update_optimizer and hasattr(self, "optimizer"):
            return

        param_groups = [
            {"params": [self._xyz], "lr": lr, "name": "xyz"},
            {"params": [self._features_dc], "lr": lr, "name": "f_dc"},
            {"params": [self._cov2d], "lr": lr, "name": "cov2d"},
            {"params": [self.gabor_freqs], "lr": lr, "name": "gabor_freqs"},
            {"params": [self.gabor_weights], "lr": lr, "name": "gabor_weights"},
        ]
        if hasattr(self, "lora_U") and not self.freeze_endmember:
            param_groups.extend(
                [
                    {"params": [self.lora_U], "lr": lr, "name": "lora_u"},
                    {"params": [self.lora_V], "lr": lr, "name": "lora_v"},
                    {"params": [self.lora_scale_logit], "lr": lr, "name": "lora_scale"},
                ]
            )

        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        else:
            self.optimizer = Adan(param_groups, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)
        self.quantize = False

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            group_name = group["name"]
            if group_name not in tensors_dict:
                optimizable_tensors[group_name] = group["params"][0]
                continue

            extension_tensor = tensors_dict[group_name]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                if "exp_avg_diff" in stored_state:
                    stored_state["exp_avg_diff"] = torch.cat(
                        (stored_state["exp_avg_diff"], torch.zeros_like(extension_tensor)), dim=0
                    )
                if "neg_pre_grad" in stored_state:
                    stored_state["neg_pre_grad"] = torch.cat(
                        (stored_state["neg_pre_grad"], torch.zeros_like(extension_tensor)), dim=0
                    )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group_name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group_name] = group["params"][0]
        return optimizable_tensors

    def update_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            group_name = group["name"]
            if group_name not in tensors_dict:
                optimizable_tensors[group_name] = group["params"][0]
                continue

            extension_tensor = tensors_dict[group_name]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(extension_tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(extension_tensor)
                if "exp_avg_diff" in stored_state:
                    stored_state["exp_avg_diff"] = torch.zeros_like(extension_tensor)
                if "neg_pre_grad" in stored_state:
                    stored_state["neg_pre_grad"] = torch.zeros_like(extension_tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(extension_tensor).requires_grad_(True)
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group_name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(extension_tensor).requires_grad_(True)
                optimizable_tensors[group_name] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            group_name = group["name"]
            if group_name not in self.GAUSSIAN_GROUP_NAMES:
                optimizable_tensors[group_name] = group["params"][0]
                continue

            group_mask = self._expand_gabor_mask(mask) if group_name in ("gabor_freqs", "gabor_weights") else mask
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][group_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][group_mask]
                if "exp_avg_diff" in stored_state:
                    stored_state["exp_avg_diff"] = stored_state["exp_avg_diff"][group_mask]
                if "neg_pre_grad" in stored_state:
                    stored_state["neg_pre_grad"] = stored_state["neg_pre_grad"][group_mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][group_mask].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group_name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][group_mask].requires_grad_(True))
                optimizable_tensors[group_name] = group["params"][0]
        return optimizable_tensors

    def _pad_feature_group(self, features: torch.Tensor, kernel_channels: int) -> torch.Tensor:
        if features.shape[1] == kernel_channels:
            return features
        padded = features.new_zeros(features.shape[0], kernel_channels)
        padded[:, : features.shape[1]] = features
        return padded.contiguous()

    def _render_abundance_groups(
        self,
        depths: torch.Tensor,
        conics: torch.Tensor,
        num_tiles_hit: torch.Tensor,
        H: int,
        W: int,
        isprint: bool = False,
    ) -> torch.Tensor:
        rendered_groups = []
        features = self.get_features
        for start, end, kernel_channels in self.channel_groups:
            feature_group = features[:, start:end]
            padded_features = self._pad_feature_group(feature_group, kernel_channels)
            background = self.abundance_background3 if kernel_channels == 3 else self.abundance_background4
            rendered = rasterize_gabor_sum(
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,
                padded_features,
                self.get_opacity,
                self.get_gabor_freqs[:, 0],
                self.get_gabor_freqs[:, 1],
                self.get_gabor_weights,
                self.num_gabor,
                H,
                W,
                self.BLOCK_H,
                self.BLOCK_W,
                background=background,
                isprint=isprint,
                radius_clip=self.radius_clip,
            )
            rendered_groups.append(rendered[..., : end - start])
        return torch.cat(rendered_groups, dim=-1)

    def forward(self, isprint=False, H=None, W=None):
        H = self.H if H is None else H
        W = self.W if W is None else W
        tile_bounds = (
            (W + self.BLOCK_W - 1) // self.BLOCK_W,
            (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_covariance(
            self.get_xyz,
            self.get_cov2d_elements,
            H,
            W,
            tile_bounds,
            coords_norm=self.coords_norm,
            clip_coe=self.gs_clip_coe,
            radius_clip=self.radius_clip,
            isprint=isprint,
        )

        abundance = self._render_abundance_groups(depths, conics, num_tiles_hit, H, W, isprint=isprint)
        abundance = torch.clamp(abundance, min=0.0)

        endmember = self.get_calibrated_endmember()
        abundance_flat = abundance.view(H * W, self.rank)
        reconstruction = abundance_flat @ endmember
        reconstruction = reconstruction.view(1, H, W, self.spectral_channels).permute(0, 3, 1, 2).contiguous()

        abundance_chw = abundance.view(1, H, W, self.rank).permute(0, 3, 1, 2).contiguous()
        return {
            "render": reconstruction,
            "abundance": abundance_chw,
            "endmember": endmember,
            "num_tiles_hit": num_tiles_hit,
            "radiii": self.radii,
            "visibility_filter": self.radii > 0,
        }

    def train_iter(
        self,
        H,
        W,
        gt_image,
        isprint=False,
        tv_weight=0.0,
        spectral_weight=0.0,
        sparse_weight=0.0,
        endmember_weight=0.0,
    ):
        render_pkg = self.forward(isprint=isprint, H=H, W=W)
        image = render_pkg["render"]
        abundance = render_pkg["abundance"]
        endmember = render_pkg["endmember"]

        if self.loss_type == "L1":
            recon_loss = F.l1_loss(image, gt_image)
        else:
            recon_loss = F.mse_loss(image, gt_image)

        loss = recon_loss
        if tv_weight > 0:
            loss = loss + tv_weight * total_variation_loss(abundance)
        if spectral_weight > 0:
            loss = loss + spectral_weight * spectral_smoothness_loss(image)
        if sparse_weight > 0:
            loss = loss + sparse_weight * abundance.mean()
        if endmember_weight > 0:
            loss = loss + endmember_weight * F.mse_loss(endmember, self.E0)

        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / max(mse_loss.item(), 1e-12))
        self.optimizer_step()
        return loss, psnr, image.detach(), recon_loss.detach(), self.get_endmember_delta_norm()
