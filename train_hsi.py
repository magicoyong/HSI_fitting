import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from endmember import nmf_initialization
from hsi_utils import compute_sam, cube_to_tensor, load_hsi_cube
from models.gaussianimage_covariance_hsi import GaussianImage_Covariance_HSI
from models.utils import LogWriter


class HSIFullTrainer:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("HSI fitting requires CUDA-enabled PyTorch and the compiled gsplat CUDA extension.")

        self.args = args
        self.device = torch.device("cuda:0")
        self.data_path = Path(args.input).expanduser()
        self.dataset_name = self.data_path.stem if self.data_path.is_file() else self.data_path.name

        cube_hwc = load_hsi_cube(args.input, mat_key=args.mat_key, channel_axis=args.channel_axis)
        self.H, self.W, self.C = cube_hwc.shape
        self.gt_cube = cube_hwc
        self.gt_image = cube_to_tensor(cube_hwc, self.device)

        E0, A0 = nmf_initialization(self.gt_image, args.rank)
        self.rank = int(E0.shape[0])

        self.log_dir = self._build_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logwriter = LogWriter(self.log_dir)

        block_h, block_w = 16, 16
        self.model = GaussianImage_Covariance_HSI(
            loss_type=args.loss_type,
            opt_type=args.opt_type,
            num_points=args.num_points,
            H=self.H,
            W=self.W,
            rank=self.rank,
            C=self.C,
            E0=E0,
            A0=A0,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            freeze_endmember=args.freeze_endmember,
            BLOCK_H=block_h,
            BLOCK_W=block_w,
            device=self.device,
            lr=args.lr,
            num_gabor=args.num_gabor,
            quantize=False,
            args=args,
            logwriter=self.logwriter,
        ).to(self.device)
        self.max_num_points = args.max_num_points

    def _build_log_dir(self) -> Path:
        endmember_tag = "freezeE" if self.args.freeze_endmember else f"lora{self.args.lora_rank}_a{self.args.lora_alpha}"
        return (
            Path(self.args.output_root)
            / self.dataset_name
            / f"rank{self.args.rank}_{endmember_tag}_g{self.args.num_gabor}_pts{self.args.num_points}"
        )

    def _sample_new_features(self, sampled_xy: torch.Tensor) -> torch.Tensor:
        sampled = self.model.sample_abundance_features(sampled_xy.float())
        if sampled is None:
            sampled = torch.full((sampled_xy.shape[0], self.rank), 0.05, device=self.device)
        return self.model.abundance_values_to_logits(sampled.float().to(self.device))

    def add_sample_positions(self, render_image: torch.Tensor, iteration: int) -> int:
        errors = torch.abs(render_image - self.gt_image).sum(dim=1)
        normalized = errors / torch.clamp(errors.sum(), min=1e-8)

        base_num_samples = 1000
        if iteration == self.args.iterations - self.args.grow_iter:
            dynamic_num_samples = max(0, self.max_num_points - self.model.cur_num_points)
        else:
            dynamic_num_samples = max(
                0,
                min(base_num_samples, self.max_num_points - self.model.cur_num_points),
            )

        if dynamic_num_samples == 0:
            return 0

        flat = normalized.view(-1)
        _, sampled_indices = torch.topk(flat, dynamic_num_samples)
        sampled_y = sampled_indices // self.W
        sampled_x = sampled_indices % self.W
        sampled_xy = torch.stack([sampled_x, sampled_y], dim=1)

        new_features = self._sample_new_features(sampled_xy)
        new_cov2d = torch.rand(dynamic_num_samples, 3, device=self.device) + torch.tensor(
            [0.5, 0.0, 0.5], device=self.device
        )
        now_points, non_definite = self.model.densification_postfix(
            new_xyz=sampled_xy.float(),
            new_features_dc=new_features,
            new_cov2d=new_cov2d,
        )
        self.logwriter.write(
            f"iter:{iteration} add {dynamic_num_samples} points, filtered {non_definite}, current {now_points}"
        )
        return dynamic_num_samples

    def _restore_model_state(self, state_dict, slv_bound):
        self.model._xyz = nn.Parameter(state_dict["_xyz"])
        self.model._features_dc = nn.Parameter(state_dict["_features_dc"])
        self.model._cov2d = nn.Parameter(state_dict["_cov2d"])
        self.model._opacity = nn.Parameter(state_dict["_opacity"], requires_grad=False)
        self.model.gabor_freqs = nn.Parameter(state_dict["gabor_freqs"])
        self.model.gabor_weights = nn.Parameter(state_dict["gabor_weights"])
        self.model.load_state_dict(state_dict, strict=False)
        self.model.cholesky_bound = slv_bound
        self.model.cur_num_points = state_dict["_xyz"].shape[0]

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(H=self.H, W=self.W)
            reconstruction = outputs["render"]
            abundance = outputs["abundance"]
            endmember = outputs["endmember"]

        mse = F.mse_loss(reconstruction, self.gt_image).item()
        psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
        pred_hwc = reconstruction.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        sam = compute_sam(self.gt_cube, pred_hwc)
        return {
            "psnr": psnr,
            "sam": sam,
            "mse": mse,
            "reconstruction": pred_hwc,
            "abundance": abundance.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(),
            "endmember": endmember.detach().cpu().numpy(),
        }

    def save_results(self, best_state_dict, slv_bound, metrics):
        checkpoint_path = self.log_dir / "gaussian_model_hsi.pth.tar"
        torch.save(
            {
                "gs": best_state_dict,
                "num_gs": int(best_state_dict["_xyz"].shape[0]),
                "rank": self.rank,
                "spectral_channels": self.C,
                "slv_bound": slv_bound,
                "metrics": {"psnr": metrics["psnr"], "sam": metrics["sam"], "mse": metrics["mse"]},
            },
            checkpoint_path,
        )

        np.save(self.log_dir / "reconstruction.npy", metrics["reconstruction"].astype(np.float32))
        np.save(self.log_dir / "abundance.npy", metrics["abundance"].astype(np.float32))
        np.save(self.log_dir / "endmember.npy", metrics["endmember"].astype(np.float32))
        np.save(self.log_dir / "endmember_init.npy", self.model.E0.detach().cpu().numpy().astype(np.float32))

        with open(self.log_dir / "metrics.json", "w", encoding="utf-8") as fp:
            json.dump({"psnr": metrics["psnr"], "sam": metrics["sam"], "mse": metrics["mse"]}, fp, indent=2)

    def train(self):
        progress_bar = tqdm(range(1, self.args.iterations + 1), desc="HSI fitting")
        best_psnr = -float("inf")
        best_iter = 0
        best_model_dict = copy.deepcopy(self.model.state_dict())
        best_slv_bound = copy.deepcopy(self.model.cholesky_bound)

        self.model.train()
        torch.cuda.synchronize()
        start_time = time.time()

        for iteration in range(1, self.args.iterations + 1):
            loss, psnr, out_image, recon_loss, delta_norm = self.model.train_iter(
                self.H,
                self.W,
                self.gt_image,
                isprint=self.args.print,
                tv_weight=self.args.tv_weight,
                spectral_weight=self.args.spectral_weight,
                sparse_weight=self.args.sparse_weight,
                endmember_weight=self.args.endmember_weight,
            )

            with torch.no_grad():
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_iter = iteration
                    best_model_dict = copy.deepcopy(self.model.state_dict())
                    best_slv_bound = copy.deepcopy(self.model.cholesky_bound)

                if iteration % 100 == 0:
                    progress_bar.set_postfix(
                        {
                            "loss": f"{loss.item():.6f}",
                            "recon": f"{recon_loss.item():.6f}",
                            "psnr": f"{psnr:.4f}",
                            "best": f"{best_psnr:.4f}",
                            "deltaE": f"{delta_norm:.4f}",
                            "num": f"{self.model.cur_num_points}",
                        }
                    )
                    progress_bar.update(100)

                if self.args.prune and iteration % self.args.prune_iter == 0:
                    self.model.non_semi_definite_prune(self.H, self.W)

                if self.args.adaptive_add and iteration % self.args.grow_iter == 0 and iteration < self.args.iterations:
                    self.add_sample_positions(out_image, iteration)

        torch.cuda.synchronize()
        train_time = time.time() - start_time
        progress_bar.close()

        self._restore_model_state(best_model_dict, best_slv_bound)
        metrics = self.evaluate()
        self.save_results(best_model_dict, best_slv_bound, metrics)
        self.logwriter.write(
            f"training complete in {train_time:.2f}s, best_iter={best_iter}, psnr={metrics['psnr']:.4f}, sam={metrics['sam']:.4f}"
        )
        return metrics, train_time, best_iter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HSI full fitting with density-controlled Gabor++ Gaussian splatting")
    parser.add_argument("--input", type=str, required=True, help="HSI source: .mat, .npy, or per-band image directory")
    parser.add_argument("--mat_key", type=str, default=None, help="Key for .mat HSI data when multiple arrays exist")
    parser.add_argument("--channel_axis", type=int, default=None, help="Optional spectral axis index for .mat/.npy data")
    parser.add_argument("--output_root", type=str, default="./checkpoints_hsi")
    parser.add_argument("--rank", type=int, default=8, help="NMF rank / abundance channels")
    parser.add_argument("--lora_rank", type=int, default=2, help="LoRA rank for endmember correction")
    parser.add_argument("--lora_alpha", type=float, default=0.1, help="Maximum endmember correction scale")
    parser.add_argument("--freeze_endmember", action="store_true", help="Use E0 directly without LoRA correction")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--prune_iter", type=int, default=100)
    parser.add_argument("--grow_iter", type=int, default=5000)
    parser.add_argument("--num_points", type=int, default=2000)
    parser.add_argument("--max_num_points", type=int, default=5000)
    parser.add_argument("--opt_type", type=str, default="adam")
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--lr", type=float, default=0.018)
    parser.add_argument("--no_prune", action="store_true", help="Disable covariance pruning")
    parser.add_argument("--no_adaptive_add", action="store_true", help="Disable densification by error sampling")
    parser.add_argument("--loss_type", type=str, default="L2")
    parser.add_argument("--num_gabor", type=int, default=2)
    parser.add_argument("--tv_weight", type=float, default=0.0)
    parser.add_argument("--spectral_weight", type=float, default=0.0)
    parser.add_argument("--sparse_weight", type=float, default=0.0)
    parser.add_argument("--endmember_weight", type=float, default=0.0)
    parser.add_argument("--SLV_init", type=bool, default=True)
    parser.add_argument("--color_norm", action="store_true")
    parser.add_argument("--coords_norm", action="store_true")
    parser.add_argument("--coords_act", type=str, default="tanh")
    parser.add_argument("--clip_coe", type=float, default=3.0)
    parser.add_argument("--radius_clip", type=float, default=1.0)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--cov_quant", type=str, default="lsq")
    parser.add_argument("--color_quant", type=str, default="lsq")
    parser.add_argument("--xy_quant", type=str, default="lsq")
    parser.add_argument("--xy_bit", type=int, default=12)
    parser.add_argument("--cov_bit", type=int, default=10)
    parser.add_argument("--color_bit", type=int, default=6)

    args = parser.parse_args(argv)
    args.prune = not args.no_prune
    args.adaptive_add = not args.no_adaptive_add
    args.quantize = False
    return args


def main(argv=None):
    args = parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    trainer = HSIFullTrainer(args)
    config_path = trainer.log_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as fp:
        fp.write(yaml.safe_dump(vars(args), default_flow_style=False))

    metrics, train_time, best_iter = trainer.train()
    print(
        f"HSI fitting finished: best_iter={best_iter}, psnr={metrics['psnr']:.4f}, "
        f"sam={metrics['sam']:.4f}, time={train_time:.2f}s"
    )


if __name__ == "__main__":
    main(sys.argv[1:])