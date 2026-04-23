import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim,ssim
import sys
from PIL import Image, ImageOps
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import wandb
import copy
import json
from models.utils import loss_fn
from models.gaussianimage_covariance import GaussianImage_Covariance
from endmember import masked_nmf_initialization, nmf_initialization


BUILTIN_HSI_DATASETS = {
    "urban": "Urban",
    "salinas": "Salinas",
    "jasperridge": "JasperRidge",
    "paviau": "PaviaU",
}

MULTISPECTRAL_SCENES = {
    "beads_ms",
    "chart_and_stuffed_toy_ms",
    "feathers_ms",
    "flowers_ms",
}

MS_BAND_PATTERN = re.compile(r".+_ms_(\d+)\.png$", re.IGNORECASE)


def _normalize_band_to_unit_interval(band):
    band = band.astype(np.float32)
    band_min = float(band.min())
    band_max = float(band.max())
    if band_max > band_min:
        return (band - band_min) / (band_max - band_min)
    return np.zeros_like(band, dtype=np.float32)


def _load_builtin_hsi_dataset(name):
    """Load built-in HSI dataset. Returns (H, W, C) float array."""
    name = name.lower()
    if name == "urban":
        I = scipy.io.loadmat("HSI/data/Urban_R162.mat")['Y'].astype(float)
        for i in range(162):
            I[i, :] /= np.max(I[i, :])
        I = I.reshape(162, 307, 307).transpose(2, 1, 0)  # (H, W, C)
    elif name == "salinas":
        I = scipy.io.loadmat("HSI/data/Salinas_crop.mat")['I'].astype(float)
        I = np.clip(I, 0, None)
        for i in range(I.shape[2]):
            I[:, :, i] /= np.max(I[:, :, i])
    elif name == "jasperridge":
        I = scipy.io.loadmat("HSI/data/jasperRidge2_R198.mat")['Y'].astype(float)
        for i in range(198):
            I[i, :] /= np.max(I[i, :])
        I = I.reshape(198, 100, 100).transpose(2, 1, 0)  # (H, W, C)
    elif name == "paviau":
        I = scipy.io.loadmat("HSI/data/PaviaU.mat")['paviaU'].astype(float)
        for i in range(103):
            I[:, :, i] /= np.max(I[:, :, i])
        # I = I[-340:, :, :]
    else:
        raise ValueError(f"Unknown HSI dataset: {name}")
    return I.astype(np.float32)


def _load_multispectral_scene_dir(scene_dir):
    """Load a directory of *_ms_XX.png bands into an (H, W, C) float32 cube."""
    scene_dir = Path(scene_dir)
    band_files = []
    for path in scene_dir.iterdir():
        match = MS_BAND_PATTERN.fullmatch(path.name)
        if match is not None:
            band_files.append((int(match.group(1)), path))

    if not band_files:
        raise ValueError(f"No multispectral band files found in: {scene_dir}")

    band_files.sort(key=lambda item: item[0])

    bands = []
    expected_size = None
    for _, band_path in band_files:
        band = np.asarray(Image.open(band_path))
        if band.ndim != 2:
            raise ValueError(
                f"Expected single-channel band image, got shape {band.shape} for {band_path}"
            )
        if expected_size is None:
            expected_size = band.shape
        elif band.shape != expected_size:
            raise ValueError(
                f"Band size mismatch in {scene_dir}: expected {expected_size}, got {band.shape} for {band_path.name}"
            )
        bands.append(_normalize_band_to_unit_interval(band))

    return np.stack(bands, axis=-1).astype(np.float32)


def resolve_hsi_dataset(dataset):
    """Resolve --dataset to either a built-in HSI name or a multispectral scene directory."""
    dataset_str = str(dataset).strip()
    dataset_path = Path(dataset_str).expanduser()
    dataset_key = dataset_str.lower()

    if dataset_path.is_dir():
        return {
            "kind": "multispectral_dir",
            "path": dataset_path,
            "label": dataset_path.name,
            "display": dataset_str,
        }

    if dataset_key in BUILTIN_HSI_DATASETS:
        return {
            "kind": "builtin_hsi",
            "name": dataset_key,
            "label": BUILTIN_HSI_DATASETS[dataset_key],
            "display": BUILTIN_HSI_DATASETS[dataset_key],
        }

    if dataset_key in MULTISPECTRAL_SCENES:
        scene_dir = Path("HSI") / dataset_key
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Multispectral scene directory not found: {scene_dir}")
        return {
            "kind": "multispectral_dir",
            "path": scene_dir,
            "label": scene_dir.name,
            "display": dataset_key,
        }

    supported = list(BUILTIN_HSI_DATASETS.values()) + sorted(MULTISPECTRAL_SCENES)
    raise ValueError(
        "Unknown dataset. Supported built-ins/scenes: "
        + ", ".join(supported)
        + "; or pass a directory path."
    )


def load_hsi_dataset(name):
    """Load and normalize HSI dataset. Returns (H, W, C) numpy array."""
    resolved = resolve_hsi_dataset(name)
    if resolved["kind"] == "builtin_hsi":
        return _load_builtin_hsi_dataset(resolved["name"])
    return _load_multispectral_scene_dir(resolved["path"])

def compute_sam(gt, pred):
    """Compute Spectral Angle Mapper (mean SAM in degrees)."""
    # gt, pred: (H, W, C) numpy arrays
    dot = np.sum(gt * pred, axis=-1)
    norm_gt = np.linalg.norm(gt, axis=-1)
    norm_pred = np.linalg.norm(pred, axis=-1)
    cos_angle = dot / (norm_gt * norm_pred + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    angles = np.arccos(cos_angle) * 180 / np.pi
    return np.mean(angles)


class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""

    def __init__(self, args):
        self.device = torch.device("cuda:0")
        self.args = args
        self.dataset_info = resolve_hsi_dataset(args.dataset)
        self.dataset_label = self.dataset_info["label"]

        # ── Load HSI data ───────────────────────────────────────────────
        I_np = load_hsi_dataset(args.dataset)  # (H, W, C)
        self.H, self.W, self.C = I_np.shape
        self.gt_image = torch.tensor(I_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        self.gt_image = torch.clamp(self.gt_image, 0, 1).to(self.device)

        E0, A0 = nmf_initialization(self.gt_image, args.rank )
        self.rank = E0.shape[0]
        assert E0.shape[1] == self.C

        # ── Output directory ────────────────────────────────────────────
        calib_tag = (
            f"calib{args.calib_rank}_g{args.gamma}"
            if not args.freeze_endmember_calibration
            else "E0only"
        )
        self.log_dir = Path(
            f"./checkpoints_hsi/{self.dataset_label}/"
            f"GaborHSI_{args.iterations}_{args.num_points}_{args.num_gabor}"
            f"_rank{args.rank}_{calib_tag}"
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logwriter = LogWriter(self.log_dir)

        self.max_num_points = args.max_num_points
        BLOCK_H, BLOCK_W = 16, 16
        from gaborimage_cholesky_hsi import GaborImage_Cholesky_HSI
        self.model = GaborImage_Cholesky_HSI(
            loss_type=args.loss_type,
            opt_type=getattr(args, 'opt_type', 'adan'),
            num_points=args.num_points,
            H=self.H, W=self.W,
            rank=self.rank, C=self.C,
            E=E0,
            calib_rank=args.calib_rank,
            gamma=args.gamma,
            freeze_endmember_calibration=args.freeze_endmember_calibration,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            device=self.device,
            lr=args.lr,
            num_gabor=args.num_gabor,
            quantize=args.quantize,
        ).to(self.device)

        self.add_stage = 0
        self.print = args.print
        
    def add_sample_positions(self, render_image, iter=0):

        errors = torch.abs(render_image - self.gt_image).sum(dim=1)

        normalized_gradient = errors / torch.sum(errors)

        base_num_samples = 1000

        if iter == self.iterations - self.args.grow_iter:
            dynamic_num_samples = max(0, self.max_num_points - self.gaussian_model.cur_num_points)  # 最大采样点数量限制
        else:
            dynamic_num_samples = max(0, min(base_num_samples,
                                             self.max_num_points - self.gaussian_model.cur_num_points))
        if dynamic_num_samples:
            P_flat = normalized_gradient.view(-1)
            _, sampled_indices_1d = torch.topk(P_flat, dynamic_num_samples)

            # 将一维索引转换为二维坐标
            sampled_y = sampled_indices_1d // self.W
            sampled_x = sampled_indices_1d % self.W
            color = torch.zeros(dynamic_num_samples, 3, device=self.device)  #
            new_points_nums = sampled_y.shape[0]
            sampled_xyz = torch.stack([sampled_x, sampled_y], dim=1)
            new_attributes = {"new_xyz": sampled_xyz.float().to(self.device),
                              "new_features_dc": color.float(),
                              "new_cov2d": torch.rand(new_points_nums, 3).to(self.device) + torch.tensor(
                                  [0.5, 0, 0.5]).to(self.device)
                              }

            now_points_nums, non_definite = self.gaussian_model.densification_postfix(**new_attributes)
            self.logwriter.write(
                f"\n iter:{iter} , Add {dynamic_num_samples} new points But {non_definite} non-defite; cur {now_points_nums} points")
            return dynamic_num_samples
        return 0

    def train(self):
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training progress")
        best_psnr = 0

        self.gaussian_model.train()
        torch.cuda.synchronize()
        start_time = time.time()

        for iter in range(1, self.iterations + 1):
            loss, psnr, out_image = self.gaussian_model.train_iter(self.H, self.W, self.gt_image, isprint=self.print)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if best_psnr < psnr:
                    best_psnr = psnr
                    # if iter % 1000 == 0:
                    best_model_dict = copy.deepcopy(self.gaussian_model.state_dict())
                    slv_bound = copy.deepcopy(self.gaussian_model.cholesky_bound)
                    best_iter = iter

                if iter % 1000 == 0:
                    progress_bar.set_postfix(
                        {f"Loss": f"{loss.item():.{7}f}", "PSNR": f"{psnr:.{4}f}", "Best PSNR": f"{best_psnr:.{4}f}",
                         "best_iter": f"{best_iter}"})
                    progress_bar.update(1000)

                if iter % self.args.prune_iter == 0:
                    none_definite, cur_gs_nums = self.gaussian_model.non_semi_definite_prune(self.H, self.W)

                if self.args.adaptive_add:
                    if iter % self.args.grow_iter == 0 and iter < self.iterations:
                        new_added_points_nums = self.add_sample_positions(out_image, iter=iter)

        torch.cuda.synchronize()
        end_time = time.time() - start_time
        progress_bar.close()

        self.gaussian_model._xyz = nn.Parameter(best_model_dict['_xyz'])
        self.gaussian_model._features_dc = nn.Parameter(best_model_dict['_features_dc'])
        self.gaussian_model._cov2d = nn.Parameter(best_model_dict['_cov2d'])
        self.gaussian_model._opacity = nn.Parameter(best_model_dict['_opacity'])
        self.gaussian_mode        new_gabor_freqs, new_gabor_weights = self._init_gabor_params(int(valid_mask.sum().item()))l.gabor_freqs = nn.Parameter(best_model_dict['gabor_freqs'])
        self.gaussian_model.gabor_weights = nn.Parameter(best_model_dict['gabor_weights'])
        self.gaussian_model.load_state_dict(best_model_dict)
        self.gaussian_model.cholesky_bound = slv_bound
        self.gaussian_model.cur_num_points = best_model_dict['_xyz'].shape[0]

        # none_definite,cur_gs_nums=self.gaussian_model.non_semi_definite_prune(self.H,self.W)
        psnr_value, ms_ssim_value, test_end_time, FPS = self.test()

        total_params = sum([p.numel() for p in self.gaussian_model.parameters() if p.requires_grad]) / 1e6
        self.logwriter.write(
            "Training Complete in {:.4f}, Eval time:{:.8f}, FPS:{:.4f}, gs_nums:{:.2e},gs_params:{:.4f}M".format(
                end_time, test_end_time, FPS, self.gaussian_model.cur_num_points, total_params))
        torch.save({"gs": best_model_dict, "num_gs": self.gaussian_model.cur_num_points,
                    "psnr": best_psnr, "ms-ssim": ms_ssim_value, "slv_bound": slv_bound},
                   self.log_dir / "gaussian_model.pth.tar")
        return psnr_value, ms_ssim_value, end_time, test_end_time, FPS

    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model(H=self.H, W=self.W)
            torch.cuda.synchronize()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model(H=self.H, W=self.W)
            torch.cuda.synchronize()
            test_end_time = (time.time() - test_start_time) / 100
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        return (psnr, ms_ssim_value, test_end_time, 1 / test_end_time)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )

    parser.add_argument(
        "--data_name", type=str, default='kodak', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--prune_iter", type=int, default=100, help="iteration of each pruning  (default: %(default)s)"
    )
    parser.add_argument(
        "--grow_iter", type=int, default=5000, help="iteration of each growing (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Covariance",
        help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )

    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument("--num_points", type=int, default=2500, help="2D GS points (default: %(default)s)")
    parser.add_argument("--max_num_points", type=int, default=5000, help="max 2D GS points (default: %(default)s)", )
    parser.add_argument("--opt_type", type=str, default="adam", help="the type of optimizer")
    parser.add_argument("-opt", "--opt_nums", type=int, default=1, help="the nums of optimizer")
    parser.add_argument("--model_path", type=str,  default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=int, default=3047, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument("--print", type=bool, default=False, help="if need print details")
    parser.add_argument("--lr", type=float, default=0.018,help="Learning rate (default: %(default)s)")
    parser.add_argument("--warmup_iter", type=float, default=15000)
    parser.add_argument('--radius_clip', type=float, default=1.0)
    parser.add_argument("--prune", type=bool, default=True,  help="turn on pruning")
    parser.add_argument("--adaptive_add", type=bool, default=True, help="turn on adaptive add densification")
    parser.add_argument("--wandb-project", type=str, default=None, help='Weights & Biases Project')
    parser.add_argument("--loss_type", type=str, default="L2")
    parser.add_argument("--SLV_init", type=bool, default=True, help="if turn on CAF filter")
    parser.add_argument("--color_norm",action='store_true')
    parser.add_argument("--coords_norm", action='store_true', help="if normalize the coordinates")
    parser.add_argument("--coords_act", type=str, default="tanh", help="tanh")
    parser.add_argument("--save_interval", type=int, default=5, help="save interval")
    parser.add_argument("--clip_coe", type=float, default=3.)
    # Gabor Parameter
    parser.add_argument("--num_gabor", type=int, default=2)
    #  quantization parameters =======================================
    parser.add_argument("--quantize", type=bool, default=False, help="Quantize")
    parser.add_argument("--cov_quant", type=str, default="lsq", help="type of covariance quantization")
    parser.add_argument("--color_quant", type=str, default="lsq")
    parser.add_argument("--xy_quant", type=str, default="lsq")
    parser.add_argument("--xy_bit", type=int, default=12, help="bitdepth of xy attri")
    parser.add_argument("--cov_bit", type=int, default=10, help="bitdepth of cov attri")
    parser.add_argument("--color_bit", type=int, default=6, help="bitdepth of color attri")

    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Save config
    args_text = yaml.safe_dump(vars(args), default_flow_style=False)
    print(f"\n=== HSI Inpainting Configuration ===")
    print(args_text)

    trainer = SimpleTrainer2d(args)

    # Save config to log dir
    config_path = trainer.log_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(args_text)

    psnr, sam, ssim, training_time = trainer.train()

    print(f"\n=== HSI Inpainting Result ===")
    print(f"Dataset: {args.dataset}, Rank: {args.rank}")
    print(f"Mask: {args.mask_type}, ratio={args.mask_ratio}")
    print(f"Calibration: freeze={args.freeze_endmember_calibration}, "
          f"calib_rank={args.calib_rank}, gamma={args.gamma}")
    print(f"PSNR(masked vs gt):    {trainer.psnr_masked:.4f}  (degradation baseline)")
    print(f"Best PSNR(recon vs gt): {psnr:.4f}")
    print(f"Best SAM:               {sam:.4f}")
    print(f"Best SSIM:              {ssim:.4f}")
    print(f"Training Time:          {training_time:.2f}s")

if __name__ == "__main__":
    main(sys.argv[1:])
