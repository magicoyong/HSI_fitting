import numpy as np
import scipy.io
import torch


DATASET_FILES = {
    "urban": "HSI/data/Urban_R162.mat",
    "salinas": "HSI/data/Salinas_crop.mat",
    "jasperridge": "HSI/data/jasperRidge2_R198.mat",
    "paviau": "HSI/data/PaviaU.mat",
}


def list_available_datasets():
    available = []
    for name, path in DATASET_FILES.items():
        try:
            scipy.io.loadmat(path)
            available.append(name)
        except FileNotFoundError:
            continue
    return available

def load_dataset(name):
    """Load dataset and corresponding endmember."""
    name = name.lower()
    if name == "urban":
        I = scipy.io.loadmat(DATASET_FILES["urban"])['Y'].astype(float)
        for i in range(162):
            denom = np.max(I[i, :])
            if denom > 0:
                I[i, :] /= denom
        I = I.reshape(162, 307, 307).transpose(2, 1, 0)

    elif name == "salinas":
        I = scipy.io.loadmat(DATASET_FILES["salinas"])['I'].astype(float)
        I = np.clip(I, 0, None)
        for i in range(204):
            denom = np.max(I[:, :, i])
            if denom > 0:
                I[:, :, i] /= denom

    elif name == "jasperridge":
        I = scipy.io.loadmat(DATASET_FILES["jasperridge"])['Y'].astype(float)
        for i in range(198):
            denom = np.max(I[i, :])
            if denom > 0:
                I[i, :] /= denom
        I = I.reshape(198, 100, 100).transpose(2, 1, 0)

    elif name == "paviau":
        I = scipy.io.loadmat(DATASET_FILES["paviau"])['paviaU'].astype(float)
        for i in range(103):
            denom = np.max(I[:, :, i])
            if denom > 0:
                I[:, :, i] /= denom
        I = I[-340:, :, :]

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    return I

def compute_sam(target_image, image) -> float:
    if torch.is_tensor(target_image) or torch.is_tensor(image):
        target = target_image if torch.is_tensor(target_image) else torch.as_tensor(target_image)
        pred = image if torch.is_tensor(image) else torch.as_tensor(image, device=target.device)
        target = target.to(dtype=torch.float32)
        pred = pred.to(dtype=torch.float32, device=target.device)

        dot = torch.sum(target * pred, dim=-1)
        norm_target = torch.linalg.norm(target, dim=-1)
        norm_pred = torch.linalg.norm(pred, dim=-1)
        cos_angle = dot / torch.clamp(norm_target * norm_pred, min=1e-8)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angles = torch.acos(cos_angle) * (180.0 / np.pi)
        return float(angles.mean().item())

    target = np.asarray(target_image, dtype=np.float32)
    pred = np.asarray(image, dtype=np.float32)
    dot = np.sum(target * pred, axis=-1)
    norm_target = np.linalg.norm(target, axis=-1)
    norm_pred = np.linalg.norm(pred, axis=-1)
    cos_angle = dot / np.clip(norm_target * norm_pred, 1e-8, None)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles = np.arccos(cos_angle) * (180.0 / np.pi)
    return float(np.mean(angles))


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    diff_h = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean()
    diff_w = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


def spectral_smoothness_loss(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] <= 1:
        return image.new_zeros(())
    return (image[:, 1:, :, :] - image[:, :-1, :, :]).abs().mean()
