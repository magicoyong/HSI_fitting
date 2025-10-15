import argparse
import numpy as np
import torch
import scipy.io
from train_compression import train_nd


def load_dataset(name):
    """Load dataset and corresponding endmember."""
    name = name.lower()
    if name == "urban":
        E = np.load('HSI/init/Urban_endmember_rank_12.npy').astype(np.float32)
        I = scipy.io.loadmat("HSI/data/Urban_R162.mat")['Y'].astype(float)
        for i in range(162):
            I[i, :] /= np.max(I[i, :])
        I = I.reshape(162, 307, 307).transpose(2, 1, 0)

    elif name == "salinas":
        E = np.load('HSI/init/Salinas_endmember_rank_12.npy').astype(np.float32)
        I = scipy.io.loadmat("HSI/data/Salinas_crop.mat")['I'].astype(float)
        I = np.clip(I, 0, None)
        for i in range(204):
            I[:, :, i] /= np.max(I[:, :, i])

    elif name == "jasperridge":
        E = np.load('HSI/init/JR_endmember_rank_10.npy').astype(np.float32)
        I = scipy.io.loadmat("HSI/data/jasperRidge2_R198.mat")['Y'].astype(float)
        for i in range(198):
            I[i, :] /= np.max(I[i, :])
        I = I.reshape(198, 100, 100).transpose(2, 1, 0)

    elif name == "paviau":
        E = np.load('HSI/init/PaviaU_endmember_rank_12.npy').astype(np.float32)
        I = scipy.io.loadmat("HSI/data/PaviaU.mat")['paviaU'].astype(float)
        for i in range(103):
            I[:, :, i] /= np.max(I[:, :, i])
        I = I[-340:, :, :]

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    return E, I


def main():
    parser = argparse.ArgumentParser(description="LoR-SGS Hyperspectral Image Compression")
    parser.add_argument('--dataset', type=str, default='JasperRidge',
                        help='Dataset name: Urban | Salinas | JasperRidge | PaviaU')
    parser.add_argument('--iterations', type=int, default=8000,
                        help='Number of training iterations')
    parser.add_argument('--num_points', type=int, default=600,
                        help='Number of Gaussian points')
    args = parser.parse_args()

    # 固定 rank（例如 Urban/Salinas/PaviaU 用 12, JasperRidge 用 10）
    dataset = args.dataset.lower()
    if dataset == "jasperridge":
        rank = 10
    else:
        rank = 12

    print(f"\n=== Training Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Rank: {rank}")
    print(f"Iterations: {args.iterations}")
    print(f"Number of Gaussian Points: {args.num_points}\n")

    # Load data and endmember
    E, I = load_dataset(args.dataset)

    # Convert to tensor
    GT = torch.tensor(I)
    GT = GT.view(-1, GT.size(0), GT.size(1), GT.size(2)).permute(0, 3, 1, 2).contiguous()
    GT = torch.clamp(GT, 0, 1)

    # Run training
    train_nd(GT, endmember=E, iterations=args.iterations, num_points=args.num_points,
             model_name="GaussianImage_Cholesky_nd", image_name=args.dataset)


if __name__ == "__main__":
    main()