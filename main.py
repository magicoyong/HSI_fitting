import argparse
import numpy as np
import torch
import scipy.io
from sklearn.decomposition import NMF
from train_compression import train_nd


def get_dataset_rank(name):
    return 10 if name.lower() == "jasperridge" else 12


def load_dataset(name):
    """Load and normalize a hyperspectral dataset."""
    name = name.lower()
    if name == "urban":
        dataset_name = "Urban"
        I = scipy.io.loadmat("HSI/data/Urban_R162.mat")['Y'].astype(float)
        for i in range(162):
            I[i, :] /= np.max(I[i, :])
        I = I.reshape(162, 307, 307).transpose(2, 1, 0)

    elif name == "salinas":
        dataset_name = "Salinas"
        I = scipy.io.loadmat("HSI/data/Salinas_crop.mat")['I'].astype(float)
        I = np.clip(I, 0, None)
        for i in range(204):
            I[:, :, i] /= np.max(I[:, :, i])

    elif name == "jasperridge":
        dataset_name = "JasperRidge"
        I = scipy.io.loadmat("HSI/data/jasperRidge2_R198.mat")['Y'].astype(float)
        for i in range(198):
            I[i, :] /= np.max(I[i, :])
        I = I.reshape(198, 100, 100).transpose(2, 1, 0)

    elif name == "paviau":
        dataset_name = "PaviaU"
        I = scipy.io.loadmat("HSI/data/PaviaU.mat")['paviaU'].astype(float)
        for i in range(103):
            I[:, :, i] /= np.max(I[:, :, i])
        I = I[-340:, :, :]

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    return dataset_name, I.astype(np.float32)


def initialize_endmember(image, rank, dataset_name):
    print(f"Running inline NMF initialization for {dataset_name} with rank={rank}")
    matrix = np.transpose(image, (2, 0, 1)).reshape(image.shape[2], -1)
    nmf = NMF(n_components=rank, init="random", random_state=42, max_iter=12000)
    return nmf.fit_transform(matrix).T.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="LoR-SGS Hyperspectral Image Fitting")
    parser.add_argument('--dataset', type=str, default='JasperRidge',
                        help='Dataset name: Urban | Salinas | JasperRidge | PaviaU')
    parser.add_argument('--rank', type=int, default=None,
                        help='NMF rank; defaults to the dataset-specific preset when omitted')
    parser.add_argument('--iterations', type=int, default=8000,
                        help='Number of training iterations')
    parser.add_argument('--num_points', type=int, default=600,
                        help='Number of Gaussian points')
    args = parser.parse_args()

    rank = args.rank if args.rank is not None else get_dataset_rank(args.dataset)

    print(f"\n=== Training Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Rank: {rank}")
    print(f"Iterations: {args.iterations}")
    print(f"Number of Gaussian Points: {args.num_points}\n")

    dataset_name, I = load_dataset(args.dataset)
    E = initialize_endmember(I, rank, dataset_name)

    # Convert to tensor
    GT = torch.tensor(I)
    GT = GT.view(-1, GT.size(0), GT.size(1), GT.size(2)).permute(0, 3, 1, 2).contiguous()
    GT = torch.clamp(GT, 0, 1)

    # Run training
    train_nd(GT, endmember=E, iterations=args.iterations, num_points=args.num_points,
             model_name="GaussianImage_Cholesky_nd", image_name=dataset_name)


if __name__ == "__main__":
    main()