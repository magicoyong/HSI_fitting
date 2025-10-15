import argparse
import numpy as np
import scipy.io
from sklearn.decomposition import NMF
import time
import os

def load_dataset(name):
    """Load and normalize hyperspectral dataset."""
    name = name.lower()
    if name == "salinas":
        data = scipy.io.loadmat("HSI/data/Salinas_crop.mat")['I'].astype(float)
        data = np.clip(data, 0, None)
        for i in range(204):
            data[:, :, i] /= np.max(data[:, :, i])
    elif name == "urban":
        data = scipy.io.loadmat("HSI/data/Urban_R162.mat")['Y'].astype(float)
        for i in range(162):
            data[i, :] /= np.max(data[i, :])
        data = data.reshape(162, 307, 307).transpose(2, 1, 0)
    elif name == "jasperridge":
        data = scipy.io.loadmat("HSI/data/jasperRidge2_R198.mat")['Y'].astype(float)
        for i in range(198):
            data[i, :] /= np.max(data[i, :])
        data = data.reshape(198, 100, 100).transpose(2, 1, 0)
    elif name == "paviau":
        data = scipy.io.loadmat("HSI/data/PaviaU.mat")['paviaU'].astype(float)
        for i in range(103):
            data[:, :, i] /= np.max(data[:, :, i])
        data = data[-340:, :, :]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # (C, H, W) → (C, H*W)
    data = np.transpose(data, (2, 0, 1)).reshape(data.shape[2], -1)
    return data


def nmf_initialization(I, rank, dataset_name):
    """Perform NMF initialization."""
    print(f"Running NMF initialization on {dataset_name} with rank={rank}")
    nmf = NMF(rank, init='random', random_state=42, max_iter=12000)
    endmember = nmf.fit_transform(I).T  # shape (rank, channels)
    abundance = nmf.components_.T       # shape (H*W, rank)

    os.makedirs("HSI/init", exist_ok=True)
    np.save(f"HSI/init/{dataset_name}_endmember_rank_{rank}_NMF.npy", endmember)
    np.save(f"HSI/init/{dataset_name}_abundance_rank_{rank}_NMF.npy", abundance)
    print(f"Saved NMF initialization results for {dataset_name}.")


def main():
    parser = argparse.ArgumentParser(description="NMF initialization for hyperspectral endmembers.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name: Salinas | Urban | JasperRidge | PaviaU")
    parser.add_argument("--rank", type=int, default=12,
                        help="Number of endmembers (rank)")
    args = parser.parse_args()

    start_time = time.time()
    I = load_dataset(args.dataset)
    nmf_initialization(I, args.rank, args.dataset)
    print(f"Finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
