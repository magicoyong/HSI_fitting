# LoR-SGS: Hyperspectral Image Compression via Low-rank Spectral Gaussian Splatting
### Requirements

```bash
cd gsplat
pip install .[dev]
cd ../
pip install -r requirements.txt
```

### Setup

Organize your files as follows:

```kotlin
HSI/
 ├── data/
 │    └── PaviaU.mat
 └── init/
```

The `.mat` file contains hyperspectral image data.

The estimated coefficient basis matrix file will be automatically generated in `HSI/init/`.

Configure hyperparameters in `train_compression.py` and `gaussianimage_cholesky_unknown.py`

### Run demo

Run the following commands sequentially to perform **NMF estimation** and **LoR-SGS training** on the *Pavia University* dataset:

```shell
python endmember.py --dataset paviau --rank 12
python main.py --dataset paviau --num_points 14500 --iterations 7000
```

## Acknowledgments

This implementation is developed based on the open-source project [GaussianImage](https://github.com/Xinjie-Q/GaussianImage), which provides the foundation for Gaussian splatting. We have modified and extended it for low-rank spectral modeling and hyperspectral image compression. We thank the original authors for their excellent work and for sharing their code.

## Citation

If you use our method or our code in your research, please kindly cite it:

```latex
@article{wang2025lorsgs,
  title={LoR-SGS: Hyperspectral Image Compression via Low-Rank Spectral Gaussian Splatting},
  author={Li, Tianyu and Wang, Ting and Zhao, Xile and Wang, Chao},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE}
}
```

