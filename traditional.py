import numpy as np
import cv2
import glymur
from sklearn.decomposition import PCA
import scipy.io 
from scipy.fftpack import dct, idct
import os
from PIL import Image
import numpy as np
import imagecodecs
from pytorch_msssim import ms_ssim
import time
import torch
import matplotlib.pyplot as plt
from utils import create_pseudorgb, compute_sam

def load_hsi_data():
    I = scipy.io.loadmat("HSI/data/Urban_R162.mat")['Y'].astype(float)
    for i in range(162): 
        I[i,:] = I[i,:]/np.max(I[i,:])
    I = I.reshape(162,307,307).transpose(2,1,0)
    return I

# 通用评估函数
def evaluate_compression(original, reconstructed, total_bits, name):
    height, width, bands = original.shape
    mse = np.mean((original - reconstructed) ** 2)
    '''
    pseudo_rgb = create_pseudorgb(reconstructed, bands=[10,90,180])
    plt.imsave(f"Salinas_{name}.png", pseudo_rgb)

    im = plt.imshow(np.mean((original - reconstructed) ** 2, axis = -1), cmap='jet', vmin=0, vmax=0.02)
    plt.axis('off')
    plt.savefig(f'./Residual/Salinas_{name}.png', bbox_inches='tight', pad_inches=0)
    '''
    psnr_value = 10 * np.log10((1.0 ** 2) / mse)  # 假设数据范围[0,1]
    ms_ssim_value = ms_ssim(torch.tensor(reconstructed).view(-1, height, width, bands).permute(0, 3, 1, 2).to(torch.float32), 
                            torch.tensor(original).view(-1, height, width, bands).permute(0, 3, 1, 2).to(torch.float32), 
                            data_range=1, size_average=True, win_size=7).item()
    
    # Per-band PSNR
    psnr_per_band = []
    for b in range(bands):
        band_mse = np.mean((original[:, :, b] - reconstructed[:, :, b]) ** 2)
        band_psnr = 10 * np.log10(1.0 / (band_mse + 1e-8))  # avoid div by 0
        psnr_per_band.append(band_psnr)
    psnr_per_band = np.array(psnr_per_band)  # shape: [bands]
    np.save(f"./PSNR/{name}.npy", psnr_per_band)
    ''''''
    mean_sam = compute_sam(original, reconstructed)
    
    bpppb = total_bits / (height * width * bands)
    return psnr_value, bpppb, ms_ssim_value, mean_sam

def jpeg_compression(original, quality=90):
    compressed_bands = []
    total_size = 0
    encoding_time = 0
    decoding_time = 0
    for i in range(original.shape[2]):
        # Scale to uint8 for image compression
        band = (original[..., i] * 255).astype(np.uint8)
        enc_start = time.time()
        # Compress each band using JPEG
        ret, buf = cv2.imencode('.jpg', band, [cv2.IMWRITE_JPEG_QUALITY, quality])
        enc_end = time.time()
        encoding_time += (enc_end - enc_start)
        if ret:
            total_size += len(buf)
            dec_start = time.time()
            # Decode the compressed data back to the image
            decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            compressed_bands.append(decoded)
            dec_end = time.time()
            decoding_time += (dec_end - dec_start)
        else:
            print(f"Error in compressing band {i}")
            return None, 0
    
    # Stack the compressed bands to reconstruct the image
    reconstructed = np.stack(compressed_bands, axis=2)
    
    return reconstructed, total_size * 8, encoding_time, decoding_time  # Return size in bits

def quantize_coeffs(coeffs, quant_step):
    """
    递归量化小波系数
    """
    if isinstance(coeffs, tuple):
        return tuple(quantize_coeffs(c, quant_step) for c in coeffs)
    elif isinstance(coeffs, np.ndarray):
        return np.round(coeffs / quant_step) * quant_step
    else:
        raise TypeError("Unsupported coefficient type")

def jpeg2000_compression(original, compression_ratio):
    compressed_bands = []
    total_size = 0
    encoding_time = 0
    decoding_time = 0
    for i in range(original.shape[2]):
        # Normalize and convert to uint8
        band = (original[..., i] * 255).astype(np.uint8)
        enc_start = time.time()
        # Compress using JPEG2000 with the specified compression ratio
        ret, buf = cv2.imencode('.jp2', band, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_ratio])
        enc_end = time.time()
        encoding_time += (enc_end - enc_start)
        # Add the size of the compressed buffer (in bytes) to total_size
        total_size += len(buf)
        
        # Decompress the image back to original size and scale back to [0, 1]
        dec_start = time.time()
        decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        compressed_bands.append(decoded)
        dec_end = time.time()
        decoding_time += (dec_end - dec_start)
    
    # Stack the compressed bands back into a 3D array (reconstructed image)
    reconstructed = np.stack(compressed_bands, axis=2)
    
    # Return the reconstructed image and the total size in bits
    return reconstructed, total_size * 8, encoding_time, decoding_time  # Convert size from bytes to bits

# PCA-DCT压缩
def pca_dct_compression(hsi, n_components, dct_quant=0.5):
    height, width, bands = hsi.shape
    encoding_start = time.time()
    # 将 HSI 数据展平为 (height * width, bands)
    X = hsi.reshape(-1, bands)
    
    
    # PCA 阶段
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)  # 形状为 (height * width, n_components)
    
    
    # DCT 压缩
    compressed_coeffs = []
    for i in range(n_components):
        band = X_pca[:, i].reshape(height, width)
        coeffs = dct(dct(band, axis=0, norm='ortho'), axis=1, norm='ortho')
        quantized = np.round(coeffs / dct_quant)
        compressed_coeffs.append(quantized)
    encoding_end = time.time()
    
    # 计算存储需求
    pca_bits = (pca.mean_.nbytes + pca.components_.nbytes) * 8
    coeff_bits = sum([(c.size * 16) for c in compressed_coeffs])  # 假设 16bit/coeff
    total_bits = pca_bits + coeff_bits
    
    decoding_start = time.time()
    # 解压缩
    reconstructed_coeffs = [c * dct_quant for c in compressed_coeffs]
    reconstructed_pca = []
    for coeff in reconstructed_coeffs:
        band = idct(idct(coeff, axis=0, norm='ortho'), axis=1, norm='ortho')
        reconstructed_pca.append(band)
    
    # 将 DCT 重建的数据重新堆叠
    X_pca_reconstructed = np.stack(reconstructed_pca, axis=2)  # 形状为 (height * width, n_components)
    
    # PCA 逆变换
    X_rec = pca.inverse_transform(X_pca_reconstructed)  # 形状为 (height * width, bands)
    
    # 恢复原始形状
    reconstructed = X_rec.reshape(height, width, bands)
    decoding_end = time.time()

    encoding_time = encoding_end - encoding_start
    decoding_time = decoding_end - decoding_start
    return reconstructed, total_bits, encoding_time, decoding_time

# 主测试流程
original = load_hsi_data()
height, width, bands = original.shape
print(f'Original Image Shape: {original.shape}')
'''
pseudo_rgb = create_pseudorgb(original, bands=[10,90,180])
plt.imsave(f"Salinas.png", pseudo_rgb)
''''''
im = plt.imshow(np.mean((original - original) ** 2, axis = -1), cmap='jet', vmin=0, vmax=0.01)
bar = plt.colorbar(im,
                   fraction=0.046,    # 原例中的 fraction
                   pad=0.04,          # 原例中的 pad
                   shrink=0.8,        # 将长度缩短到原来的 60%
                   aspect=12)         # 将厚度放大

# 3. 放大刻度字体
bar.set_ticks(np.linspace(0, 0.01,3))
bar.ax.tick_params(labelsize=16)   # 或者更大，视图需求调整
plt.axis('off')
plt.savefig('./Residual/PaviaU_GT.png', bbox_inches='tight', pad_inches=0)
bar.remove()

'''
# 测试不同压缩方法
methods = {
    'JPEG': [jpeg_compression, {'quality': [1]}],#
    'JPEG2000': [jpeg2000_compression, {'compression_ratio': [13]}],
    'PCA-DCT': [pca_dct_compression, {'n_components': [1]}]
}
''''''

results = {}

for method_name, (func, params) in methods.items():
    print(f'Testing {method_name}...')
    results[method_name] = {'psnr': [], 'bpppb': [], 'SAM':[]}
    
    # 参数网格搜索
    for param in params.values():
        for p in param:
            kwargs = {k: p if k in params.keys() else None for k in params.keys()}
            reconstructed, bits, enc_time, dec_time = func(original, **kwargs)
            psnr_val, bpppb_val, msssim, sam = evaluate_compression(original, reconstructed, bits, method_name)
            results[method_name]['psnr'].append(psnr_val)
            results[method_name]['bpppb'].append(bpppb_val)
            results[method_name]['SAM'].append(sam)
            print(f'Param: {p} | Encoding Time: {enc_time:.4f}s | Decoding Time: {dec_time:.4f}s | {bpppb_val:4f}, {psnr_val:4f}, {sam:4f}')
