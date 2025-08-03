import scipy
import numpy as np
import scipy.linalg
from sklearn.decomposition import PCA, NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from VCA import vca
import time
 
start_time = time.time()
def var(data, n_endmembers):
    """
    Selects endmembers based on rows with the highest variance.

    :param data: Input 2D array (rows = samples, columns = features)
    :param n_endmembers: Number of endmembers to select
    :return: Array of selected endmembers
    """
    row_variances = np.var(data, axis=1)
    top_indices = np.argsort(row_variances)[-n_endmembers:] 
    endmembers = data[top_indices]
    return endmembers

def kmeans_initialization(data, rank):
    """
    Initialize endmembers using K-means clustering.
    
    Parameters:
    - data: np.ndarray of shape (num_pixels, num_bands), unfolded hyperspectral image.
    - rank: int, number of endmembers.
    
    Returns:
    - endmembers: np.ndarray of shape (rank, num_bands).
    """
    kmeans = KMeans(n_clusters=rank, random_state=42)
    kmeans.fit(data)
    endmembers = kmeans.cluster_centers_
    return endmembers

def knn_initialization(data, rank):
    """
    Initialize endmembers using K-Nearest Neighbors (KNN) approach.
    
    Parameters:
    - data: np.ndarray of shape (num_pixels, num_bands).
    - rank: int, number of endmembers.
    
    Returns:
    - endmembers: np.ndarray of shape (rank, num_bands).
    """
    nbrs = NearestNeighbors(n_neighbors=rank).fit(data)
    distances, indices = nbrs.kneighbors(data)
    endmembers = data[indices[:, 0], :]  # Selecting nearest points as endmembers
    return endmembers
'''
I = scipy.io.loadmat("HSI/data/jasperRidge2_R198.mat")['Y'].astype(float)
for i in range(198): 
    I[i,:] = I[i,:]/np.max(I[i,:])

I = scipy.io.loadmat("HSI/data/Urban_R162.mat")['Y'].astype(float)
for i in range(162): 
    I[i,:] = I[i,:]/np.max(I[i,:])

'''
I = scipy.io.loadmat("HSI/data/Salinas_crop.mat")['I'].astype(float)
I = np.clip(I, 0, None)
for i in range(204): 
    I[:,:,i] = I[:,:,i]/ np.max(I[:,:,i])
I = np.transpose(I, (2, 0, 1)).reshape(204, -1) 
'''
I = scipy.io.loadmat("HSI/data/PaviaU.mat")['paviaU'].astype(float)
for i in range(103): 
    I[:,:,i] = I[:,:,i] / np.max(I[:,:,i])
I = np.transpose(I, (2, 0, 1)).reshape(103, -1)   
''''''
nmf1 = NMF(16, init='random', random_state=42, max_iter=12000)

endmember = nmf1.fit_transform(I).T
A = nmf1.components_.T

np.save('HSI/init/Salinas_endmember_rank_16.npy',endmember)
np.save("HSI/init/Salinas_abundance_rank_16.npy",A)

# PCA 降维
pca = PCA(n_components=12)
A = pca.fit_transform(I.T)          
endmember = pca.components_                 

# 保存与 NMF 一致的形状 (rank, channels), (H*W, rank)
np.save('HSI/init/Salinas_endmember_rank_12_PCA.npy', endmember)
np.save("HSI/init/Salinas_abundance_rank_12_PCA.npy", A)

# SVD 分解
U, S, Vt = np.linalg.svd(I.T, full_matrices=False)  # U: (H*W, C), S: (C,), Vt: (C, C)

k = 12  # 设置 rank
A = U[:, :k] @ np.diag(S[:k])         # shape: (H*W, k)
endmember = Vt[:k, :]                 # shape: (k, C=204)

# 保存
np.save('HSI/init/Salinas_endmember_rank_12_SVD.npy', endmember)
np.save("HSI/init/Salinas_abundance_rank_12_SVD.npy", A)
'''

# === 2. 调用 VCA 提取 endmembers ===
rank = 12  # number of endmembers
endmember, indices = vca(I, rank)

# === 3. 计算 abundance (非负最小二乘回归) ===
# 输出 endmember 是 shape: (204, 16)
# 转置后为 shape: (16, 204)，与 NMF 一致
endmember_T = endmember.T
'''
# 使用最小二乘估计 abundance（非负可选）
from scipy.optimize import nnls
abundance = np.zeros((I.T.shape[1], rank))
for i in range(I.T.shape[1]):
    abundance[i], _ = nnls(endmember_T.T, I.T[:, i])
'''
# === 4. 保存结果 ===
np.save("HSI/init/Salinas_endmember_rank_12_VCA.npy", endmember_T)  # shape: (16, 204)
#np.save("HSI/init/Salinas_abundance_rank_12_VCA.npy", abundance)    # shape: (H*W, 16)

end_time = time.time()
run_time = end_time - start_time
print("Finished in", run_time, "sec")
