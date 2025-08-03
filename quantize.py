from vector_quantize_pytorch import *

import torch
from torch import nn
from utils import *
import torch.nn.functional as F
import constriction
import numpy as np

def grad_scale(x, scale):
    return (x - x * scale).detach() + x * scale

def ste(x):
    return (x.round() - x).detach() + x

class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x):
        return x.half().float()

    @staticmethod
    def backward(_, grad_output):
        return grad_output

class UniformQuantizer(nn.Module):
    def __init__(self, signed=False, bits=8, learned=False, num_channels=1, entropy_type="none", weight=0.001):
        super().__init__()
        if signed:
            self.qmin = -2**(bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1

        self.learned = learned
        self.entropy_type = entropy_type
        if self.learned:
            self.scale = nn.Parameter(torch.ones(num_channels)/self.qmax, requires_grad=True) # gamma
            self.beta = nn.Parameter(torch.ones(num_channels)/self.qmax, requires_grad=True) # beta

        self.weight = weight

    def _init_data(self, tensor):
        device = tensor.device
        t_min, t_max = tensor.min(dim=0)[0], tensor.max(dim=0)[0] # 输入的最大值和最小值
        scale = (t_max - t_min) / (self.qmax-self.qmin)
        self.beta.data = t_min.to(device)
        self.scale.data = scale.to(device)

    def forward(self, x):
        if self.learned:
            grad = 1.0 / ((self.qmax * x.numel()) ** 0.5) 
            s_scale = grad_scale(self.scale, grad)
            beta_scale = grad_scale(self.beta, grad)
            s_scale, beta_scale = self.scale, self.beta
            code = ((x - beta_scale) / s_scale).clamp(self.qmin, self.qmax)
            quant = ste(code)
            dequant = quant * s_scale + beta_scale
        else:
            code = (x * self.qmax).clamp(self.qmin, self.qmax)
            quant = ste(code)
            dequant = quant / self.qmax

        bits, entropy_loss = 0, 0
        if not self.training:
            num_points, num_channels = x.shape
            bits = self.size(quant)
            # unit_bit = bits / num_points / num_channels
        return dequant, entropy_loss*self.weight, bits # 编码又解码后数值，熵损失，压缩后大小（位数）

    def size(self, quant):
        index_bits = 0
        compressed, histogram_table, unique = compress_matrix_flatten_categorical(quant.int().flatten().tolist())
        index_bits += (get_np_size(compressed) * 8 + 64)
        index_bits += get_np_size(histogram_table) * 8
        index_bits += get_np_size(unique) * 8 
        index_bits += self.scale.numel()*torch.finfo(self.scale.dtype).bits # gamma字节数
        index_bits += self.beta.numel()*torch.finfo(self.beta.dtype).bits # beta字节数
        return index_bits

    def compress(self, x): # 压缩
        code = ((x - self.beta) / self.scale).clamp(self.qmin, self.qmax)
        return code.round(), code.round()* self.scale + self.beta

    def decompress(self, x): # 解压
        return x * self.scale + self.beta

class VectorQuantizer(nn.Module):
    def __init__(self, num_quantizers=1, codebook_dim=1, codebook_size=64, kmeans_iters=10, vector_type="vector"):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.vector_type = vector_type
        if self.num_quantizers == 1:
            if self.vector_type == "vector":
                self.quantizer = VectorQuantize(dim=codebook_dim, codebook_size=codebook_size, decay = 0.8, commitment_weight = 1., kmeans_init = True, 
                    kmeans_iters = kmeans_iters,heads = 4, separate_codebook_per_head = True)#learnable_codebook=True, ema_update = False, orthogonal_reg_weight =1.)
        else:
            if self.vector_type == "vector":
                self.quantizer = ResidualVQ(dim=codebook_dim, codebook_size=codebook_size, num_quantizers=num_quantizers, commitment_weight = 1., decay =0.4, kmeans_init=True,
                    kmeans_iters = kmeans_iters) 
                # orthogonal_reg_weight=0., in_place_codebook_optimizer=torch.optim.Adam)

    def forward(self, x):
        if self.training:
            x, _, l_vq = self.quantizer(x)
            l_vq = torch.sum(l_vq)
            return x, l_vq, 0
        else:
            num_points, num_channels = x.shape
            x, embed_index, l_vq = self.quantizer(x)
            l_vq = torch.sum(l_vq)
            bits = self.size(embed_index)
            # unit_bit = bits / num_points / num_channels
            return x, l_vq, bits

    def size(self, embed_index):
        if self.num_quantizers == 1:
            if self.vector_type == "vector":
                codebook_bits = self.quantizer._codebook.embed.numel()*torch.finfo(self.quantizer._codebook.embed.dtype).bits
            elif self.vector_type == "ste":
                codebook_bits = self.quantizer.embedding.weight.data.numel()*torch.finfo(self.quantizer.embedding.weight.data.dtype).bits
            index_bits = 0
            compressed, histogram_table, unique = compress_matrix_flatten_categorical(embed_index.int().flatten().tolist())
            index_bits += get_np_size(compressed) * 8
            index_bits += get_np_size(histogram_table) * 8
            index_bits += get_np_size(unique) * 8  
        elif hasattr(self.quantizer, "rvqs"):
            codebook_bits = 0
            # Iterate over each ResidualVQ in the group
            for rvq in self.quantizer.rvqs:
                for quantizer_index, layer in enumerate(rvq.layers):
                    if self.vector_type == "vector":
                        codebook_bits += layer._codebook.embed.numel() * torch.finfo(layer._codebook.embed.dtype).bits
                    elif self.vector_type == "ste":
                        codebook_bits += layer.embedding.weight.data.numel() * torch.finfo(layer.embedding.weight.data.dtype).bits
            compressed, histogram_table, unique = compress_matrix_flatten_categorical(
                embed_index.int().flatten().tolist()
            )
            index_bits = (get_np_size(compressed) * 8 +
                      get_np_size(histogram_table) * 8 +
                      get_np_size(unique) * 8)
        # Add a fixed overhead (e.g., 64 bits) per group
            num_groups = len(self.quantizer.rvqs)
            index_bits += num_groups * 64
        else:
            codebook_bits, index_bits = 0, 0
            for quantizer_index, layer in enumerate(self.quantizer.layers):
                if self.vector_type == "vector":
                    codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
                elif self.vector_type == "ste":
                    codebook_bits += layer.embedding.weight.data.numel()*torch.finfo(layer.embedding.weight.data.dtype).bits
            compressed, histogram_table, unique = compress_matrix_flatten_categorical(embed_index.int().flatten().tolist())
            index_bits += (get_np_size(compressed) * 8 + 64)
            index_bits += get_np_size(histogram_table) * 8
            index_bits += get_np_size(unique) * 8  
        
        total_bits = codebook_bits + index_bits
        #print("vq:", embed_index.shape, codebook_bits, index_bits)
        return total_bits

    def compress(self, x):
        x, embed_index, _ = self.quantizer(x)
        return x, embed_index

    def decompress(self, embed_index):
        recon = 0
        for i,layer in enumerate(self.quantizer.layers):
            recon += layer._codebook.embed[0, embed_index[:, i]]
        return recon

class SimpleLearnedScalarQuantizer(nn.Module):
    """
    轻量级学习标量量化器，用于对 feature 参数（形状为 [num_points, rank]）进行量化。
    
    工作流程：
      1. 对输入进行学习的归一化（使用 per-dimension 的 scale 和 beta）。
      2. 将归一化后的值按 [0, 1] 区间映射到 [0, num_levels-1] 上，并使用 STE 进行离散化，
         得到整数编码（每个元素使用 log2(num_levels) 位）。
      3. 将离散化后的值反归一化，恢复到原始量级，并利用一个小型 MLP 对量化结果进行残差校正，
         进一步减小重构误差。
      
    参数：
      - dim: 特征向量的维度（rank）。
      - num_levels: 量化级数（例如 16 级即 4 位表示）。
      - use_mlp: 是否使用 MLP 残差校正（默认 True）。
      - mlp_hidden_dim: MLP 隐藏层维度，默认设为 dim。
    """
    def __init__(self, *, dim, num_levels=16, use_mlp=True, mlp_hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.use_mlp = use_mlp
        
        # 学习归一化参数：每个维度的 scale 和 beta
        self.scale = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
        # 如果使用 MLP 校正，则构建一个简单的两层 MLP
        if self.use_mlp:
            mlp_hidden_dim = mlp_hidden_dim or dim
            self.correction_mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_dim, dim)
            )
    
    def forward(self, x):
        """
        前向过程：
          1. 对 x (形状 [num_points, dim]) 使用 learned scale 和 beta 进行归一化。
          2. 将归一化后的 x 限定在 [0, 1] 内，然后映射到 [0, num_levels-1] 上。
          3. 使用 STE 进行四舍五入得到整数编码（量化指数）。
          4. 将编码反映射到 [0,1]，再通过 scale 和 beta 恢复出量化后的值。
          5. 如果使用 MLP，则对量化值进行校正。
          
        返回：
          - x_out: 重构后的量化输出
          - x_code: 离散化的编码（浮点数表示，但在前向过程中使用 STE 传递梯度）
          - loss: 量化重构损失（例如 L2 均方误差）
          - total_bits: 估计的编码总位数（按每元素 log2(num_levels) 计算）
        """
        # 归一化输入（逐维处理）
        x_norm = (x - self.beta) / self.scale
        # 假定归一化后数据应位于 [0, 1] 区间
        x_norm = x_norm.clamp(0, 1)
        # 映射到 [0, num_levels - 1]
        x_scaled = x_norm * (self.num_levels - 1)
        # 使用 STE 进行离散化（四舍五入）得到编码
        x_code = ste(x_scaled)
        # 将离散化编码反映射回 [0, 1] 区间
        x_quant = x_code / (self.num_levels - 1)
        # 反归一化还原原始尺度
        x_out = x_quant * self.scale + self.beta
        
        # 使用小型 MLP 对量化结果进行残差校正，以进一步减小重构误差
        if self.use_mlp:
            correction = self.correction_mlp(x_out)
            x_out = x_out + correction
        
        # 计算重构损失（例如均方误差）
        loss = F.mse_loss(x, x_out)
        # 每个元素使用 log2(num_levels) 位存储，总位数估计：
        bits_per_element = np.log2(self.num_levels)
        total_bits = x.numel() * bits_per_element
        
        return x_out,  loss, total_bits #x_code,
    
    def compress(self, x):
        """
        压缩过程：返回离散化后的整数编码。
        """
        x_norm = (x - self.beta) / self.scale
        x_norm = x_norm.clamp(0, 1)
        x_scaled = x_norm * (self.num_levels - 1)
        x_code = x_scaled.round().long()
        return x_code
    
    def decompress(self, codes):
        """
        解压过程：将整数编码还原为量化后的值。
        """
        x_quant = codes.float() / (self.num_levels - 1)
        x_out = x_quant * self.scale + self.beta
        if self.use_mlp:
            correction = self.correction_mlp(x_out)
            x_out = x_out + correction
        return x_out

def compress_matrix_flatten_categorical(matrix, return_table=False):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    matrix = np.array(matrix) #matrix.flatten()
    # unique values, their indices, data replaced with indices of unique values, counts of them
    unique, unique_indices, unique_inverse, unique_counts = np.unique(matrix, return_index=True, return_inverse=True, return_counts=True, axis=None)
    # find range and type (bits) of unique values and compress the values
    min_value = np.min(unique)
    max_value = np.max(unique)
    unique = unique.astype(judege_type(min_value, max_value))
    # compress data replaced with indices of unique values
    message = unique_inverse.astype(np.int32)
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(probabilities,perfect=True)
    encoder = constriction.stream.stack.AnsCoder() 
    encoder.encode_reverse(message, entropy_model)
    # Final compressed output
    compressed = encoder.get_compressed()
    return compressed, unique_counts, unique

def decompress_matrix_flatten_categorical(compressed, unique_counts, quant_symbol, symbol_length, symbol_shape):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(probabilities)
    decoder = constriction.stream.stack.AnsCoder(compressed)
    decoded = decoder.decode(entropy_model, symbol_length)
    decoded = quant_symbol[decoded].reshape(symbol_shape)#.astype(np.int32)
    return decoded


def judege_type(min, max):
    if min>=0:
        if max<=256:
            return np.uint8
        elif max<=65535:
            return np.uint16
        else:
            return np.uint32
    else:
        if max<128 and min>=-128:
            return np.int8
        elif max<32768 and min>=-32768:
            return np.int16
        else:
            return np.int32
        
def get_np_size(x):
    return x.size * x.itemsize
