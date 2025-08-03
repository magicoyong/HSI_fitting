from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from quantize import *
from optimizer import Adan
from torch.utils.data import DataLoader, Dataset


class GaussianImage_Cholesky_EA(nn.Module):
    def __init__(self, loss_type="L2", **kwargs): #L2 SSIM Fusion1 Fusion2
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W, self.rank, self.C = kwargs["H"], kwargs["W"], kwargs["rank"], kwargs["C"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) 
        #torch.seed(1234)
        torch.cuda.synchronize()
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) # MB
        self.device = kwargs["device"]
        self.image = kwargs["GT"]#.to(torch.float32) # (1, C, H, W)
        
        self.endmember = torch.tensor(kwargs["E"]).to(self.device)#.half() # (rank, C)
        
        torch.cuda.synchronize()
        E_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) # MB
        print(f"E0 GPU memory usage: {E_gpu_memory - gpu_memory} MB")

        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        #self._xyz = self._initialize_xyz_from_abundance(self.abundance, self.init_num_points)
        self._cholesky = nn.Parameter(torch.zeros(self.init_num_points, 3)) # 0.5 * torch.rand(self.init_num_points, 3)
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        
        self._features_dc = nn.Parameter(0.5 * torch.rand(self.init_num_points, self.rank))
        self.coef = nn.Parameter(torch.tensor(0.0))

        self.last_size = (self.H, self.W)
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(self.rank))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        
        self.endmember_quantizer = FakeQuantizationHalf.apply #UniformQuantizer(signed=False, bits=6, learned=True, num_channels=self.rank)
        self.xyz_quantizer = FakeQuantizationHalf.apply 
        self.features_dc_quantizer = VectorQuantizer(codebook_dim=self.rank, codebook_size=72,num_quantizers=2, vector_type="vector", kmeans_iters=8) 
        self.cholesky_quantizer = UniformQuantizer(signed=False, bits=8, learned=True, num_channels=3)

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.5)

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound

    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        features_split = torch.split(self.get_features, 4, dim=1)  

        out_img1 = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features_split[0], self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )
        
        out_img2 = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features_split[1], self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )
        
        out_img3 = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features_split[2], self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )
        '''
        out_img4 = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features_split[3], self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )
        
        out_img5 = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features_split[4], self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )
        
        out_img6 = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features_split[5], self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )
        
        out_img7 = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features_split[6], self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )

        out_img8 = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features_split[7], self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )
        '''
        #, out_img4, out_img5, out_img6, out_img7, out_img8
        out_img = torch.cat((out_img1, out_img2, out_img3), dim = 2)
        out_img = torch.clamp(out_img, 0, 1) #(H, W, rank)
        out_img = out_img.view(self.H * self.W, self.rank).contiguous() #(H * W, rank)
        return {"render": out_img}

    def train_iter(self):
        render_pkg = self.forward()
        A = render_pkg["render"] # (H * W, rank)
        flatimage = (A @ self.endmember) # image: (H * W, C)
        #image_cube = flatimage.view(self.H, self.W, self.C)
        image = flatimage.view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous() #[1, C, H, W]
        
        # update abundance
        loss =   loss_fn(image, self.image, self.loss_type, lambda_value=0.7) 
        loss.backward()
        '''0.15 * loss_fn(A, self.abundance) + 0.85 *0.05 * loss_fn(A, self.abundance) + 0.95 *
        # update endmember: multiplication rule
        with torch.no_grad():
            A_new = self.forward()["render"]
            image_new = (A_new @ self.endmember)
            self.endmember *=  (A_new.T @ self.image.view(-1, self.C)) / (A_new.T @ image_new)
        '''
        with torch.no_grad():
            I = (A @ self.endmember * self.coef_endmember).view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()
            mse_loss = F.mse_loss(I, self.image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
            
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
        return loss, psnr
    
    def forward_quantize(self):
        # quantize plan1: "GaussianImage"
        l_vqm, m_bit = 0, 16*self.init_num_points*2
        means = torch.tanh(FakeQuantizationHalf.apply(self._xyz))#FakeQuantizationHalf.apply(self._xyz)
        
        cholesky_elements, l_vqs, s_bit = self.cholesky_quantizer(self._cholesky)#self._cholesky, 0, 32*self.init_num_points*3 32*self.init_num_points*3#self.cholesky_quantizer(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        l_vqr, r_bit = 0, 0
        
        features, l_vqc, c_bit = self.get_features, 0, 32*self.init_num_points*self.rank#self.features_dc_quantizer(self.get_features)#self.get_features, 0, 32*self.init_num_points*self.rank
        
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, 
                cholesky_elements, self.H, self.W, self.tile_bounds)
        
        features_split = torch.split(features, (4,3,3), dim=1)  
        
        out_img1 = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features_split[0], self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )#
        
        out_img2 = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features_split[1], self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )
        
        out_img3 = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features_split[2], self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )
        '''
        out_img4 = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features_split[3], self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )
        
        '''
        #, out_img4
        out_img = torch.cat((out_img1, out_img2, out_img3), dim = 2)
        out_img = torch.clamp(out_img, 0, 1) #(H, W, rank)
        out_img = out_img.view(self.H * self.W, self.rank).contiguous() * torch.exp(self.coef)#(H * W, rank)
        
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
        return {"render": out_img, "vq_loss": vq_loss, "unit_bit":[m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self):
        render_pkg = self.forward_quantize()
        A = render_pkg["render"] # (H * W, rank)
        E = FakeQuantizationHalf.apply(self.endmember)
        flatimage = A @ E # image: (H * W, C)
        image = flatimage.view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous() #[1, C, H, W]
        
        # update abundance
        loss =  loss_fn(image, self.image, self.loss_type, lambda_value=0.7) + 0.05 * render_pkg['vq_loss']
        loss.backward()
        #0.05 * loss_fn(A, self.abundance) + 0.95 * 
        with torch.no_grad():
            I = (A @ self.endmember).view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()
            # Compute the elementwise MSE between the predicted and target images.
            # This yields a tensor of shape [B, C, H, W].
            mse_per_channel = F.mse_loss(I, self.image, reduction='none')
            # Average over the batch, height, and width dimensions.
            # The resulting tensor has shape [C], where each element is the average MSE for that channel.
            mse_per_channel_avg = mse_per_channel.mean(dim=(0, 2, 3))
            # Compute PSNR for each channel using the formula:
            # PSNR = 10 * log10(1 / MSE)
            psnr_per_channel = 10 * torch.log10(1.0 / mse_per_channel_avg)
            # If desired, you can further average these values to get a single scalar PSNR:
            psnr = psnr_per_channel.mean().item()
        
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr
    