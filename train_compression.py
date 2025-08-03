import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import copy
from quantize import *
import matplotlib.pyplot as plt
from thop import profile

class SimpleTrainerHSI:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        ground_truth: torch.tensor,
        endmember: np.array,
        #rank: int,
        num_points: int = 2000,
        model_name: str = "GaussianImage_Cholesky_nd",
        iterations: int = 50000,
        model_path = None,
        data_name = "HSI",
        image_name = None,
    ):
        self.device = torch.device("cuda:0")
        torch.cuda.synchronize()
        self.gt_image = ground_truth.to(self.device).half()
        self.endmember = endmember
        self.num_points = num_points
        BLOCK_H, BLOCK_W = 16, 16
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) # MB
        self.H, self.W, self.rank, self.C = self.gt_image.shape[2], self.gt_image.shape[3], endmember.shape[0], self.gt_image.shape[1]
        
        self.iterations = iterations
        self.log_dir = Path(f"./checkpoints/{data_name}/{model_name}_{iterations}_{num_points}_{self.rank}/{image_name}")
        self.image_name = image_name
        
        torch.cuda.synchronize()
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) # MB
        print(f"GPU memory GT: {gpu_memory:.2f} MB")

        if model_name == "GaussianImage_Cholesky_nd":
            from gaussianimage_cholesky_unknown import GaussianImage_Cholesky_EA
            self.gaussian_model = GaussianImage_Cholesky_EA(loss_type="L2", opt_type="adan", num_points=self.num_points, GT = self.gt_image, E = self.endmember, H=self.H, W=self.W, C = self.C, rank = self.rank, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=5e-3, quantize=False).to(self.device)
            
        torch.cuda.synchronize()
        model_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) # MB
        print(f"GPU memory after model initialization: {model_gpu_memory:.2f} MB (Model size: {model_gpu_memory - gpu_memory:.2f} MB)")
        '''
        #dummy_input = torch.zeros(1, device=self.device)  # 输入形状需匹配模型的forward输入
        macs, params = profile(self.gaussian_model, inputs=())
        print(f"FLOPs: {macs / 1e9} GFLOPs, Params: {params / 1e3} K")
        '''
        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def train(self):
        psnr_list, iter_list = [], []     
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        self.gaussian_model.train()
        start_time = time.time()
        best_psnr = 0
        
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.train_iter_quantize()
            psnr_list.append(psnr)
            iter_list.append(iter)
            if best_psnr < psnr:
                best_psnr = psnr
                best_model_dict = copy.deepcopy(self.gaussian_model.state_dict())
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "Total PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10) 
        
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value, sam, bpppb = self.test()
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        self.gaussian_model.load_state_dict(best_model_dict)
        best_psnr_value, best_ms_ssim_value, best_sam, best_bpppb = self.test(True)
        torch.save(best_model_dict, self.log_dir / "gaussian_model.best.pth.tar")
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model.forward_quantize()
            test_end_time = (time.time() - test_start_time)/100
        
        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time, bpppb, best_psnr_value, best_ms_ssim_value, best_sam, best_bpppb

    def test(self, best = False):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model.forward_quantize()
            A = out["render"].float()
            E = FakeQuantizationHalf.apply(self.gaussian_model.endmember.to(torch.float32))
            I = A @ E
            I = I.view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()
            '''
            A_np = A.view(self.H, self.W, self.rank).cpu().numpy()  # [H, W, R]
            np.save('A.npy', A_np)

            I_final = np.transpose(I.cpu().numpy(), (0, 2, 3, 1))
            GT_final = np.transpose(self.gt_image.cpu().numpy(), (0, 2, 3, 1))
            I_np  = np.squeeze(I_final, axis=0)   # [H, W, C]
            GT_np = np.squeeze(GT_final, axis=0)  # [H, W, C]

            np.save('I.npy',  I_np)
            np.save('GT.npy', GT_np)
            print(f"Saved A.npy (shape={A_np.shape}), I.npy (shape={I_np.shape}), GT.npy (shape={GT_np.shape})")
            
            plt.imshow(np.mean((np.squeeze(GT_final)- np.squeeze(I_final)) ** 2, axis = -1), cmap='jet', vmin=0, vmax=0.02)
            plt.axis('off')
            plt.savefig(f'./Residual/{self.image_name}_GS.png', bbox_inches='tight', pad_inches=0)

            pseudo_rgb = create_pseudorgb(I[0].permute(1, 2, 0).cpu().numpy(), bands=[10,90,180])
            plt.imsave(f"{self.image_name}_GS.png", pseudo_rgb)
            '''
            # Compute the elementwise MSE between the predicted and target images.
            # This yields a tensor of shape [B, C, H, W].
            mse_per_channel = F.mse_loss(I, self.gt_image, reduction='none')

            # Average over the batch, height, and width dimensions.
            # The resulting tensor has shape [C], where each element is the average MSE for that channel.
            mse_per_channel_avg = mse_per_channel.mean(dim=(0, 2, 3))

            # Compute PSNR for each channel using the formula:
            # PSNR = 10 * log10(1 / MSE)
            psnr_per_channel = 10 * torch.log10(1.0 / mse_per_channel_avg)
            #np.save(f"./PSNR/GS.npy", psnr_per_channel.cpu().numpy())
            # If desired, you can further average these values to get a single scalar PSNR:
            psnr = psnr_per_channel.mean().item()
        
        ms_ssim_value = ms_ssim(I, self.gt_image.float(), data_range=1, size_average=True, win_size=7).item()#
        mean_sam = compute_sam(self.gt_image.squeeze(0).permute(1, 2, 0).cpu().numpy(), I.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        m_bit, s_bit, r_bit, c_bit = out["unit_bit"]
        e_bit = self.rank * self.C * 16
        bpppb = (m_bit + s_bit + r_bit + c_bit + e_bit) / self.H /self.W / self.C 
        
        strings = "Best Test" if best else "Test"
        self.logwriter.write("{} PSNR:{:.4f}, MS_SSIM:{:.6f}, bpppb:{:.4f}".format(strings, psnr, 
                            ms_ssim_value, bpppb))
        return psnr, ms_ssim_value, mean_sam, bpppb
    
def train_nd(gt, endmember, image_name, iterations, num_points, model_name = "Gaussian_Cholesky_nd"):
    logwriter = LogWriter(Path(f"./checkpoints/compression/{model_name}_{iterations}_{num_points}"))
    trainer = SimpleTrainerHSI(ground_truth=gt, endmember = endmember, num_points=num_points, iterations=iterations, model_name=model_name, image_name = image_name)
    psnr, ms_ssim, training_time, eval_time, eval_fps, bpppb, best_psnr_value, best_ms_ssim_value, best_sam, best_bpppb= trainer.train()
    logwriter.write("{}: {}x{}x{}, Rank: {}, bpppb: {:.4f}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Best bpppb: {:.4f}, Best PSNR:{:.4f}, Best MS-SSIM:{:.4f},Best SAM: {:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            image_name, trainer.H, trainer.W, trainer.C, trainer.rank, bpppb, psnr, ms_ssim, best_bpppb, best_psnr_value, best_ms_ssim_value, best_sam, training_time, eval_time, eval_fps))
    
if __name__ == "__main__":
    ''' 
    E = np.load('HSI/init/Urban_endmember_rank_12.npy').astype(np.float32)
    I = scipy.io.loadmat("HSI/data/Urban_R162.mat")['Y'].astype(float)
    for i in range(162): 
        I[i,:] = I[i,:]/np.max(I[i,:])
    I = I.reshape(162,307,307).transpose(2,1,0)
    
    E = np.load('HSI/init/Salinas_endmember_rank_12.npy').astype(np.float32)
    I = scipy.io.loadmat("HSI/data/Salinas_crop.mat")['I'].astype(float)
    I = np.clip(I, 0, None)
    for i in range(204): 
        I[:,:,i] = I[:,:,i]/ np.max(I[:,:,i])
    '''  
    E = np.load('HSI/init/JR_endmember_rank_10.npy').astype(np.float32)
    I = scipy.io.loadmat("HSI/data/jasperRidge2_R198.mat")['Y'].astype(float)
    for i in range(198): 
        I[i,:] = I[i,:]/np.max(I[i,:])
    I = I.reshape(198,100,100).transpose(2,1,0)
    '''
    E = np.load('HSI/init/PaviaU_endmember_rank_12.npy').astype(np.float32)
    I = scipy.io.loadmat("HSI/data/PaviaU.mat")['paviaU'].astype(float)
    for i in range(103): 
        I[:,:,i] = I[:,:,i] / np.max(I[:,:,i])
    I = I[-340:,:,:]
    '''
    GT = torch.tensor(I)
    GT = GT.view(-1, GT.size(0), GT.size(1), GT.size(2)).permute(0, 3, 1, 2).contiguous()
    GT = torch.clamp(GT,0,1)
    ''' 
    #for i in np.arange(44000,56001,4000):
    for i in [12000]:#, 14500
        train_nd(GT, endmember = E, iterations = 7000, num_points = i, 
             model_name = "GaussianImage_Cholesky_nd", image_name = "Urban")
    
    #for i in np.arange(21600, 28002, 1600):
    for i in [5000]:#, 5000
        train_nd(GT, endmember = E, iterations = 7000, num_points = i, 
             model_name = "GaussianImage_Cholesky_nd", image_name = "Salinas")
    '''
    
    #for i in np.arange(5400, 7001, 400):
    for i in [600]:#, 10002750
        train_nd(GT, endmember = E, iterations = 8000, num_points = i, 
             model_name = "GaussianImage_Cholesky_nd", image_name = "JasperRidge")
    '''
    
    #for i in np.arange(4000,64001,4000):
    for i in [10000]:#, 14500
        train_nd(GT, endmember = E, iterations = 7000, num_points = i, 
             model_name = "GaussianImage_Cholesky_nd", image_name = "PaviaU")
    '''