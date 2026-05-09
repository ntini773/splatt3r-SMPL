

import torch,os
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from submodules.lpipsPyTorch import LPIPS
import pytorch3d
import lightning as L
from models.modules.flame.FLAME import FLAME 
# try:
#     from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
# except:
#     pass
from fused_ssim import fused_ssim
from torchvision.transforms.functional import to_pil_image
C1 = 0.01 ** 2
C2 = 0.03 ** 2

# class FusedSSIMMap(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, C1, C2, img1, img2):
#         ssim_map = fusedssim(C1, C2, img1, img2)
#         ctx.save_for_backward(img1.detach(), img2)
#         ctx.C1 = C1
#         ctx.C2 = C2
#         return ssim_map

#     @staticmethod
#     def backward(ctx, opt_grad):
#         img1, img2 = ctx.saved_tensors
#         C1, C2 = ctx.C1, ctx.C2
#         grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
#         return None, None, grad, None

def cal_l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def cal_l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def cal_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# def fast_ssim(img1, img2):
#     ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
#     return ssim_map.mean()
def fast_ssim(img1, img2):
    ssim_value  = fused_ssim(img1, img2)
    return ssim_value

def cal_mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def cal_psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def cal_point_nn_weight(xyz,vertex,K=1):
    
    vertex=vertex.detach()
    nn_dist, nn_idxs, nn_points = pytorch3d.ops.knn_points(xyz, vertex, None, None, K=K,return_nn=True)  # N, K
    nn_dist, nn_idxs = nn_dist[:,:,:], nn_idxs[:,:,:]  # B,N, K

    nn_weight = 1/(K)  # N, K/ (2 * nn_radius ** 2)

    return nn_idxs,nn_weight, nn_dist,nn_points

class Optimization_Loss(L.LightningModule):
    def __init__(self,cfg,laplacian_matrix,v_template,smplx2flame_ind):
        super().__init__()
        self.cfg=cfg.OPTIMIZE
        self.bg_color=cfg.MODEL.bg_color
        self.up_size=cfg.MODEL.unprojection_size
        self.with_uv_gaussian=cfg.MODEL.with_uv_gaussian
        
        self.perpetual_loss_f=LPIPS('alex', '0.1')##None
        self.perpetual_loss_f.eval()
        
        self.l1_loss_f=F.l1_loss
        self.ssim_loss_f=cal_ssim#fast_ssim
        # self.k_nearest=1
        # self.with_two_plane=cfg.MODEL.with_two_plane
        #self.dummy_param=torch.nn.Parameter(torch.zeros(1),requires_grad=True)
        self.laplacian_matrix=torch.nn.Parameter(laplacian_matrix.clone(),requires_grad=False)
        self.v_template=torch.nn.Parameter(v_template.clone(),requires_grad=False)
        self.flame=FLAME(cfg.MODEL.flame_assets_dir)
        self.smplx2flame_ind =smplx2flame_ind
        #  np.load(os.path.join(cfg.MODEL.smplx_assets_dir, 'SMPL-X__FLAME_vertex_ids.npy'))
        self.reg_offset_rigid_for: tuple[str, ...] = ("left_ear", "right_ear", "neck", "left_eye", "right_eye", "lips_tight")

        self.reg_offset_relax_for: tuple[str, ...] = ("hair", "ears")
        self.reg_offset_lap_relax_coef: float = 0.1
        self.reg_offset_relax_coef: float = 1.0
        
    def init_perpetual_loss(self,perpetual_loss_model):
        self.perpetual_loss_f=perpetual_loss_model
        
    def forward(self,render_results,batch,extra_results,iter_idx):
        
        # batch_size=batch['image'].shape[0]
        render_images=render_results['renders']
        render_masks =  render_results['render_masks'] 
        gt_images=batch['target_image']
        gt_masks=batch['target_mask']
        # extra_results['ver_lbs_weights']=extra_results['ver_lbs_weights'][None].expand(batch_size,-1,-1)

        gt_images = gt_images * (gt_masks) + (1-gt_masks) * self.bg_color # target image 一开始没做mask， 这里才做mask，在这里扣出人体 反向传播
        loss_dict={}




        if iter_idx < 1000:
            render_images = render_images*(gt_masks)+(1-gt_masks) * self.bg_color 
        loss_dict['image_loss'] = self.l1_loss_f(render_images, gt_images) * self.cfg.lambda_l1
        
        mask =   gt_masks > 0.3 # (gt_masks[:, 0:1] > 0.6) & (render_masks > 0.6)
        loss_dict['mask_loss'] = self.l1_loss_f(gt_masks[mask], render_masks[mask]) * 0.


        lambda_perpetual=self.cfg.lambda_perpetual
        if iter_idx > self.cfg.perpetual_increase_iter:
            lambda_perpetual = self.cfg.lambda_perpetual_high

        loss_dict['perpetual_loss'] = self.perpetual_loss_f(render_images,gt_images) * lambda_perpetual


        
        loss_opacity = (torch.relu(0.9-extra_results['vertex_opacity'])**2).mean()*self.cfg.lambda_opacity_map
        loss_opacity += (torch.relu(0.9-extra_results['uv_point_opacity'])**2).mean()*self.cfg.lambda_opacity_map
        loss_local_xyz = F.relu((extra_results['uv_point_xyz']).norm(dim=-1) - self.cfg.threshold_local_xyz).mean() * self.cfg.lambda_local_xyz
        loss_local_scale = F.relu(extra_results['uv_point_scale'] - self.cfg.threshold_scale).norm(dim=-1).mean() * self.cfg.lambda_local_scale
        if not self.with_uv_gaussian:
            loss_dict['scaling_loss']=F.relu(extra_results['vertex_scale'] - 0.003).norm(dim=-1).mean() * 1.0
        
        loss_dict['opacity_loss']=loss_opacity
        loss_dict['local_xyz_loss']=loss_local_xyz
        loss_dict['local_scale_loss']=loss_local_scale

        show_loss={}
        for key in loss_dict.keys():
            # if torch.isnan(loss_dict[key]).any():
            show_loss[key]=loss_dict[key].item()
        return loss_dict, show_loss
    
    def cal_box_loss(self,render_images,gt_images,box,loss_funs,loss_lambdas):
        #box:left,right,top,bottom
        batch_size = render_images.size(0)
        gt_crops,render_crops=[],[]
        loss=0.0
        for i in range(batch_size):
            gt_crop=gt_images[i, :, box[i, 2]:box[i, 3], box[i, 0]:box[i, 1]]
            render_crop=render_images[i, :, box[i, 2]:box[i, 3], box[i, 0]:box[i, 1]]
            if gt_crop.shape[1]<1 or gt_crop.shape[2]<1:
                continue
            gt_crop=F.interpolate(gt_crop[None],(256,256),mode='bilinear')
            render_crop=F.interpolate(render_crop[None],(256,256),mode='bilinear')
            gt_crops.append(gt_crop)
            render_crops.append(render_crop)
        render_crops=torch.cat(render_crops,dim=0)
        gt_crops=torch.cat(gt_crops,dim=0)
        for ii in range(len(loss_funs)):
            loss=loss+loss_funs[ii](render_crops,gt_crops)*loss_lambdas[ii]
        
        return loss
    
    def compute_laplacian_smoothing_loss(self, verts, offset_verts):
        
        batch_size = offset_verts.shape[0]
        L = self.laplacian_matrix[None, ...].detach()  # (1, V, V)
        basis_lap = L.bmm(verts[None]).detach()  #.norm(dim=-1) * weights
        offset_lap = L.expand(batch_size,-1,-1).bmm(offset_verts)  #.norm(dim=-1) # * weights
        diff = (offset_lap - basis_lap) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff
    
    def scale_vertex_weights_by_region(self, weights, scale_factor, region):
        indices = self.flame.mask.get_vid_by_region(region)
        weights[:,self.smplx2flame_ind][:, indices] *= scale_factor
        return weights
    
def save_mesh_image(extra_results,save_path='z_temp/pos_mesh_image.png'):
    import torchvision.utils as vutils
    mesh_image0=extra_results['mesh_image'][0,:,:,:3]
    # alpha_images=extra_results['mesh_image'][0,:,:,3:]
    alpha_images=(extra_results['mesh_image'][0,...,3:]>=0.5).float()#(extra_results['mesh_image'][0,...,3:])*
    xyz_range=mesh_image0.reshape(-1,3)
    xyz_min=xyz_range.min(dim=0)[0][None,None]
    xyz_max=xyz_range.max(dim=0)[0][None,None]
    mesh_image0=(mesh_image0-xyz_min)/(xyz_max-xyz_min)
    mesh_image0=mesh_image0*(alpha_images)
    # mesh_image0[alpha_images.expand(-1, -1, 3)<0.5] = 0.0
    vutils.save_image(mesh_image0.permute(2,0,1), save_path)
def save_crop_image(gt_crops,save_path='z_temp/gt_crop_0.png'):
    import torchvision.utils as vutils
    vutils.save_image(gt_crops[0], save_path)




class Ehm_Optimization_Loss(L.LightningModule):
    def __init__(self,cfg,laplacian_matrix,v_template,smplx2flame_ind):
        super().__init__()
        self.cfg=cfg.OPTIMIZE
        self.bg_color=cfg.MODEL.bg_color
        self.up_size=cfg.MODEL.unprojection_size
        self.with_uv_gaussian=cfg.MODEL.with_uv_gaussian
        
        self.perpetual_loss_f=LPIPS('alex', '0.1')##None
        self.perpetual_loss_f.eval()
        
        self.l1_loss_f =F.l1_loss # torch.nn.L1Loss(reduction = 'mean')  # F.l1_loss
        self.ssim_loss_f = cal_ssim#fast_ssim
        # self.k_nearest=1
        # self.with_two_plane=cfg.MODEL.with_two_plane
        #self.dummy_param=torch.nn.Parameter(torch.zeros(1),requires_grad=True)
        self.laplacian_matrix=torch.nn.Parameter(laplacian_matrix.clone(),requires_grad=False)
        self.v_template=torch.nn.Parameter(v_template.clone(),requires_grad=False)
        self.flame=FLAME(cfg.MODEL.flame_assets_dir)
        self.smplx2flame_ind =smplx2flame_ind
        #  np.load(os.path.join(cfg.MODEL.smplx_assets_dir, 'SMPL-X__FLAME_vertex_ids.npy'))
        self.reg_offset_rigid_for: tuple[str, ...] = ("left_ear", "right_ear", "neck", "left_eye", "right_eye", "lips_tight")

        self.reg_offset_relax_for: tuple[str, ...] = ("hair", "ears")
        self.reg_offset_lap_relax_coef: float = 0.1
        self.reg_offset_relax_coef: float = 1.0
        
    def  init_perpetual_loss(self,perpetual_loss_model):
        self.perpetual_loss_f=perpetual_loss_model
        
    def forward(self,render_results,batch,iter_idx):
        
        # batch_size=batch['image'].shape[0]
        render_images = render_results['renders']
        render_masks =  render_results['render_masks'] 
        gt_images=batch['target_image']  # 512 resolution
        gt_masks=batch['target_mask']

        gt_images = gt_images * (gt_masks) + (1-gt_masks) * self.bg_color # 这个 mask 是大于等于人体的
        loss_dict = {}

        if iter_idx < 1000: 
            render_images = render_images * (gt_masks) + (1-gt_masks) * self.bg_color   
        render_valid = batch['render_valid'].reshape(-1,1,1,1)
        loss_dict['image_loss'] = self.l1_loss_f( render_valid * render_images, render_valid *  gt_images) * 0.1 #  a little smaller than smplx loss #  self.cfg.lambda_l1  

        # type 1: mask loss   # 不完全舍弃非重合区域，而是让它们贡献较小的损失，更平滑。
        # mask_weight = (gt_masks[:, 0:1] * render_masks).detach()  # 可选 detach 避免影响梯度传播
        # diff = (gt_masks[:, 0:1] - render_masks).abs()          
        # loss_dict['mask_loss'] = 0.1 * (diff * mask_weight).mean()     
        # type 2:     
        # mask =   gt_masks > 0.3 # (gt_masks[:, 0:1] > 0.6) & (render_masks > 0.6)
        # loss_dict['mask_loss'] = self.l1_loss_f( gt_masks[mask],  render_masks[mask] ) * 0 # 0.01


        show_loss={}
        for key in loss_dict.keys():
            # if torch.isnan(loss_dict[key]).any():
            show_loss[key]=loss_dict[key].item()
        return loss_dict, show_loss
    
    def cal_box_loss(self,render_images,gt_images,box,loss_funs,loss_lambdas):
        #box:left,right,top,bottom
        batch_size = render_images.size(0)
        gt_crops,render_crops=[],[]
        loss=0.0
        for i in range(batch_size):
            gt_crop=gt_images[i, :, box[i, 2]:box[i, 3], box[i, 0]:box[i, 1]]
            render_crop=render_images[i, :, box[i, 2]:box[i, 3], box[i, 0]:box[i, 1]]
            if gt_crop.shape[1]<1 or gt_crop.shape[2]<1:
                continue
            gt_crop=F.interpolate(gt_crop[None],(256,256),mode='bilinear')
            render_crop=F.interpolate(render_crop[None],(256,256),mode='bilinear')
            gt_crops.append(gt_crop)
            render_crops.append(render_crop)
        render_crops=torch.cat(render_crops,dim=0)
        gt_crops=torch.cat(gt_crops,dim=0)
        for ii in range(len(loss_funs)):
            loss=loss+loss_funs[ii](render_crops,gt_crops)*loss_lambdas[ii]
        
        return loss
    
    def compute_laplacian_smoothing_loss(self, verts, offset_verts):
        
        batch_size = offset_verts.shape[0]
        L = self.laplacian_matrix[None, ...].detach()  # (1, V, V)
        basis_lap = L.bmm(verts[None]).detach()  #.norm(dim=-1) * weights
        offset_lap = L.expand(batch_size,-1,-1).bmm(offset_verts)  #.norm(dim=-1) # * weights
        diff = (offset_lap - basis_lap) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff
    
    def scale_vertex_weights_by_region(self, weights, scale_factor, region):
        indices = self.flame.mask.get_vid_by_region(region)
        weights[:,self.smplx2flame_ind][:, indices] *= scale_factor
        return weights
    
def save_mesh_image(extra_results,save_path='z_temp/pos_mesh_image.png'):
    import torchvision.utils as vutils
    mesh_image0=extra_results['mesh_image'][0,:,:,:3]
    # alpha_images=extra_results['mesh_image'][0,:,:,3:]
    alpha_images=(extra_results['mesh_image'][0,...,3:]>=0.5).float()#(extra_results['mesh_image'][0,...,3:])*
    xyz_range=mesh_image0.reshape(-1,3)
    xyz_min=xyz_range.min(dim=0)[0][None,None]
    xyz_max=xyz_range.max(dim=0)[0][None,None]
    mesh_image0=(mesh_image0-xyz_min)/(xyz_max-xyz_min)
    mesh_image0=mesh_image0*(alpha_images)
    # mesh_image0[alpha_images.expand(-1, -1, 3)<0.5] = 0.0
    vutils.save_image(mesh_image0.permute(2,0,1), save_path)
def save_crop_image(gt_crops,save_path='z_temp/gt_crop_0.png'):
    import torchvision.utils as vutils
    vutils.save_image(gt_crops[0], save_path)