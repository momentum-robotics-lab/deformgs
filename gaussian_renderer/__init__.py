#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import matplotlib.pyplot as plt

def filter_gaussians(gaussians,bounding_box):
    # gaussians : N x 3 torch tensor
    # bounding_box : 6 x 1 torch tensor
    # return : N x 3 torch tensor
    mask = (gaussians[:,0] > bounding_box[0]) & (gaussians[:,0] < bounding_box[1]) & \
            (gaussians[:,1] > bounding_box[2]) & (gaussians[:,1] < bounding_box[3]) & \
            (gaussians[:,2] > bounding_box[4]) & (gaussians[:,2] < bounding_box[5])
    return mask

def get_pos_t0(pc:GaussianModel):
    means3D = pc.get_xyz
    scales = pc._scaling
    rotations = pc._rotation
    opacity = pc._opacity
    time = torch.tensor(0.0).to(means3D.device).repeat(means3D.shape[0],1)
    deformation_point = pc._deformation_table
    t_0_points, _, _, _, _ =  pc._deformation(means3D[deformation_point], scales[deformation_point], 
                                                                         rotations[deformation_point], opacity[deformation_point],
                                                                         time[deformation_point])  
    means3D_final = torch.zeros_like(means3D)
    means3D_final[deformation_point] =  t_0_points
    means3D_final[~deformation_point] = means3D[~deformation_point]
    
    return means3D_final


def get_all_pos(pc:GaussianModel):
    """
    Returns a trajectory for each Gaussian in the scene
    Output: (N_gaussians, N_frames, 3)
    """
    means3D = pc.get_xyz
    scales = pc._scaling
    rotations = pc._rotation
    opacity = pc._opacity
    deformation_point = pc._deformation_table

    n_gaussians = means3D.shape[0]
    all_times = pc.all_times
    n_times = len(all_times)

    # generate the time tensor, (N_frames x N_gaussians, 1) and cast to same dtype as means3D
    time = torch.tensor(all_times).to(means3D.device).reshape(-1,1)
    time = time.repeat(n_gaussians,1)
    # cast to dtype of means3D
    time = time.to(means3D.dtype)
    # sort time
    time, _ = time.sort(dim=0)
    
    
    #also repeat the other tensors
    means3D = means3D.repeat(n_times,1)
    scales = scales.repeat(n_times,1)
    rotations = rotations.repeat(n_times,1)
    opacity = opacity.repeat(n_times,1)
    deformation_point = deformation_point.repeat(n_times)

    if deformation_point.sum() > 0: 
        means3D_deformed, _, _, _, _ =  pc._deformation(means3D[deformation_point], scales[deformation_point],
                                                                                rotations[deformation_point], opacity[deformation_point],
                                                                                time[deformation_point])
    else:
        means3D_deformed = means3D[deformation_point]

    means3D_final = torch.zeros_like(means3D)
    means3D_final[deformation_point] =  means3D_deformed
    means3D_final[~deformation_point] = means3D[~deformation_point]
    means3D_final = means3D_final.reshape(n_times,n_gaussians,3)
    # cast to (N_gaussians x N_frames x 3)
    means3D_final = means3D_final.permute(1,0,2)


    return means3D_final



def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine",log_deform_path=None,no_shadow=False,split=None,bounding_box=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    if viewpoint_camera.f_x is None:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        tanfovx = viewpoint_camera.image_width/(2*viewpoint_camera.f_x)
        tanfovy = viewpoint_camera.image_height/(2*viewpoint_camera.f_y)
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=False
    )

    # filename = viewpoint_camera.image_path.split('/')[-1].split('.')[0]
    # #write all inputs to GaussianRasterizationSettings to a txt file
    # with open('4dgs_inputs/{}.txt'.format(filename), 'w') as f:
    #     print(raster_settings, file=f)


   

   

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation
    means3D = pc.get_xyz
        
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    means2D = screenspace_points
    opacity = pc._opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    shadow_scalars = None
    shadow_scalars_static = None
    if stage == "coarse" :
    # if tur
        means3D_deform, scales_deform, rotations_deform, opacity_deform = means3D, scales, rotations, opacity
        shadow_scalars = pc._deformation(means3D[deformation_point], scales[deformation_point], 
                                                                         rotations[deformation_point], opacity[deformation_point],
                                                                         time[deformation_point],shadow_only=True)    

    else:
        if deformation_point.sum() > 0:
            means3D_deform, _, rotations_deform, _ , shadow_scalars = pc._deformation(means3D[deformation_point], scales[deformation_point], 
                                                                         rotations[deformation_point], opacity[deformation_point],
                                                                         time[deformation_point])        

            scales_deform, opacity_deform = scales[deformation_point], opacity[deformation_point] 
        else:
            means3D_deform, scales_deform, rotations_deform, opacity_deform = means3D[deformation_point], scales[deformation_point], rotations[deformation_point], opacity[deformation_point]
            shadow_scalars = None

        # scales_deform = scales
        if (~deformation_point).sum() > 0:
            shadow_scalars_static = pc._deformation(means3D[~deformation_point], scales[~deformation_point],
                                                                         rotations[~deformation_point], opacity[~deformation_point],
                                                                         time[~deformation_point],shadow_only=True)
        


    # print(time.max())
    #with torch.no_grad():
        #pc._deformation_accum[deformation_point] += torch.abs(means3D_deform-means3D[deformation_point])

    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    shadow_scalars_final = torch.ones_like(opacity_final)

    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]

    if shadow_scalars is not None:
        shadow_scalars_final[deformation_point] = shadow_scalars
    if shadow_scalars_static is not None:
        shadow_scalars_final[~deformation_point] = shadow_scalars_static

    
    if log_deform_path is not None:
            np.savez(log_deform_path,means3D=means3D.cpu().numpy(),means3D_deform=means3D_final.cpu().numpy(),
                     rotations=rotations_final.cpu().numpy())
            
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if no_shadow or override_color is not None:
        shadow_scalars = None
    if override_color is None:
        if shadow_scalars_final is not None: # we compute colors in python to multiply with our shadow scalars
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if shadow_scalars is not None:
        # shadow_scalars = [N,1]
        # colors_precomp = [N,3]
        # element-wise multiplication
        colors_precomp = colors_precomp * shadow_scalars_final.repeat(1,3)
        #colors_precomp = colors_precomp * shadow_scalars_final

    mask = torch.gt(torch.ones_like(opacity_final), 0.0)
    if split is not None:
        if split == "static":
            mask[deformation_point] = False #only render the static part
        elif split == "dynamic":
            mask[~deformation_point] = False #only render the dynamic part
    # flatten to [N]
    mask = mask.flatten()    

    if bounding_box is not None:
        bounding_mask = filter_gaussians(means3D_final,bounding_box)
        mask = mask & bounding_mask

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final[mask],
        means2D = means2D[mask],
        shs = shs,
        colors_precomp = colors_precomp[mask],
        opacities = opacity[mask],
        scales = scales_final[mask],
        rotations = rotations_final[mask],
        cov3D_precomp = cov3D_precomp)
    

    mask_color = pc.mask_activation(pc._mask.repeat(1,3))
    rendered_mask, _, _ = rasterizer(
        means3D = means3D_final[mask],
        means2D = means2D[mask],
        shs = shs,
        colors_precomp = mask_color[mask],
        opacities = opacity[mask],
        scales = scales_final[mask],
        rotations = rotations_final[mask],
        cov3D_precomp = cov3D_precomp)
    
    # projecting to cam frame for later use in optic flow
    means_deform_h = torch.cat([means3D_final,torch.ones_like(means3D_final[:,0:1])],dim=1).T 
    cam_transform = viewpoint_camera.full_proj_transform.to(means_deform_h.device).T

    projections = cam_transform.matmul(means_deform_h)
    projections = projections/projections[3,:]

    projections = projections[:2].T
    H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)

    projections_cam = torch.zeros_like(projections).to(projections.device)
    projections_cam[:,0] = ((projections[:,0] + 1.0) * W - 1.0) * 0.5
    projections_cam[:,1] = ((projections[:,1] + 1.0) * H - 1.0) * 0.5

    shadows_mean = None
    shadows_std = None

    if shadow_scalars_final is not None:
        shadows_mean = torch.mean(shadow_scalars_final)
        shadows_std = torch.std(shadow_scalars_final)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "means3D_deform":means3D_final,
            "shadows_mean":shadows_mean,    
            "shadows_std":shadows_std,
            "projections":projections_cam,
            "rotations": rotations_final,
            "opacities": opacity_final,
            "shadows":shadow_scalars_final,
            }

