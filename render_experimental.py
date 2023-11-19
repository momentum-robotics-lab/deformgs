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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import glob 
import matplotlib.pyplot as plt
from colormap import colormap


tonumpy = lambda x : x.cpu().numpy()
to8 = lambda x : np.uint8(np.clip(x,0,1)*255)

def merge_deform_logs(folder):
    npz_files = glob.glob(os.path.join(folder,'log_deform_*.npz'),recursive=True)
    # sort based on the float number in the file name
    npz_files.sort(key=lambda f: float(f.split('/')[-1].replace('log_deform_','').replace('.npz','')))
    times = [float(''.join(filter(str.isdigit, os.path.basename(f)) )) for f in npz_files]
    trajs = []
    rotations = []
    for npz_file in npz_files:
        deforms_data = np.load(npz_file)
        xyzs_deformed = deforms_data['means3D_deform']
        rotations.append(deforms_data['rotations'])
        trajs.append(xyzs_deformed)


    trajs = np.stack(trajs)
    rotations = np.stack(rotations)
    
    np.savez(os.path.join(folder,'all_trajs.npz'),traj=trajs,rotations=rotations)
    print("saved all trajs to {}".format(os.path.join(folder,'all_trajs.npz')))
    print("shape of all trajs: {}".format(trajs.shape))
    


def render_set(model_path, name, iteration, views, gaussians, pipeline, background,log_deform=False,args=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    video_imgs = []
    save_imgs = []
    gt_list = []
    render_list = []
    
    all_times = [view.time for view in views]
    n_gaussians = len(all_times)
    todo_times = np.unique(all_times)
    n_times = len(todo_times)
    colors = colormap[np.arange(n_gaussians) % len(colormap)]

    prev_projections = None 
    current_projections = None 
    prev_visible = None

    view_id = views[0].view_id

    arrow_color = (0,255,0)
    arrow_tickness = 2
    raddii_threshold = 0

    desired_idx = 824
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        log_deform_path = None

        view_time = view.time
        view_id = view.view_id
        time_id = view.time_id
        

        if idx == desired_idx:
            print("found desired view and time")
            print(view.view_id,view.time_id)
            
            print(view.full_proj_transform)
            exit()
            break
        

        if prev_projections is None:
            traj_img = np.zeros((view.image_height,view.image_width,3))

        if log_deform and view_time in todo_times:
            log_deform_path = os.path.join(model_path, name, "ours_{}".format(iteration), "log_deform_{}".format(view.time))

            # remove time from todo_times
            todo_times = todo_times[todo_times != view_time]
        
        render_pkg = render(view, gaussians, pipeline, background,log_deform_path=log_deform_path,no_shadow=args.no_shadow)
        rendering = tonumpy(render_pkg["render"]).transpose(1,2,0)

        depth = render_pkg["depth"].to("cpu").numpy()

        if args.show_flow:
            current_projections = render_pkg["projections"].to("cpu").numpy()
            
            current_mask_in_image = (current_projections[:,0] >= 0) & (current_projections[:,0] < view.image_height) & (current_projections[:,1] >= 0) & \
            (current_projections[:,1] < view.image_width)
            
            # current_depth_projections = depth[0,current_projections[:,1].astype(np.int),current_projections[:,0].astype(np.int)]
            # current_gaussian_positions = render_pkg["means3D_deform"].cpu().numpy()
            # cam_center = view.camera_center.cpu().numpy()
            # gaussian_dists = np.linalg.norm(current_gaussian_positions - cam_center,axis=-1)

            # depth_mask = (gaussian_dists - 0.05) <= current_depth_projections


            opacity = render_pkg["opacities"].to("cpu").numpy().flatten()
            
            radii = render_pkg["radii"].to("cpu").numpy()
            current_visible = radii > raddii_threshold
            # fraction of visible gaussians
            current_mask = current_visible & current_mask_in_image        
            for i in range(n_gaussians)[::args.flow_skip]:
                if current_mask[i]:
                    color_idx = (i//args.flow_skip) % len(colors)
                
                    rendering[int(current_projections[i,1]),int(current_projections[i,0]),:] = colors[color_idx]

            if view_id != view.view_id:
                prev_projections = None
            else:
                if prev_projections is not None:
                    # draw flow at previous frame
                    prev_mask_in_image = (prev_projections[:,0] >= 0) & (prev_projections[:,0] < view.image_height) & (prev_projections[:,1] >= 0) & \
                    (prev_projections[:,1] < view.image_width)
                    prev_mask = prev_visible & prev_mask_in_image
                    
                    traj_img = np.ascontiguousarray(traj_img)
                    for i in range(current_projections.shape[0])[::args.flow_skip]:
                        # draw arrow from prev_projections to current_projections
                        color_idx = (i//args.flow_skip) % len(colors)
                        if prev_mask[i] and current_mask[i]:
                            traj_img = cv2.arrowedLine(traj_img,(int(prev_projections[i,0]),int(prev_projections[i,1])),(int(current_projections[i,0]),int(current_projections[i,1])),colors[color_idx],arrow_tickness)

                rendering[traj_img > 0] = traj_img[traj_img > 0]
                prev_projections = current_projections
                prev_visible = current_visible
            
            view_id = view.view_id
            
        
        render_list.append(rendering)
            

        if name in ["train", "test"]:
            gt = view.original_image[0:3, :, :]
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            gt_list.append(gt)

    video_imgs = [to8(img) for img in render_list]
    save_imgs = [torch.tensor((img.transpose(2,0,1)),device="cpu") for img in render_list ]


    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    count = 0
    print("writing training images.")
    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count+=1
    count = 0
    print("writing rendering images.")
    if len(save_imgs) != 0:
        for image in tqdm(save_imgs):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), video_imgs, fps=30, quality=8)
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool,log_deform=False,user_args=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,user_args=user_args)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,log_deform=log_deform,args=user_args)
        if not skip_test:
            log_folder = os.path.join(args.model_path, "test", "ours_{}".format(scene.loaded_iter))
            delete_previous_deform_logs(log_folder)
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,log_deform=log_deform,args=user_args) 
            if user_args.log_deform:
                merge_deform_logs(log_folder)           
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,log_deform=log_deform,args=user_args)
 
def delete_previous_deform_logs(folder):
    npz_files = glob.glob(os.path.join(folder,'log_deform_*.npz'),recursive=True)
    for npz_file in npz_files:
        os.remove(npz_file)
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--time_skip",type=int,default=None)
    parser.add_argument("--view_skip",default=None,type=int)
    parser.add_argument("--log_deform", action="store_true")
    parser.add_argument("--three_steps_batch",type=bool,default=False)
    parser.add_argument("--show_flow",action="store_true")
    parser.add_argument("--flow_skip",type=int,default=1)
    parser.add_argument("--no_shadow",action="store_true")

    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video,log_deform=args.log_deform,user_args=args)
    