import numpy as np 
import argparse
import json 
import os
import torch
import matplotlib.pyplot as plt
import cv2 
from colormap import colormap


device = 'cpu'

def project_points(means3D_deform,projection):
    
    # projecting to cam frame for later use in optic flow
    means_deform_h = torch.cat([means3D_deform,torch.ones_like(means3D_deform[:,0:1])],dim=1).T 
    cam_transform = projection.to(means_deform_h.device).T

    projections = cam_transform.matmul(means_deform_h)
    projections = projections/projections[3,:]

    projections = projections[:2].T
    H, W = 800,800

    projections_cam = torch.zeros_like(projections).to(projections.device)
    projections_cam[:,0] = ((projections[:,0] + 1.0) * W - 1.0) * 0.5
    projections_cam[:,1] = ((projections[:,1] + 1.0) * H - 1.0) * 0.5
    return projections_cam




parser = argparse.ArgumentParser()
parser.add_argument('--folder',type=str,required=True)
args = parser.parse_args()

colors = colormap[np.arange(1000) % len(colormap)]
json_path = os.path.join(args.folder,'pose.json')
# load json file
with open(json_path) as json_file:
    data = json.load(json_file)
proj_mat = torch.tensor(data['transform_matrix'],device=device,dtype=torch.float32)

traj_path = os.path.join(args.folder,'traj.npz')
gt_path = os.path.join(args.folder,'gt.npz')
gt_traj = torch.tensor(np.load(gt_path)['traj'],device=device,dtype=torch.float32)

traj = torch.tensor(np.load(traj_path)['traj'],device=device,dtype=torch.float32)

img = plt.imread(os.path.join(args.folder,'img.png'))


prev_points = None
prev_mask = None

prev_gt_points = None
prev_gt_mask = None

arrow_tickness = 2
flow_skip = 10


for i in range(40):
    points = project_points(traj[i],proj_mat)
    current_mask = (points[:,0] >= 0) & (points[:,0] < 800) & (points[:,1] >= 0) & \
                (points[:,1] < 800)
    
    gt_points = project_points(gt_traj[i],proj_mat)
    current_gt_mask = (gt_points[:,0] >= 0) & (gt_points[:,0] < 800) & (gt_points[:,1] >= 0) & (gt_points[:,1] < 800)
    
    
    if prev_points is not None:
        mask_in_image = current_mask & prev_mask & current_gt_mask & prev_gt_mask
        for j in range(points.shape[0])[::flow_skip]:
            # draw arrow from prev_projections to current_projections
            color_idx = (j) % len(colors)
            if mask_in_image[j]:
                # inferred
                img = cv2.arrowedLine(img,(int(prev_points[j,0]),int(prev_points[j,1])),(int(points[j,0]),int(points[j,1])),np.array([0,0,1.0]),arrow_tickness)
                # img = cv2.circle(img,(int(points[j,0]),int(points[j,1])),3,np.array([0,0,1.0]),-1) 
                img = cv2.arrowedLine(img,(int(prev_gt_points[j,0]),int(prev_gt_points[j,1])),(int(gt_points[j,0]),int(gt_points[j,1])),np.array([1.0,0,0]),arrow_tickness)
                # gt
                
                    
    prev_points = points
    prev_mask = current_mask
    
    prev_gt_points = gt_points
    prev_gt_mask = current_gt_mask
            
points = points[mask_in_image]
plt.imshow(img)
# plt.scatter(points[:,0],points[:,1],s=1)
plt.savefig(os.path.join(args.folder,'img_proj.png'))
