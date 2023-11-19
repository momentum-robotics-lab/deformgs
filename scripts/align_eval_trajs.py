import sys
import argparse 
import numpy as np 
import matplotlib.pyplot as plt
import tqdm
import copy
import torch

def build_rotation(q):
    q = torch.tensor(q,device='cuda')
    norm = torch.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    q = q / norm
    rot = torch.zeros((3, 3), device='cuda')
    r = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    rot[0, 0] = 1 - 2 * (y * y + z * z)
    rot[0, 1] = 2 * (x * y - r * z)
    rot[0, 2] = 2 * (x * z + r * y)
    rot[1, 0] = 2 * (x * y + r * z)
    rot[1, 1] = 1 - 2 * (x * x + z * z)
    rot[1, 2] = 2 * (y * z - r * x)
    rot[2, 0] = 2 * (x * z - r * y)
    rot[2, 1] = 2 * (y * z + r * x)
    rot[2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def find_traj(gt_t0,trajs_t0):
    gt_t0 = gt_t0[None,None,:]
    dists = np.linalg.norm(gt_t0 - trajs_t0,axis=-1) 
    closest_idx = np.argmin(dists)
    
    return closest_idx

def align_traj(full_traj,gt_t0,rotations):
    translation = gt_t0 - full_traj[0,:]
    torch_translation = torch.tensor(translation,device='cuda',dtype=torch.float32).reshape((3,1))
    t0_rot = build_rotation(rotations[0,:])
    
    n_steps = full_traj.shape[0]
    new_traj = copy.deepcopy(full_traj)
    # new_traj += translation
    new_traj[0, :] += translation
    
    for i in range(1,n_steps):
        curr_rot = build_rotation(rotations[i,:])
        rel_rot = torch.matmul(curr_rot,t0_rot.T)
        current_translation = rel_rot@torch_translation
        new_traj[i,:] += current_translation.reshape((3,)).cpu().numpy()    
    
    new_traj = new_traj[:,None,:]
    return new_traj

def compute_mte(gt_traj,traj):
    mte = np.linalg.norm(gt_traj-traj.reshape((-1,3)),axis=-1)
    return np.mean(mte)

def viz_traj(ax,gt_traj,traj):
    ax.plot(gt_traj[:,0],gt_traj[:,1],gt_traj[:,2],'r')
    ax.plot(traj[:,:,0],traj[:,:,1],traj[:,:,2],'b')
    
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("--gt_file",type=str,required=True)
parser.add_argument("--traj_file",type=str,required=True)
args = parser.parse_args()

gt_data = np.load(args.gt_file)
gt_traj = gt_data['traj']
print('Gt traj: {}'.format(gt_traj.shape))


traj_data = np.load(args.traj_file)
trajs = traj_data['traj']
print("Inferred trajs shape: {}".format(trajs.shape))
rotations = traj_data['rotations']

# prep for 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

all_aligned_trajs = []
all_mtes = []
for i in tqdm.tqdm(range(gt_traj.shape[1])):
    gt_t0 = gt_traj[0,i,:]
    closest_idx = find_traj(gt_t0,trajs[0,:,:])
    traj = align_traj(trajs[:,closest_idx,:],gt_t0,rotations[:,closest_idx,:])
    mte = compute_mte(gt_traj[:,i,:],traj)
    all_mtes.append(mte)
    # viz_traj(ax,gt_traj[:,i,:],traj)
    all_aligned_trajs.append(traj)

print("mean mte: {}".format(np.mean(all_mtes)))
all_aligned_trajs = np.concatenate(all_aligned_trajs,axis=1)
np.savez(args.traj_file.replace(".npz","_aligned.npz"), traj=all_aligned_trajs, rotations=rotations)