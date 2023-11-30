import argparse
import json
import os 
import numpy as np 
import matplotlib.pyplot as plt 
import copy 

parser = argparse.ArgumentParser()
parser.add_argument('--folder',type=str,required=True)
args = parser.parse_args()

json_file = os.path.join(args.folder,'transforms_test.json')

with open(json_file) as json_file:
    data = json.load(json_file)

cam_positions = [np.array(frame['transform_matrix'])[:3,3].reshape((1,3)) for frame in data['frames']]
cam_positions = np.concatenate(cam_positions,axis=0)
_, idx = np.unique(cam_positions,axis=0,return_index=True)
unique_cam_positions = cam_positions[np.sort(idx)]
n_poses = unique_cam_positions.shape[0]
dt = 1.0/(n_poses//2-1)

times = np.arange(0,1.0+dt,dt)
times = np.concatenate([times,times[::-1]])
n_times = times.shape[0]
assert n_times == n_poses

frames = []
for id, idx in enumerate(np.sort(idx)):
    frame = data['frames'][idx]
    frame['time'] = times[id]
    frame['transform_matrix'] = data['frames'][idx]['transform_matrix']
    frames.append(frame)

traj_json = copy.deepcopy(data)
data['frames'] = frames
json_path = os.path.join(args.folder,'video.json')
# save json
with open(json_path, 'w') as outfile:
    json.dump(data, outfile, indent=4)


mean_cam_dist = np.linalg.norm(cam_positions,axis=1).mean()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(unique_cam_positions[:,0],unique_cam_positions[:,1],unique_cam_positions[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_aspect('equal', adjustable='box')
ax.set_title('Camera Positions')
plt.show()