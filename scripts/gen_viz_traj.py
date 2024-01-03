import argparse
import json
import os 
import numpy as np 
import matplotlib.pyplot as plt 
import copy 

parser = argparse.ArgumentParser()
parser.add_argument('--folder',type=str,required=True)
parser.add_argument('--densify',action='store_true')
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
final_pos = []
for id, idx in enumerate(np.sort(idx)):
    frame = data['frames'][idx]
    frame['time'] = times[id]
    frame['transform_matrix'] = data['frames'][idx]['transform_matrix']
    frames.append(frame)
    final_pos.append(np.array(frame['transform_matrix'])[:3,3].reshape((1,3)))

if args.densify:
    # insert frame between every two frames that is the average time and transform matrix
    original_frames = copy.deepcopy(frames)
    frames = []
    final_pos = []
    for i in range(len(original_frames)-1):
        frames.append(copy.deepcopy(original_frames[i]))
        final_pos.append(np.array(original_frames[i]['transform_matrix'])[:3,3].reshape((1,3)))

        new_frame = copy.deepcopy(original_frames[i])
        new_frame['time'] = (original_frames[i]['time'] + original_frames[i+1]['time'])/2.0
        new_frame['transform_matrix'] = ((np.array(original_frames[i]['transform_matrix']) + np.array(original_frames[i+1]['transform_matrix']))/2.0).tolist()
        frames.append(new_frame)
        final_pos.append(np.array(new_frame['transform_matrix'])[:3,3].reshape((1,3)))
    
    frames.append(original_frames[-1])
    final_pos.append(np.array(original_frames[-1]['transform_matrix'])[:3,3].reshape((1,3)))
    
    intermediate_frame = copy.deepcopy(original_frames[-1])
    intermediate_frame['time'] =  (original_frames[-1]['time'] + original_frames[0]['time'])/2.0
    intermediate_frame['transform_matrix'] = ((np.array(original_frames[-1]['transform_matrix']) + np.array(original_frames[0]['transform_matrix']))/2.0).tolist()
    frames.append(intermediate_frame)
    final_pos.append(np.array(intermediate_frame['transform_matrix'])[:3,3].reshape((1,3)))
    

final_pos = np.array(final_pos).reshape((-1,3))
traj_json = copy.deepcopy(data)
data['frames'] = frames
print(len(frames))
json_path = os.path.join(args.folder,'video.json')
# save json
with open(json_path, 'w') as outfile:
    json.dump(data, outfile, indent=4)


mean_cam_dist = np.linalg.norm(cam_positions,axis=1).mean()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.scatter(cam_positions[:,0],cam_positions[:,1],cam_positions[:,2],c='r',marker='o',s=10)
ax.scatter(final_pos[:,0],final_pos[:,1],final_pos[:,2],c='b',marker='o',s=10)
# first_original_point = np.array(original_frames[0]['transform_matrix'])[0:3,3]
# ax.scatter(final_pos[0,0],final_pos[0,1],final_pos[0,2],c='g',marker='o',s=10) # first point 
# ax.scatter(first_original_point[0],first_original_point[1],first_original_point[2],c='g',marker='o',s=10) # first original point
# last_original_point = np.array(original_frames[-1]['transform_matrix'])[0:3,3]

# ax.scatter(last_original_point[0],last_original_point[1],last_original_point[2],c='r',marker='o',s=10) # last original point
# ax.scatter(final_pos[-1,0],final_pos[-1,1],final_pos[-1,2],c='r',marker='o',s=10) # last point
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_aspect('equal', adjustable='box')
ax.set_title('Camera Positions')
plt.show()
# plt.savefig(os.path.join(args.folder,'camera_positions.png'))