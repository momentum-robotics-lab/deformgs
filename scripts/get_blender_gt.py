import argparse
import bpy
import numpy as np 
import os 
import tqdm

# assume that cloth already exists and is named as 'Deformable'
parser = argparse.ArgumentParser()
parser.add_argument('-i','--input',required=True)
parser.add_argument('-o','--output',default="trajectory.npz")
parser.add_argument('--frame_start', type=int, default=None)
parser.add_argument('--frame_end', type=int, default=None)
parser.add_argument('--n_traj', type=int, default=1000)
args = parser.parse_args()

bpy.ops.wm.open_mainfile(filepath=args.input)
scene = bpy.context.scene
total_frames = scene.frame_end - scene.frame_start + 1

psn_all = []
for frame_no in tqdm.tqdm(range(0, total_frames+1)):
    if args.frame_start is not None:
        if frame_no < args.frame_start:
            continue
    
    if args.frame_end is not None:
        if frame_no > args.frame_end:
            break
        
    scene.frame_set(frame_no)
    psn = []
    
    dg = bpy.context.evaluated_depsgraph_get()
    cloth = bpy.context.scene.objects['Deformable'].evaluated_get(dg)
    mesh = cloth.to_mesh(preserve_all_data_layers=True, depsgraph=dg)

    for vertex in mesh.vertices:
        matrix_world = cloth.matrix_world
        psn.append((matrix_world @ vertex.co)[:])

    psn = np.array(psn)
    
    psn_all.append(psn[None,...])
    
psn = np.concatenate(psn_all, axis=0)

#add '_full' to the file name
full_output = args.output.replace(".npz","_full.npz")
np.savez(full_output, traj=psn)

eval_idxs = np.random.choice(psn.shape[1], args.n_traj, replace=False)
traj_eval = psn[:,eval_idxs,:]

eval_output = args.output.replace(".npz","_eval.npz")
np.savez(eval_output, traj=traj_eval)