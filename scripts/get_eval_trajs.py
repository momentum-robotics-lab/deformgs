import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gt_file",type=str,required=True)
parser.add_argument("--n_traj",type=int, required=True)
parser.add_argument("--output",type=str,default=None)
args = parser.parse_args()


import numpy as np
np.random.seed(0)

data = np.load(args.gt_file)
traj_full = data['traj']

eval_idxs = np.random.choice(traj_full.shape[1], args.n_traj, replace=False)
traj_eval = traj_full[:,eval_idxs,:]

if args.output is None:
    np.savez(args.gt_file.replace(".npz","_eval.npz"), traj=traj_eval)
else:
    np.savez(args.output, traj=traj_eval)
