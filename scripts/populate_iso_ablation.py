import argparse
import os 
import glob
import natsort
import shutil 

parser = argparse.ArgumentParser()
parser.add_argument('--results',type=str,default="/data/bart/CVPR_2024/results_iso_ablation/")
parser.add_argument('--output',type=str,default="/data/bart/4DGaussians/output/iso_ablation/")
parser.add_argument('--test_path',type=str,default="test/ours_20000")
parser.add_argument("--executable",type=str,default="/data/bart/4DGaussians/scripts/align_eval_trajs.py")
args = parser.parse_args()

output_dirs = natsort.natsorted(glob.glob(os.path.join(args.output,"**")))
# filter out non directories
output_dirs = [x for x in output_dirs if os.path.isdir(x)]
scenes = [x.split("/")[-1] for x in output_dirs]
iso_nums = [float(x.split("_")[-1]) for x in scenes]
scenes = [x.split("_")[0] + "_" + x.split("_")[1] for x in scenes]

output_dirs = [os.path.join(x,args.test_path) for x in output_dirs]
trajs_paths = [os.path.join(x,"all_trajs.npz") for x in output_dirs]

results_dirs = [os.path.join(args.results,scene) for scene in scenes]
gt_paths = [os.path.join(x,"gt.npz") for x in results_dirs]

for iso, gt_path, traj_path, results_dir in zip(iso_nums, gt_paths,trajs_paths, results_dirs):
    print("gt_path: {}".format(gt_path))
    print("traj_path: {}".format(traj_path))
    print("results dir: {}".format(results_dir))
    # exit()
    
    command = "python3 {} --gt_file {} --traj_file {}".format(args.executable,gt_path,traj_path)
    
    # execute command
    os.system(command)
    
    output_dir = os.path.join(results_dir,str(iso))
    # delete the output dir if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    source_file = traj_path.replace(".npz","_aligned.npz")
    output_file = os.path.join(output_dir,"traj.npz")
    
    print("Copying the output")
    # copy the aligned traj to the output dir
    shutil.copyfile(source_file,output_file)