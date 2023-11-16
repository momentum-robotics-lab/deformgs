import numpy as np
import json
import argparse
import os 
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset name", default="corl_1_dense_pano")
parser.add_argument("--n_frames",type=int,default=40)
args = parser.parse_args()


def prune_json(json_name):
    # open json
    with open(json_name,'r') as f:
        data = json.load(f)
    data_new = copy.deepcopy(data)

    frames = data['frames']
    unique_times = np.unique([frame['time'] for frame in frames])
    n_times = len(unique_times)
    valid_times = unique_times[:args.n_frames]
    new_times = np.arange(args.n_frames)/(args.n_frames-1)
    new_frames = []
    delete_filepaths = []
    for frame in frames:
        time = frame['time']
        if time in valid_times:
            new_time = new_times[np.where(valid_times==time)[0][0]]
            frame['time'] = new_time
            new_frames.append(frame)
        else:
            file_path = frame['file_path']
            delete_filepaths.append(os.path.join(args.dataset,file_path))
    

    data_new['frames'] = new_frames
    with open(json_name,'w') as f:
        json.dump(data_new,f,indent=4)

    print('Deleting {} files'.format(len(delete_filepaths)))
    for filepath in delete_filepaths:
        os.remove(filepath)


splits = ['train','test']
for split in splits:
    json_name = os.path.join(args.dataset,'transforms_{}.json'.format(split))
    prune_json(json_name)
