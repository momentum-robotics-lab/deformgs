import numpy as np
import json 
import os 
import argparse
import imageio
import tqdm
def main(args):
    # Load the input json file
    for input in args.input:
        with open(input, 'r') as f:
            data = json.load(f)
        base_dir = os.path.dirname(input)

        file_paths = [os.path.join(base_dir,frame['file_path']) for frame in data['frames']]
        all_transforms = [frame['transform_matrix'] for frame in data['frames']]
        _, idxs = np.unique(all_transforms, axis=0,return_index=True)
        idxs.sort()
        unique_transforms = np.array(all_transforms)[idxs]
        all_times = [frame['time'] for frame in data['frames']]
        unique_times = np.sort(np.unique(all_times))
        
        # goal: every next frame is from new viewpoint but one timestep later
        skip = unique_times.shape[0] + 1
        
        
        print("Loading all images...")
        
        imgs = []
        scene_name = base_dir.split('/')[-1]
        writer = imageio.get_writer(os.path.join(base_dir, scene_name+'_animation.mp4'), fps=args.fps)
        for file_path in tqdm.tqdm(file_paths[::skip]):
            assert os.path.exists(file_path), f"File {file_path} does not exist!"
            img = imageio.imread(file_path) 
            img[img[:,:,-1]==0] = np.ones_like(img[img[:,:,-1]==0])*255
            img=img[:,:,:3]
            imgs.append(img)
            writer.append_data(imgs[-1])
        writer.close()
        
        # imgs = [imageio.imread(file_path) for file_path in file_paths]

        print("Saving GIF animation...")
        # make a GIF animation
        duration = 1000.0/args.fps
        imageio.mimsave(os.path.join(base_dir, 'animation.GIF'), imgs, duration=duration)
        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,nargs='+', help='path to the input file')
    parser.add_argument('-fps', '--fps', type=int, default=30, help='frames per second')
    args = parser.parse_args()
    main(args)