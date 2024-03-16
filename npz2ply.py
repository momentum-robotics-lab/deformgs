import numpy as np 
from plyfile import PlyData, PlyElement
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='Convert npz to ply')
    parser.add_argument('--input', type=str, required=True, help='input npz file')
    parser.add_argument('--output', type=str, required=True, help='output ply file')
    parser.add_argument('--subsample', type=int, default=50000, help='subsample rate')
    args = parser.parse_args()
    return args


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def load_npz(path,subsample):
    init_pt_cld = np.load(path)
    if 'data' in init_pt_cld:
        init_pt_cld = init_pt_cld["data"]
    else:
        key = list(init_pt_cld.keys())[0]
        init_pt_cld = init_pt_cld[key]
    if subsample > 0:
        idxs = np.random.choice(len(init_pt_cld), subsample, replace=(len(init_pt_cld) < subsample))
        print(f"Subsampling {len(init_pt_cld)} points to {subsample}")
        init_pt_cld = init_pt_cld[idxs]

    xyz =  init_pt_cld[:, :3]
    rgb = init_pt_cld[:, 3:6]

    return xyz, rgb

if __name__ == "__main__":
    args = parse_args()
    xyz, rgb = load_npz(args.input,args.subsample)
    storePly(args.output, xyz, rgb)