from argparse import ArgumentParser
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob 
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='workspace')
    parser.add_argument('--slice',type=int,default=10)
    parser.add_argument('--z_max',type=float,default=None)
    parser.add_argument('--animation',action='store_true')
    args = parser.parse_args()
    return args

def plot_deforms(xyzs, xyzs_deformed,args):
    # make 3d plot 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xyzs_deformed[::args.slice,0], xyzs_deformed[::args.slice,1], xyzs_deformed[::args.slice,2],alpha=0.5)

    # ax.scatter(xyzs[::args.slice,0], xyzs[::args.slice,1], xyzs[::args.slice,2],alpha=0.5)

    # ax.legend(['Metric','Canonical'])

    # # # visualize deforms vectors (N x 3)
    # ax.quiver(xyzs[::args.slice,0], xyzs[::args.slice,1], xyzs[::args.slice,2], 
    #           (xyzs_deformed[::args.slice,0]-xyzs[::args.slice,0]), (xyzs_deformed[::args.slice,1]-xyzs[::args.slice,1]), (xyzs_deformed[::args.slice,2]-xyzs[::args.slice,2]))

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    ax.set_aspect('equal', adjustable='box')



    if args.animation:

        def rotate(angle):
            ax.view_init(azim=angle)
        
        rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=30)
        rot_animation.save('rotation.gif', dpi=80, writer='imagemagick')


    plt.show()


def plot(trajs,xyszs,times):
    print('tada!')

def main():
    args = parse_args()

    npz_files = glob.glob(os.path.join(args.dir,'log_deform_*.npz'),recursive=True)
    # sort based on the float number in the file name
    npz_files.sort(key=lambda f: float(''.join(filter(str.isdigit, f))))
    times = [float(''.join(filter(str.isdigit, os.path.basename(f)) )) for f in npz_files]
   
    for npz_file in npz_files:
        deforms_data = np.load(npz_file)
        xyzs = deforms_data['means3D']
        xyzs_deformed = deforms_data['means3D_deform']

        if args.z_max is not None:
            xyzs = xyzs[xyzs_deformed[:,2] < args.z_max]
            xyzs_deformed = xyzs_deformed[xyzs_deformed[:,2] < args.z_max]

        print('xyzs shape: ', xyzs.shape)
        print('xyz_deformed shape: ', xyzs_deformed.shape)
    

    # plot_deforms(xyzs, xyzs_deformed,args)


if __name__ == '__main__':
    main()