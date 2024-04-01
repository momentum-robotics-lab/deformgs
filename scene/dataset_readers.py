#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.hyper_loader import Load_hyper_data, format_hyper_data
import torchvision.transforms as transforms
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
import h5py
import cv2 

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    view_id: int 
    time_id: int
    flow: np.array = None
    
    c_x: np.array = None 
    c_y: np.array = None 
    f_x: np.array = None
    f_y: np.array = None
    
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int
    all_times: np.array = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = PILtoTorch(image,None)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time = 0)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path,panopto=False):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    #if panopto:
        # flip z axis
    #    positions[:,2] = -positions[:,2]
    
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

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

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=train_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime,time_skip=None,single_cam_video=False):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    n_poses = 80
    if single_cam_video == False:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,n_poses+1)[:-1]], 0)
    else:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.ones(n_poses)*-90 ], 0)

    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        if "camera_angle_x" in template_json.keys():
            fovx = template_json["camera_angle_x"]
        else:
            fovx = None
    # load a single image to get image info.
    
        
    
    for idx, frame in enumerate(template_json["frames"]):
        file_path = frame["file_path"]
        viable_extensions = [".png", ".jpg", ".jpeg"]
        if not any([file_path.endswith(ext) for ext in viable_extensions]):
            file_path += extension
        cam_name = os.path.join(path, file_path)
        
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        # image = PILtoTorch(image,(800,800))
        image = PILtoTorch(image,None)
        break
    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]

        if fovx is not None:
            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                                time = time,time_id=None,view_id=None))
            
        else:
            k = np.array(frame['k'])
            f_x = k[0][0]
            f_y = k[1][1]
            c_x = k[0][2]
            c_y = k[1][2]
            w = frame['w']
            h = frame['h']
            
            FovX = w / (2 * f_x)
            FovY = h / (2 * f_y)
            
            fovx = None 

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h,
                            time = time,view_id=None,time_id=None,flow=None,c_x=c_x,c_y=c_y,f_x=f_x,f_y=f_y))

    return cam_infos


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", mapper = {},time_skip=None,view_skip=None,split='train',panopto=False,scale=None):
    cam_infos = []
    
    flow_file = os.path.join(path, 'optic_flow',split, "optic_flow.h5")
    imgpaths_file = os.path.join(path, 'optic_flow',split, "img_paths.npy")
    # if os.path.exists(flow_file) and os.path.exists(imgpaths_file):
    if False:
        data =  h5py.File(flow_file,'r')
        print("Loading optic flow..")
        all_flow = data['flow'][:]
        print("optic flow shape: ",all_flow.shape)
        
        print("Finished loading optic flow..")
        img_paths_flow = np.load(imgpaths_file)

    else:
        print("No optic flow found")
        all_flow = None
        img_paths_flow = None

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = None 
        if "camera_angle_x" in contents.keys():
            fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        
        unique_times = np.unique([frame["time"] for frame in frames])
        unique_transforms = np.unique(np.stack([np.array(frame["transform_matrix"]) for frame in frames]),axis=0)

        if time_skip is not None:
            # find all unique timesteps
            
            kept_times = unique_times[::time_skip]


        
        for idx, frame in tqdm(enumerate(frames),total=len(frames)):
            time = frame["time"]
            
            if time_skip is None or time in kept_times:
                # if file_path ends with an extension don't add the extension 
                file_path = frame["file_path"]
                viable_extensions = [".png", ".jpg", ".jpeg"]
                if not any([file_path.endswith(ext) for ext in viable_extensions]):
                    file_path += extension
                
                file_name = file_path.split("/")[-1]
                if any([file_path.endswith(ext) for ext in viable_extensions]):
                    file_name = file_name.split(".")[0]
                
                # format is r_viewid_timeid
                
                if len(file_name.split("_")) > 2:
                    view_id = int(file_name.split("_")[-2])
                    time_id = int(file_name.split("_")[-1])
                else:
                    # compute view_id and time_id based on unique transforms and times
                    view_id = np.where(np.all(unique_transforms == np.array(frame["transform_matrix"]),axis=1))[0][0]
                    time_id = np.where(unique_times == frame["time"])[0][0]
                
                if view_skip is not None:
                    if view_id % view_skip != 0:
                        continue
                    
                flow = None
                # check if file_path is in img_paths
                if img_paths_flow is not None:
                    if file_path in img_paths_flow:
                        flow = all_flow[img_paths_flow == file_path]

                
                cam_name = os.path.join(path, file_path)
                # time = mapper[frame["time"]]
                time = frame["time"]

                if panopto == False:
                    matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
                    R = -np.transpose(matrix[:3,:3])
                    R[:,0] = -R[:,0]
                    T = -matrix[:3, 3]
                else:
                    matrix = np.array(frame["transform_matrix"])
                    R = matrix[:3,:3]
                    T = matrix[:3, 3]

                #matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
                #R = -np.transpose(matrix[:3,:3])
                #R[:,0] = -R[:,0]
                #T = -matrix[:3, 3] 

                image_path = os.path.join(path, cam_name)
                image_name = Path(cam_name).stem
                image = Image.open(image_path)

                im_data = np.array(image.convert("RGBA"))

                if scale is not None:
                    #cv2 resize according to scale 
                    im_data = cv2.resize(im_data,(int(im_data.shape[1]*scale),int(im_data.shape[0]*scale)))

                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                # image = PILtoTorch(image,(800,800))
                image = PILtoTorch(image,None)
                
                if fovx is not None:
                    fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
                    FovY = fovy 
                    FovX = fovx
                    
                    cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[2],
                                time = time,view_id=view_id,time_id=time_id,flow=flow))
                    
                else:
                    k = np.array(frame['k'])
                    if scale is None:
                        f_x = k[0][0]
                        f_y = k[1][1]
                        c_x = k[0][2]
                        c_y = k[1][2]
                        w = frame['w']
                        h = frame['h']
                    else:
                        f_x = k[0][0]*scale
                        f_y = k[1][1]*scale
                        c_x = k[0][2]*scale
                        c_y = k[1][2]*scale
                        w = int(frame['w']*scale)
                        h = int(frame['h']*scale)
                    
                    FovX = focal2fov(f_x,w)
                    FovY = focal2fov(f_y,h)
                    
                    fovx = None 

                    cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                    image_path=image_path, image_name=image_name, width=w, height=h,
                                    time = time,view_id=view_id,time_id=time_id,flow=flow,c_x=c_x,c_y=c_y,f_x=f_x,f_y=f_y))

    return cam_infos


def read_timeline(path):
    with open(os.path.join(path, "transforms_train.json")) as json_file:
        train_json = json.load(json_file)
    with open(os.path.join(path, "transforms_test.json")) as json_file:
        test_json = json.load(json_file)  
    time_line = [frame["time"] for frame in train_json["frames"]] + [frame["time"] for frame in test_json["frames"]]
    time_line = set(time_line)
    time_line = list(time_line)
    time_line.sort()
    timestamp_mapper = {}
    max_time_float = max(time_line)
    for index, time in enumerate(time_line):
        # timestamp_mapper[time] = index
        timestamp_mapper[time] = time/max_time_float

    return timestamp_mapper, max_time_float

def readPanoptoSceneInfo(path, white_background, eval, extension=".png", time_skip=None,view_skip=None,scale=None):
    timestamp_mapper, max_time = read_timeline(path)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, timestamp_mapper, time_skip=time_skip,view_skip=view_skip,split='train',panopto=True,scale=scale)
    
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, timestamp_mapper, time_skip=time_skip,view_skip=view_skip,split='test',panopto=True,scale=scale)
    print("Generating Video Transforms")

    video_path = os.path.join(path, "video.json")
    video_cam_infos = None
    if os.path.exists(video_path):
        video_cam_infos = readCamerasFromTransforms(path, "video.json", white_background, extension, timestamp_mapper, time_skip=time_skip,view_skip=view_skip,split='video',panopto=True)

    if video_cam_infos is None:
        video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time, time_skip=time_skip)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    init_ply_path = os.path.join(path, "init_pt_cld.ply")
    # Since this data set has no colmap data, we start with random point
    
    if not os.path.exists(init_ply_path):
        num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        scene_size = 4.0
        xyz = np.random.random((num_pts, 3)) * scene_size - scene_size / 2
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        pcd = fetchPly(init_ply_path,panopto=True)
        print("Loaded initial point cloud from ",init_ply_path)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", time_skip=None,view_skip=None):
    # time_skip = 4
    timestamp_mapper, max_time = read_timeline(path)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, timestamp_mapper, time_skip=time_skip,view_skip=view_skip,split='train')
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, timestamp_mapper, time_skip=time_skip,view_skip=view_skip,split='test')
    print("Generating Video Transforms")

    # computing all times used 
    all_times = [train_cam.time for train_cam in train_cam_infos] + [test_cam.time for test_cam in test_cam_infos]
    all_times = np.unique(all_times)
    all_times = np.sort(all_times)

    video_path = os.path.join(path, "video.json")
    video_cam_infos = None
    if os.path.exists(video_path):
        video_cam_infos = readCamerasFromTransforms(path, "video.json", white_background, extension, timestamp_mapper, time_skip=1,view_skip=1,split='video')

    if video_cam_infos is None:
        video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time, time_skip=time_skip,single_cam_video=single_cam_video)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # Since this data set has no colmap data, we start with random points
    num_pts = 2000
    num_pts = 2000
    print(f"Generating random point cloud ({num_pts})...")
    
    # We create random points inside the bounds of the synthetic Blender scenes
    scene_size = 2.0
    xyz = np.random.random((num_pts, 3)) * scene_size - scene_size / 2
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time,
                           all_times=all_times
                           )
    return scene_info
def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time))

    return cameras


def readHyperDataInfos(datadir,use_bg_points,eval):
    train_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split ="train")
    test_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split="test")

    train_cam = format_hyper_data(train_cam_infos,"train")
    max_time = train_cam_infos.max_time
    video_cam_infos = copy.deepcopy(test_cam_infos)
    video_cam_infos.split="video"

    ply_path = os.path.join(datadir, "points.npy")

    xyz = np.load(ply_path,allow_pickle=True)
    xyz -= train_cam_infos.scene_center
    xyz *= train_cam_infos.coord_scale
    xyz = xyz.astype(np.float32)
    shs = np.random.random((xyz.shape[0], 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))


    nerf_normalization = getNerfppNorm(train_cam)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )

    return scene_info
def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time = time))
    return cameras


def readdynerfInfo(datadir,use_bg_points,eval):
    # loading all the data follow hexplane format
    ply_path = os.path.join(datadir, "points3d.ply")

    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
    datadir,
    "train",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )    
    test_dataset = Neural3D_NDC_Dataset(
    datadir,
    "test",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )
    train_cam_infos = format_infos(train_dataset,"train")
    
    # test_cam_infos = format_infos(test_dataset,"test")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    # create pcd
    # if not os.path.exists(ply_path):
    # Since this data set has no colmap data, we start with random points
    num_pts = 2000
    print(f"Generating random point cloud ({num_pts})...")
    threshold = 3
    # xyz_max = np.array([1.5*threshold, 1.5*threshold, 1.5*threshold])
    # xyz_min = np.array([-1.5*threshold, -1.5*threshold, -3*threshold])
    xyz_max = np.array([1.5*threshold, 1.5*threshold, 1.5*threshold])
    xyz_min = np.array([-1.5*threshold, -1.5*threshold, -1.5*threshold])
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = (np.random.random((num_pts, 3)))* (xyz_max-xyz_min) + xyz_min
    print("point cloud initialization:",xyz.max(axis=0),xyz.min(axis=0))
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        # xyz = np.load
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=300
                           )
    return scene_info
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Panopto": readPanoptoSceneInfo,
    "dynerf" : readdynerfInfo,
    "nerfies": readHyperDataInfos,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
}
