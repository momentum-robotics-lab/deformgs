import matplotlib.pyplot as plt
import numpy as np
from utils import flow_viz
import argparse
import os 
import tqdm
import torch 
from PIL import Image
from utils.utils import InputPadder
import shutil
import cv2 
import matplotlib

DEVICE = "cpu"
def load_image(imgpath):
    img = np.array(Image.open(imgpath)).astype(np.uint8)[:,:,:3] # take out alpha channel if it's there
    
    img = torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE)
    return img


def viz(imgpaths, flos):
    # img = img.cpu().numpy()
    debug_imgs = []

    images = torch.stack([load_image(img_path) for img_path in imgpaths],axis=0)
    imgs1 = images[:-1]
    imgs2 = images[1:]
          
    padder = InputPadder(imgs1.shape)
    imgs1, imgs2 = padder.pad(imgs1, imgs2)

    imgs = imgs1
    for i in tqdm.tqdm(range(imgs.shape[0])):
        plt.clf()
        fig = plt.figure(figsize=(20,10))
        canvas = fig.canvas
        # matplotlib.rcParams['figure.figsize'] = [20, 20] # for square canvas
        ax = fig.gca()

        img = torch.tensor(imgs[i],device=DEVICE).permute(1,2,0).cpu().numpy()
        flo = torch.tensor(flos[i],device=DEVICE).permute(1,2,0).cpu().numpy()
        flo_raw = flo.copy()
        # map flow to rgb image

        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([img, flo], axis=1)

        plt.imshow(img_flo/255.0)

        # 10 random points
        n_points = 150
        found_all_points = False
        magnitude_treshold = 1.5
        found_points = 0
        while not found_all_points:
            x = np.random.randint(0, img.shape[0])
            y = np.random.randint(0, img.shape[1])

            magnitude = np.sqrt(flo_raw[x,y,0]**2 + flo_raw[x,y,1]**2)
            if magnitude < magnitude_treshold:
                continue
            # quiver flo on top of img
            plt.quiver(y, x, flo_raw[x,y,0], flo_raw[x,y,1], color='r', width=0.001,angles='xy', scale_units='xy', scale=0.5)
            found_points += 1
            if found_points == n_points:
                found_all_points = True
        
        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
        # NOTE: reversed converts (W, H) from get_width_height to (H, W)
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
        
        debug_imgs.append(image)
    
    debug_imgs = np.stack(debug_imgs,axis=0)
    return debug_imgs



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',type=str,required=True)
    parser.add_argument('-dir','--dir',type=str,default=None)
    args = parser.parse_args()

    data = np.load(args.input)
    if args.dir is None:
        args.dir = os.path.join(*args.input.split("/")[:-3])
    
    img_paths = data['img_paths']
    img_paths = [os.path.join(args.dir,img_path) for img_path in img_paths]
    flow = data['flow']
    debug_imgs = viz(img_paths,flow)

    if os.path.exists(os.path.join(args.dir,'debug_imgs')):
        shutil.rmtree(os.path.join(args.dir,'debug_imgs'))
    
    os.makedirs(os.path.join(args.dir,'debug_imgs'),exist_ok=True)

    for i in range(debug_imgs.shape[0]):
            # convert img from rgb to bgr 

            cv2.imwrite(os.path.join(args.dir,'debug_imgs',str(i)+'.png'),debug_imgs[i][:,:,::-1] )

