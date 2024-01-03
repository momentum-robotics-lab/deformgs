import argparse 
import numpy as np 
import glob 
import matplotlib.pyplot as plt 
import natsort
from iso_psnr_manual_points import manual_points
parser = argparse.ArgumentParser()
parser.add_argument('-i','--input',default = "output/iso_ablation")
args = parser.parse_args()

result_files = glob.glob(args.input + "/**/results.txt",recursive=True)

class Measurement:
    def __init__(self,scene,iso,psnr=None,ssim=None,lpips=None):
        self.scene = scene
        self.iso = iso
        self.psnr = psnr
        self.ssim = ssim
        self.lpips = lpips
        
        # self.print()
    
    def print(self):
        print("scene: {}, iso: {}, psnr: {}, ssim: {}, lpips: {}".format(self.scene,self.iso,self.psnr,self.ssim,self.lpips))
        
class Scene:
    def __init__(self,scene):
        self.scene = scene
        self.measurements = []
    
    def add_measurement(self,measurement):
        self.measurements.append(measurement)
        self.sort_by_iso()
        
    def print(self):
        print("scene: {}".format(self.scene))
        for measurement in self.measurements:
            measurement.print()
        print()
    
    def sort_by_iso(self):
        self.measurements = sorted(self.measurements,key=lambda x: x.iso)

all_scenes = []
measurements = []
for result_file in result_files:
    folder = result_file.split("/")[-2]
    scene = folder.split("_")[0] + "_" + folder.split("_")[1]
    all_scenes.append(scene)
    iso = float(folder.split("_")[-1])
    
    
    with open(result_file,'r') as f:
        lines = f.readlines()
        PSNR, SSIM, LPIPS = None, None, None
        for line in lines:
            if "PSNR" in line:
                PSNR = float(line.split(":")[-1])
            elif "SSIM" in line:
                SSIM = float(line.split(":")[-1])
            elif "LPIPS" in line:
                LPIPS = float(line.split(":")[-1])
        measurements.append(Measurement(scene,iso,psnr=PSNR,ssim=SSIM,lpips=LPIPS))

for manual_point in manual_points:
    measurements.append(Measurement(manual_point.scene,manual_point.iso,psnr=manual_point.psnr))

                
unique_scenes = natsort.natsorted(np.unique(all_scenes))
scene_objects = []
for scene in unique_scenes:
    scene_objects.append(Scene(scene))

for measurement in measurements:
    for scene_object in scene_objects:
        if measurement.scene == scene_object.scene:
            
            scene_object.add_measurement(measurement)

for scene_object in scene_objects:
    scene_object.print()

# plot the PSNR
scene_4 = scene_objects[-1]
scene_objects = scene_objects[:-1]
#insert back in 
scene_objects.insert(3,scene_4)
plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(12,2))
for scene_object in scene_objects:
    iso = [x.iso for x in scene_object.measurements]
    psnr = np.array([x.psnr for x in scene_object.measurements])
    psnr /= psnr[0]
    scene = scene_object.scene
    if scene == 'scene_7':
        scene = 'scene_4'
    
    scene = scene.replace("_"," ")
    plt.plot(iso,psnr,label=scene)
    
plt.xlabel(r'$\mathcal{L}^{{iso}}$')
plt.ylabel("PSNR [-] \n Normalized")
# put legend right above the plot without intersecting with the plot
# plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3)

# log scale on x-axis
plt.xscale('log')
# set sticks to 10^-2 -> 10^0 
plt.xticks([0.01,0.1,1])
plt.grid()
plt.savefig("PSNR.pdf",bbox_inches='tight')

