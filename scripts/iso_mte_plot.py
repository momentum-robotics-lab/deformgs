import pickle 
import argparse 
import natsort 
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument("--input",type=str,default="/data/bart/4DGaussians/eval_dict_iso.pkl")
args = parser.parse_args()

class Measurement:
    def __init__(self,scene,iso,mte=None):
        self.scene = scene
        self.iso = iso
        self.mte = mte
        
        # self.print()
    
    def print(self):
        print("scene: {}, iso: {}, mte: {}".format(self.scene,self.iso,self.mte))
        
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



with open(args.input,'rb') as f:
    data = pickle.load(f)

all_mte = data['$3D~MTE~\\downarrow$']
all_isos = list(all_mte.keys())
all_measurements = []
all_scenes = []
for iso in all_isos:
    scenes = list(all_mte[iso].keys())
    for scene in scenes:
        all_measurements.append(Measurement(scene,float(iso),float(all_mte[iso][scene])))
        all_scenes.append(scene)
# print(all_isos)


unique_scenes = natsort.natsorted(np.unique(all_scenes))
scene_objects = []
for scene in unique_scenes:
    scene_objects.append(Scene(scene))

for measurement in all_measurements:
    for scene_object in scene_objects:
        if measurement.scene == scene_object.scene:
            
            scene_object.add_measurement(measurement)

for scene_object in scene_objects:
    scene_object.print()

# print(data)