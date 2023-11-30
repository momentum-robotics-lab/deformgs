
manual_points = []
scenes = ['scene_1','scene_2', 'scene_3', 'scene_7', 'scene_5', 'scene_6']
class ManualPoint:
    def __init__(self,scene,iso,psnr):
        self.scene = scene
        self.iso = iso
        self.psnr = psnr

    def print(self):
        print("scene: {}, iso: {}, psnr: {}".format(self.scene,self.iso,self.psnr))

iso = 0.01

psnrs = [40.67, 39.62, 43.27, 42.61, 33.46, 34.96]

for scene, psnr in zip(scenes,psnrs):
    manual_points.append(ManualPoint(scene,iso,psnr))

iso = 1.0
psnrs = [38.76, 38.16, 40.34, 40.61, 32.47, 32.39]
for scene, psnr in zip(scenes,psnrs):
    manual_points.append(ManualPoint(scene,iso,psnr))

