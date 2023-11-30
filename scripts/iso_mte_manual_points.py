
manual_points = []
scenes = ['scene 1','scene 2', 'scene 3', 'scene 7', 'scene 5', 'scene 6']
class ManualPoint:
    def __init__(self,scene,iso,mte):
        self.scene = scene
        self.iso = iso
        self.mte = mte

    def print(self):
        print("scene: {}, iso: {}, mte: {}".format(self.scene,self.iso,self.mte))

iso = 0.01

mtes = [5.173, 63.894, 81.917, 9.449, 5.679, 3.385]

for scene, mte in zip(scenes,mtes):
    manual_points.append(ManualPoint(scene,iso,mte))

iso = 1.0
mtes = [2.881, 46.257, 88.169, 9.686, 4.741, 3.175]
for scene, mte in zip(scenes,mtes):
    manual_points.append(ManualPoint(scene,iso,mte))

