import argparse 
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help='input directory')
args = parser.parse_args()

data = np.load(args.input)
depth = data['depth']
print(depth.shape)