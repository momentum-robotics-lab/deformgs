from argparse import ArgumentParser
import json 
import numpy as np

def main():
    parser = ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    args = parser.parse_args()


    with open(args.json, 'r') as f:
        data = json.load(f)

    f_x = data["fl_x"]
    f_y = data["fl_y"]
    c_x = data["cx"]
    c_y = data["cy"]
    k = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

    for i in range(len(data["frames"])):
        data["frames"][i]["k"] = k.tolist()

    output = args.json.replace(".json", "_panoptic.json")
    with open(output, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()