# MD-Splatting: Learning Metric Deformation from 4D Gaussians in Highly Deformable Scenes

## arXiv Preprint

### [Project Page](https://md-splatting.github.io/)| [arXiv Paper](https://arxiv.org/abs/2312.00583)

---------------------------------------------------

---

![block](assets/teaserfig.png)   


## Installation 

**Docker Image**

We use docker to run our code, you will need to install docker and [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). You can build the docker image by running the following command:
```
docker build -f md_splatting.dockerfile -t md_splatting .
```

If you don't want to build the docker image, you can pull a pre-built image from docker hub:
```
docker pull bartduis/md_splatting:latest
```

Now create a container from the image and run it.
``` 
docker run -it --gpus all --network=host --shm-size=2G  --name md_splatting -v /home/username:/workspace md_splatting
cd /workspace 
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```
At this point your container is ready to run the code.


## Data
**For synthetic scenes:**  
The dataset provided [here](https://drive.google.com/drive/folders/116XTLBUvuiEQPjKXKZP8fYab3F3L1cCd?usp=sharing) can be used with MD-Splatting to enable novel view synthesis and dense tracking. After downloading the dataset, extract the files to the `data` folder. The folder structure should look like this:

```
├── data
│   | final_scenes 
│     ├── scene_1
│     ├── scene_2 
│     ├── ...
```


## Training
To train models for all scenes from the paper, run the following script:
``` 
./run_scripts/run_all.sh
``` 

## Rendering
Run the following script to render images for all scenes. 

```
./run_scripts/render_all.sh
```

## Run Scripts

There are some other useful scripts in the run_scripts directory. Some of it is messy and needs to be cleaned up, but they'll allow you to easily run ablations and log the results.

## How to prepare your dataset?




---
## Contributions

---
Some source code of ours is borrowed from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [k-planes](https://github.com/Giodiro/kplanes_nerfstudio),[HexPlane](https://github.com/Caoang327/HexPlane), [TiNeuVox](https://github.com/hustvl/TiNeuVox), [4DGS](https://github.com/hustvl/4DGaussians). We appreciate the excellent works of these authors.



## Citation
```
@misc{duisterhof2023mdsplatting,
      title={MD-Splatting: Learning Metric Deformation from 4D Gaussians in Highly Deformable Scenes}, 
      author={Bardienus P. Duisterhof and Zhao Mandi and Yunchao Yao and Jia-Wei Liu and Mike Zheng Shou and Shuran Song and Jeffrey Ichnowski},
      year={2023},
      eprint={2312.00583},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
