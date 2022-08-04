# Run your deep learning-based 3D object detectors on NVIDIA Jetsons

<div align="center">

<p align="center">
<img width="500" src="https://github.com/LoyalLumber/Benchmark_3DOD/blob/a963d08bb802177554c17b67d8bea5a27b0d5898/samples/sample.png"><br>
</p>

</div>

# Contents
    
1. [Introduction](#Introduction)
2. [Environment](#Environment)
3. [Datasets](#Datasets)
4. [Run](#Run)
5. [Frameworks](#Frameworks)
6. [Citation](#Citation)
7. [Acknowledgement](#Acknowledgement)
<br><br>

## Introduction

This repository provides a benchmark tool for well-known deep learning-based 3D detectors on NVIDIA Jeston boards. 
Currently, we provide benchmarks of 12 detectors (Check (#Frameworks)!).
We have tested the tool on the four Jetson series including AGX, NX, TX2, and Nano. 

The work analyzes frame per second (FPS) and resource usages (CPU, GPU, RAM, Power consumption) of each detector on the Jetsons. 

### Clone and install requirements
    1. git clone "this repository" 
    2. sudo pip install -r requirements.txt

### Download pre-trained weights 
    1. cd weights/
    2. bash download_weights.sh

## Environment

- Jetpack 4.4.1
- CUDA Toolkit 10.2
- Python 3.6.9
- Please check "requirements.txt" for the detailed libraries. 


## Datasets

We run the benchmak using two datasets: KITTI and nuScenes. 
You can download the datasets from below links. 

Make sure that place the datasets in 'datasets' folder.
- datasets/KITTI/*
- datasets/nuScenes/*

| Dataset | Link |
| :---:        |     :---:  |     
| **KITTI**   | [link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) |
| **nuScenes**   | [link](https://www.nuscenes.org/nuscenes?externalData=all&mapData=all&modalities=Any) |

## Run

Run 'resource_anlyzer.py' in 'src/resource_analyzer' folder.
You need to specify the "--model" and "--output". 

    $  python resource_analyzer.py --model Complex-YOLOv4 --output/C-YOLOv4  


## Frameworks

Thanks for the contributors on 3D detectors. 
Please move to each branch for detailed instructions about source codes. 

| No.    | Dataset | Link |
| :---:        | :---:        |     :---:  |     
|   **1**   | **Complex YOLOv3 w/Tiny version**   | [link](https://github.com/ghimiredhikura/Complex-YOLOv3) |
|   **2**   | **Complex YOLOv4 w/Tiny version**   | [link](https://github.com/maudzung/Complex-YOLOv4-Pytorch) |
|   **3**   | **SECOND**   | [link](https://github.com/open-mmlab/OpenPCDet) |
|   **4**   | **PointPillar**   | [link](https://github.com/open-mmlab/OpenPCDet) |
|   **5**   | **CIA-SSD**   | [link](https://github.com/Vegeta2020/CIA-SSD) |
|   **6**   | **SE-SSD**   | [link](https://github.com/Vegeta2020/SE-SSD) |
|   **7**   | **PointRCNN**   | [link](https://github.com/open-mmlab/OpenPCDet) |
|   **8**   | **Part-A^2**   | [link](https://github.com/open-mmlab/OpenPCDet) |
|   **9**   | **PV-RCNN**   | [link](https://github.com/open-mmlab/OpenPCDet) |
|   **10**   | **CenterPoint**   | [link](https://github.com/tianweiy/CenterPoint) |
|   **11**   | **CenterPoint (TensorRT)**   | [link](https://github.com/CarkusL/CenterPoint) |


## Citation

Not yet available..

```
@article{Soon...
}
```

## Acknowledgement

Not yet available..
