import matplotlib as plt
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from shapely.geometry import Polygon
import utils.config as cnf
import utils.kitti_bev_utils as bev_utils

# read values from text files with evatuation results
IOU_info = []
FPS_info = []
CPU_info = []
RAM_percent_info = []
RAM_usage_info = []
GPU_info = []

for i in range(10):

    with open('readme.txt') as f:
        lines = f.readlines()
        print(lines)
        IOU_info.append(lines[0])
        FPS_info.append(lines[1])
        CPU_info.append(lines[2])
        RAM_percent_info.append(lines[3])
        RAM_usage_info.append(lines[4])
        GPU_info.append(lines[5])
    f.close()

# make a plot graph with values 


# save it in an image file

