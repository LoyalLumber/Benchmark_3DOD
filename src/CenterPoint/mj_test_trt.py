import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
from torch import nn
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
    example_to_device,
)
from det3d.torchie.trainer import load_checkpoint
import pickle 
import time 
from matplotlib import pyplot as plt 
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import subprocess
import cv2
from tools.demo_utils import visual 
from collections import defaultdict
from det3d.torchie.trainer.utils import all_gather, synchronize
from pathlib import PosixPath
import glob

import time
import statistics
'''from jtop import jtop

# from utils.misc import time_synchronized

jetson = jtop()
jetson.start() # MJ (210810) the position of this statement should be change according to platforms (i.e., AGX, NANO)
# record = Usage()
FPS = []
CPU = []
GPU = []
RAM = []
CPU_usg = []
GPU_usg = []'''

#
def convert_box(info):
    boxes =  info["gt_boxes"].astype(np.float32)
    names = info["gt_names"]

    assert len(boxes) == len(names)

    detection = {}

    detection['box3d_lidar'] = boxes

    # dummy value 
    detection['label_preds'] = np.zeros(len(boxes)) 
    detection['scores'] = np.ones(len(boxes))

    return detection

#
def read_trt_result(path):
    token = path.split("/")[-1].split(".")[0]
    trt_pred = {} 
    with open(path) as f:
        trt_res = f.readlines()

    boxs = []
    box3d = []
    score = []
    cls = []
    for line in trt_res:
        box3d += [np.array([float(it) for it in line.strip().split(" ")[:9]])]
        score += [np.array([float(line.strip().split(" ")[-2])])]
        cls += [np.array([int(line.strip().split(" ")[-1])])]

    trt_pred["box3d_lidar"] = torch.from_numpy(np.array(box3d))
    trt_pred["scores"] = torch.from_numpy(np.array(score))
    trt_pred["label_preds"] = torch.from_numpy(np.array(cls,np.int32))
    trt_pred["metadata"] = {}

    trt_pred["metadata"]["num_point_features"] = 5
    trt_pred["metadata"]["token"] = token
    
    return trt_pred, token

#
cfg = Config.fromfile('configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo_mini.py')
dataset = build_dataset(cfg.data.val)

data_loader = DataLoader(
    dataset,
    batch_size=1,
    sampler=None,
    shuffle=False,
    num_workers=8,
    collate_fn=collate_kitti,
    pin_memory=False,
)

#
points_dict = {}
points_list = []
token_list = []
gt_annos_dict =  {}
for i, data_batch in enumerate(data_loader):
    token = data_batch['metadata'][0]['token']
    token_list.append(token)
    points = data_batch['points'][:, 1:4].cpu().numpy()
    points_dict[token] = points.T
    
    info = dataset._nusc_infos[i]
    gt_annos_dict[token] = convert_box(info)

#
   
trt_pred = {}
detections = {}
detections_for_draw = []
gt_annos = []
res_path_list = glob.glob("/home/mj/YOLO/source/CenterPoint/tensorrt/data/centerpoint/results/*.txt")
output_dict = {}

for path in res_path_list:
    output, token = read_trt_result(path)
    output_dict[token] = output

for token in token_list:    
    points_list.append(points_dict[token])
    start_time = time.time()
    gt_annos.append(gt_annos_dict[token])
    output = output_dict[token]
    for k, v in output.items():
        if k not in [
            "metadata",
        ]:
            output[k] = v
    detections_for_draw.append(output)
    detections.update(
        {token: output,}
    )
    '''FPS.append(1.0/(time.time() - start_time))
    CPU.append(jetson.power[1]['CPU']['avg']/1000)
    GPU.append(jetson.power[1]['GPU']['avg']/1000)
    RAM.append(jetson.ram['use'] / 2.**20)
    CPU_usg.append(jetson.cpu['CPU1']['val'])
    CPU_usg.append(jetson.cpu['CPU2']['val'])
    CPU_usg.append(jetson.cpu['CPU3']['val'])
    CPU_usg.append(jetson.cpu['CPU4']['val'])
    CPU_usg.append(jetson.cpu['CPU5']['val'])
    CPU_usg.append(jetson.cpu['CPU6']['val'])
    CPU_usg.append(jetson.cpu['CPU7']['val'])
    CPU_usg.append(jetson.cpu['CPU8']['val'])
    GPU_usg.append(jetson.gpu['val'])
jetson.close()'''
all_predictions = all_gather(detections)

predictions = {}
for p in all_predictions:
    predictions.update(p)

result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir="./", testset=False)

if result_dict is not None:
    for k, v in result_dict["results"].items():
        print(f"Evaluation {k}: {v}")
'''p = open(f"/home/mj/YOLO/source/ssd/CIA-SSD/usg_data/centerTrt_mean_iou_0.75.txt", 'a')   
p_data = f"{statistics.mean(FPS)} {statistics.mean(CPU)} {statistics.mean(GPU)} {statistics.mean(RAM)} {statistics.mean(CPU_usg)} {statistics.mean(GPU_usg)}\n" 
p.write(p_data)
f = open(f"/home/mj/YOLO/source/ssd/CIA-SSD/usg_data/centerTrt_inference_iou_0.75.txt", 'w')  
for i in range(len(FPS)):
    A_data = f"{FPS[i]} {CPU[i]} {GPU[i]} {RAM[i]} {GPU_usg[i]}\n" 
    f.write(A_data)
f.close()
g = open(f"/home/mj/YOLO/source/ssd/CIA-SSD/usg_data/centerTrt_cpuUsg_iou_0.75.txt", 'w')  
for i in range(len(CPU_usg)):
    g_data = f"{CPU_usg[i]}\n" 
    g.write(g_data)
g.close()'''
#trt
#car Nusc dist AP@0.5, 1.0, 2.0, 4.0
#77.06, 88.78, 91.80, 93.74 mean AP: 0.8784607282526194
#pth
#car Nusc dist AP@0.5, 1.0, 2.0, 4.0
#77.71, 89.48, 92.73, 94.64 mean AP: 0.8864185638415553

