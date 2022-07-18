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
from jtop import jtop

# from utils.misc import time_synchronized

jetson = jtop()
jetson.start() # MJ (210810) the position of this statement should be change according to platforms (i.e., AGX, NANO)
# record = Usage()
FPS = []
POW = []
# CPU = []
# GPU = []
RAM = []
CPU_usg = []
GPU_usg = []

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

cfg = Config.fromfile('configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo_mini.py')

model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

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

checkpoint = load_checkpoint(model, '/home/mj/YOLO/source/CenterPoint/checkpoints/latest.pth', map_location="cpu")
model.eval()

model = model.cuda()

gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")
points_list = [] 
gt_annos = [] 
detections = {}
detections_for_draw = []
points_list = []
token_list = []

#

for i, data_batch in enumerate(data_loader):
    token = data_batch['metadata'][0]['token']
    token_list.append(token)
    
    # save points data for tensorrt
    data_batch["points"].cpu().numpy()[:,1:].astype(np.float32).tofile( \
                      "./tensorrt/data/centerpoint/points/%s.bin"%token)    
    # points_list for visulize
    points = data_batch['points'][:, 1:4].cpu().numpy()
    points_list.append(points.T)
    start_time = time.time()
    with torch.no_grad():
        outputs = batch_processor(
            model, data_batch, train_mode=False, local_rank=0
        )
    info = dataset._nusc_infos[i]
    gt_annos.append(convert_box(info))
    for output in outputs:
        token = output["metadata"]["token"]
        for k, v in output.items():
            if k not in [
                "metadata",
            ]:
                output[k] = v.to(cpu_device)
        detections_for_draw.append(output)
        detections.update(
            {token: output,}
        )
    FPS.append(1.0/(time.time() - start_time))
    # CPU.append(jetson.power[1]['CPU']['avg']/1000)
    # GPU.append(jetson.power[1]['GPU']['avg']/1000)
    POW.append(jetson.power[0]['avg']/1000)
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
jetson.close()

all_predictions = all_gather(detections)

predictions = {}
for p in all_predictions:
    predictions.update(p)

result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir="./", testset=False)

if result_dict is not None:
    for k, v in result_dict["results"].items():
        print(f"Evaluation {k}: {v}")
p = open(f"/home/mj/YOLO/source/ssd/CIA-SSD/usg_data/centerPth_mean_iou_0.75.txt", 'a')   
p_data = f"{statistics.mean(FPS)} {statistics.mean(POW)} {statistics.mean(RAM)} {statistics.mean(CPU_usg)} {statistics.mean(GPU_usg)}\n" 
p.write(p_data)
f = open(f"/home/mj/YOLO/source/ssd/CIA-SSD/usg_data/centerPth_inference_iou_0.75.txt", 'w')  
for i in range(len(FPS)):
    A_data = f"{FPS[i]} {POW[i]} {RAM[i]} {GPU_usg[i]}\n" 
    f.write(A_data)
f.close()
g = open(f"/home/mj/YOLO/source/ssd/CIA-SSD/usg_data/centerPth_cpuUsg_iou_0.75.txt", 'w')  
for i in range(len(CPU_usg)):
    g_data = f"{CPU_usg[i]}\n" 
    g.write(g_data)
g.close()
