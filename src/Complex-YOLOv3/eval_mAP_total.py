from __future__ import division

from models import *
from utils.utils import *

import os, sys, time, datetime, argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

import utils.config as cnf
from utils.kitti_yolo_dataset import KittiYOLODataset

# for time analysis
import time
import statistics

import os
import psutil
from eval_mAP import evaluate
  
# Getting loadover15 minutes
#load1, load5, load15 = psutil.getloadavg()
  
#cpu_usage = (load15/os.cpu_count()) * 100
  
#print("The CPU usage is : ", cpu_usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/complex_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_epoch-298.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=cnf.BEV_WIDTH, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #data_config = parse_data_config(opt.data_config)
    #class_names = load_classes(data_config["names"])
    class_names = load_classes(opt.class_path)

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))

    AP_list = []
    FPS_list = []

    print("Compute mAP_total...")
    for iou in [float(j)/100 for j in range(50, 100, 5)]:
        precision, recall, AP, f1, ap_class, FPS = evaluate(
            model,
            iou_thres=iou, # insert iou = 0.5~0.95
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            img_size=opt.img_size,
            batch_size=opt.batch_size,
        )

        print("Average Precisions:")
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

        print(f"mAP: {AP.mean()}")
        print(f"FPS_mean: {FPS.mean()}")
        FPS_list.append(statistics.mean(FPS))
        AP_list.append(AP.mean())
    
    print("--- %s FPS_total ---" % statistics.mean(FPS_list))
    print("--- %s AP_total ---" % statistics.mean(AP_list))