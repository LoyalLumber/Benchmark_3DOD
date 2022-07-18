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

# for analysis
import time
import statistics
from jtop import jtop

def evaluate(model, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    
    FPS = []
    CPU = []
    GPU = []
    RAM = []
    CPU_cur = []
    GPU_cur = []

    jetson = jtop()
    jetson.start()
 
    # Get dataloader
    split='valid'
    dataset = KittiYOLODataset(cnf.root_dir, split=split, mode='EVAL', folder='training', data_aug=False) #load evauation data set

    # Pre_process datasets
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            start_time = time.time() # start time

            outputs = model(imgs)
            outputs = non_max_suppression_rotated_bbox(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            # P_data = f"{jetson.power[1]['GPU']['avg']} {jetson.power[1]['CPU']['avg']}.\n" 
            # f2.write(P_data)

            FPS.append(1.0/(time.time() - start_time))
            CPU.append(jetson.power[1]['CPU']['avg']/1000)
            GPU.append(jetson.power[1]['GPU']['avg']/1000)
            CPU_cur.append(jetson.power[1]['CPU']['cur']/1000)
            GPU_cur.append(jetson.power[1]['GPU']['cur']/1000)
            RAM.append(jetson.ram['use'] / 2.**20)

            #print("--- %s FPS ---" % (1.0/(time.time() - start_time))) # end time, FPS

        sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    jetson.close()

    return precision, recall, AP, f1, ap_class, FPS, CPU, GPU, RAM, CPU_cur, GPU_cur

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/complex_tiny_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/tiny-yolov3_ckpt_epoch-220.pth", help="path to weights file")
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

    print("Compute mAP...")
    precision, recall, AP, f1, ap_class, FPS, CPU, GPU, RAM, CPU_cur, GPU_cur = evaluate(
        model,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size
        )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
    print(f"FPS_mean: {statistics.mean(FPS)}")

    f = open(f"/home/mj/YOLO/source/Complex-YOLOv3/f_v3_tiny_eval_result/f_AGX_fps_iou_{opt.iou_thres}.txt", 'a')    
    for i in range(len(FPS)):
        A_data = f"{FPS[i]} {CPU[i]} {GPU[i]} {RAM[i]} {CPU_cur[i]} {GPU_cur[i]}.\n" 
        f.write(A_data)

    f.close()

    p = open(f"/home/mj/YOLO/source/Complex-YOLOv3/f_v3_tiny_eval_result/f_AGX_mAP_iou_{opt.iou_thres}.txt", 'a')    
    p_data = f"{AP.mean()}.\n" 
    p.write(p_data)

    p.close()
