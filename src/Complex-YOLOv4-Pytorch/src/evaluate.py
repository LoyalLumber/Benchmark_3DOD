import argparse
import os
import time
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.utils.data.distributed
from tqdm import tqdm
from easydict import EasyDict as edict

sys.path.append('./')

from data_process.kitti_dataloader import create_val_dataloader
from models.model_utils import create_model
from utils.misc import AverageMeter, ProgressMeter
from utils.evaluation_utils import post_processing, get_batch_statistics_rotated_bbox, ap_per_class, load_classes, post_processing_v2


from utils.misc import time_synchronized
import time
import statistics
from jtop import jtop
# from usage_resource import Usage
# from utils.misc import time_synchronized

jetson = jtop()
 # MJ (210810) the position of this statement should be change according to platforms (i.e., AGX, NANO)
# record = Usage()
FPS = []
POW = []
# CPU = []
# GPU = []
RAM = []
CPU_usg = []
GPU_usg = []
#jetson = jtop() #comment out for the TensorRT Test
def evaluate_mAP(val_loader, model, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    progress = ProgressMeter(len(val_loader), [batch_time, data_time],
                             prefix="Evaluation phase...")
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        jetson.start() #comment out for the TensorRT Test
        start_time = time.time()

        for batch_idx, batch_data in enumerate(tqdm(val_loader)):
            data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale x, y, w, h of targets ((box_idx, class, x, y, w, l, im, re))
            targets[:, 2:6] *= configs.img_size
            imgs = imgs.to(configs.device, non_blocking=True)

            # record.update_time()
            #s_time = time_synchronized()
            #jetson.start() # MJ (210810) - The position for NANO!!!!!
            begin_time = time.time()
            outputs = model(imgs)
            outputs = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            FPS.append(1.0/(time.time() - begin_time))
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
            #e_time = time_synchronized()
            #print('test time: ', 1 / (e_time - s_time))
            # record.FPS_update()
            # record.record_usage(jetson)
           

            sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=configs.iou_thresh)

            # measure elapsed time
            # torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)

            # Log message
            if logger is not None:
                if ((batch_idx + 1) % configs.print_freq) == 0:
                    logger.info(progress.get_message(batch_idx))

            start_time = time.time()
            # record.update_time()

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
        #jetson.close()
    jetson.close()

    return precision, recall, AP, f1, ap_class
    #, FPS, CPU, GPU, RAM, CPU_cur, GPU_cur, CPU_usg, GPU_usg


def parse_eval_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--classnames-infor-path', type=str, default='../dataset/kitti/classes_names.txt',
                        metavar='PATH', help='The class names of objects in the task')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4_tiny.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default='../checkpoints/complex_YOLOv4/Model_complex_yolov4_tiny_epoch_295.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for class conf')
    parser.add_argument('--nms-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for nms')
    parser.add_argument('--iou-thresh', type=float, default=0.7,
                        help='for evaluation - the threshold for IoU')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')

    return configs


if __name__ == '__main__':
    configs = parse_eval_configs()
    configs.distributed = False  # For evaluation
    class_names = load_classes(configs.classnames_infor_path)

    model = create_model(configs)
    # model.print_network()
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path,map_location='cuda:0')) #,map_location='cuda:0'is added

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)
    model.eval()

    print('Create the validation dataloader')
    val_dataloader = create_val_dataloader(configs)

    print("\nStart computing mAP...\n")
    precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, None)
    #, FPS, CPU, GPU, RAM, CPU_cur, GPU_cur, CPU_usg, GPU_usg
    print("\nDone computing mAP...\n")

    # p = open(f"/home/mj/YOLO/source/Complex-YOLOv4-Pytorch/f_v3_tiny_eval_result/usage_iou_{configs.iou_thresh}.txt", 'a') 

    for idx, cls in enumerate(ap_class):
        print("\t>>>\t Class {} ({}): precision = {:.4f}, recall = {:.4f}, AP = {:.4f}, f1: {:.4f}".format(cls, \
                class_names[cls][:3], precision[idx], recall[idx], AP[idx], f1[idx]))
        # p_data = f"{cls} {class_names[cls][:3]} {AP[idx]}\n" 
        # p.write(p_data)

    #210831 - CJ: Comment below out for the test of TensorRT!!!
    p = open(f"/home/mj/YOLO/source/ssd/CIA-SSD/usg_data/YOLOv4tiny_mean_iou_0.75.txt", 'a')   
    p_data = f"{statistics.mean(FPS)} {statistics.mean(POW)} {statistics.mean(RAM)} {statistics.mean(CPU_usg)} {statistics.mean(GPU_usg)}\n" 
    p.write(p_data)
    f = open(f"/home/mj/YOLO/source/ssd/CIA-SSD/usg_data/YOLOv4tiny_inference_iou_0.75.txt", 'w')  
    for i in range(len(FPS)):
        A_data = f"{FPS[i]} {POW[i]} {RAM[i]} {GPU_usg[i]}\n" 
        f.write(A_data)
    f.close()
    g = open(f"/home/mj/YOLO/source/ssd/CIA-SSD/usg_data/YOLOv4tiny_cpuUsg_iou_0.75.txt", 'w')  
    for i in range(len(CPU_usg)):
        g_data = f"{CPU_usg[i]}\n" 
        g.write(g_data)
    g.close()
