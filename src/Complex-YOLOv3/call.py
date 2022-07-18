import os

def execute(iou):                                                             
    os.system(f'python eval_mAP.py --iou_thres {iou}')  

for iou in range(50, 100, 5):
    iou = iou/100.0
    print(iou)
    execute(iou)
