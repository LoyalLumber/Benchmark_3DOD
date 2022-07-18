import os

def execute(iou):                                                             
    os.system(f'python evaluate.py --iou-thresh {iou}')  

for iou in [0.5,0.6,0.7,0.75,0.8]:
    print(iou)
    execute(iou)
