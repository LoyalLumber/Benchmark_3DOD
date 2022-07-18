import numpy as np
import statistics
from jtop import jtop

# CPU = []
# GPU = []
# RAM = []
# GPU_cur = []
# CPU_cur = []
CPU_usg = []
GPU_usg = []
    
for i in range(1):
    jetson = jtop()
    jetson.start()

    # CPU.append(jetson.power[1]['CPU']['avg']/1000)
    # GPU.append(jetson.power[1]['GPU']['avg']/1000)
    # CPU_cur.append(jetson.power[1]['CPU']['cur']/1000)
    # GPU_cur.append(jetson.power[1]['GPU']['cur']/1000)
    # RAM.append(jetson.ram['use'] / 2.**20)
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

p = open(f"/home/mj/YOLO/source/Complex-YOLOv4-Pytorch/idle_usg.txt", 'a')    
p_data = f"{statistics.mean(CPU_usg)} {statistics.mean(GPU_usg)}\n" 
p.write(p_data)
p.close()