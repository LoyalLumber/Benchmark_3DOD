import numpy as np
import statistics
from jtop import jtop

CPU = []
GPU = []
RAM = []
GPU_cur = []
CPU_cur = []

for i in range(1):
    jetson = jtop()
    jetson.start()

    CPU.append(jetson.power[1]['CPU']['avg']/1000)
    GPU.append(jetson.power[1]['GPU']['avg']/1000)
    CPU_cur.append(jetson.power[1]['CPU']['cur']/1000)
    GPU_cur.append(jetson.power[1]['GPU']['cur']/1000)
    RAM.append(jetson.ram['use'] / 2.**20)
    print(jetson.cpu['CPU1'])
    print(jetson.gpu['val'])


    jetson.close()

# p = open(f"/home/mj/YOLO/source/Complex-YOLOv4-Pytorch/idle.txt", 'a')    
# p_data = f"mean_CPU_avg: {statistics.mean(CPU)} power/w \nmean_GPU_avg: {statistics.mean(GPU)} power/w \nmean_RAM: {statistics.mean(RAM)} GB \nmean_CPU_cur: {statistics.mean(CPU_cur)} power/w \nmean_GPU_cur: {statistics.mean(GPU_cur)} power/w "
# print(p_data)
# p.write(p_data)
# p.close()