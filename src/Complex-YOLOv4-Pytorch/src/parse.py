import numpy as np
import statistics

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

f = open(f"/home/mj/Downloads/nano_result/nano_v3_tiny_eval_result/f_AGX_iou_0.5.txt", 'r')

line = f.readline()

FPS = []
CPU = []
GPU = []
RAM = []
GPU_cur = []
CPU_cur = []

while line:
    # print(line)
    parse = [name.strip() for name in line.split(' ')]
    # print(parse)
    FPS.append(float(parse[0]))
    CPU.append(float(parse[1]))
    GPU.append(float(parse[2]))
    RAM.append(float(parse[3]))
    CPU_cur.append(float(parse[4]))
    v = rreplace(parse[5], '.', '', 0)
    print(v)
    GPU_cur.append(float(v))

    # print(float(parse[0]))
    line = f.readline()    
f.close()

#     # mAP, FPS, power/W, power/W, GB
#     # A_data = f"{AP.mean()} {statistics.mean(FPS)} {statistics.mean(CPU)} {statistics.mean(GPU)} {statistics.mean(RAM)} {statistics.mean(CPU_cur)} {statistics.mean(GPU_cur)}.\n" 
# for i in range(len(FPS)):
#     A_data = f"{FPS[i]} {CPU[i]} {GPU[i]} {RAM[i]} {CPU_cur[i]} {GPU_cur[i]}.\n" 
#     f.write(A_data)

p = open(f"/home/mj/Downloads/nano_result/nano_v3_tiny_eval_result/f_nano_mean_iou_0.5.txt", 'w')    
p_data = f"mean_FPS: {statistics.mean(FPS)} fps \nmean_CPU_avg: {statistics.mean(CPU)} power/w \nmean_GPU_avg: {statistics.mean(GPU)} power/w \nmean_RAM: {statistics.mean(RAM)} GB \nmean_CPU_cur: {statistics.mean(CPU_cur)} power/w \nmean_GPU_cur: {statistics.mean(GPU_cur)} power/w "
print(p_data)
p.write(p_data)
p.close()
