import psutil
import os
from jtop import jtop
  
# # Getting loadover15 minutes
# load1, load5, load15 = psutil.getloadavg()
  
# cpu_usage = (load15/os.cpu_count()) * 100
  
# print("The CPU usage is : ", cpu_usage)

# # Getting all memory using os.popen()
# total_memory, used_memory, free_memory = map(
#     int, os.popen('free -t -m').readlines()[-1].split()[1:])
  
# # Memory usage
# print("RAM memory % used:", round((used_memory/total_memory) * 100, 2))

# print("=="*20)
# print("== memory usage check")

# for exec_num in range(0, 2):
        # Getting all memory using os.popen()
    # total_memory, used_memory, free_memory = map(
    # int, os.popen('free -t -m').readlines()[-1].split()[1:])

    # Memory usage

    # print("RAM memory % used:", round((used_memory/total_memory) * 100, 2))
    # BEFORE code
    # print(f"== {exec_num:2d} exec")
    # general RAM usage
  
def ram_usage():

    memory_usage_dict = dict(psutil.virtual_memory()._asdict())
    memory_usage_percent = memory_usage_dict['percent']
    mem_total = memory_usage_dict['total']
    print(f"CODE: Out of total {mem_total} memory_usage_percent: {memory_usage_percent}%")
    # current process RAM usage
    pid = os.getpid()
    current_process = psutil.Process(pid)
    current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2.**20
    print(f"CODE: Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")
    return memory_usage_percent, current_process_memory_usage_as_KB

def cpu_usage():

    cpu = psutil.cpu_times_percent()
    free = cpu.idle
    used = 100 - free
    print(f"CODE: current CPU_usage_percent: {used}%")
    return used 

def gpu_usage():

    print("Simple jtop reader")
    with jtop() as jetson:
        while(jetson.ok()):
            print(jetson.stats['GPU'])
        
    # with jtop() as jetson:
        # # jetson.ok() will provide the proper update frequency
        # while jetson.ok():
        #     # Read tegra stats
        #     print(jetson.stats['GPU'])

    