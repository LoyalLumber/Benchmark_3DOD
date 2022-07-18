from jtop import jtop


if __name__ == "__main__":

    print("All accessible jtop properities")
    g = open(f"/home/mj/YOLO/source/ssd/CIA-SSD/usg_data/centerTrt_cpuUsg_iou_0.75.txt", 'a')
    f = open(f"/home/mj/YOLO/source/ssd/CIA-SSD/usg_data/centerTrt_inference_iou_0.75.txt", 'a')  
    with jtop() as jetson:
        # jetson.ok() will provide the proper update frequency
        while jetson.ok():
            # CPU
            print('*** CPUs ***')
            print(jetson.cpu)
            print(jetson.gpu)
            A_data = f"{jetson.power[0]['avg']/1000} {jetson.ram['use'] / 2.**20} {jetson.gpu['val']}\n" 
            f.write(A_data)
            g_data = f"{jetson.cpu['CPU1']['val']} {jetson.cpu['CPU2']['val']} {jetson.cpu['CPU3']['val']} {jetson.cpu['CPU4']['val']} {jetson.cpu['CPU5']['val']} {jetson.cpu['CPU6']['val']} {jetson.cpu['CPU7']['val']} {jetson.cpu['CPU8']['val']}\n"        
            g.write(g_data)

f.close()
g.close()