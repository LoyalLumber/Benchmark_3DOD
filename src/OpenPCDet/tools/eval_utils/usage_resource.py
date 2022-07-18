import statistics
import time

# for checking up data
class Usage:
	def __init__(self):
		self.FPS = []
		self.CPU = []
		self.GPU = []
		self.RAM = []
		self.CPU_usg = []
		self.GPU_usg = []
		#jetson = jtop()
		#self.jetson = jtop() #if not working, put it back to on top of record_usage func
		
	def update_time(self):
		self.begin_time = time.time()

	def FPS_update(self):
		self.FPS.append(1.0/(time.time() - self.begin_time))
		# print(f"{1.0/(time.time() - self.begin_time)}")

	# for AGX Xavier with 8 cpu core
	def record_usage(self, jetson):
		# self.FPS.append(1.0/(time.time() - self.begin_time))

		# print(f"{1.0/(time.time() - self.begin_time)}")

		#jetson = jtop()
		#jetson.start()
		self.CPU.append(jetson.power[1]['CPU']['avg']/1000)
		self.GPU.append(jetson.power[1]['GPU']['avg']/1000)
		self.RAM.append(jetson.ram['use'] / 2.**20)
		self.CPU_usg.append(jetson.cpu['CPU1']['val'])
		self.CPU_usg.append(jetson.cpu['CPU2']['val'])
		self.CPU_usg.append(jetson.cpu['CPU3']['val'])
		self.CPU_usg.append(jetson.cpu['CPU4']['val'])
		self.CPU_usg.append(jetson.cpu['CPU5']['val'])
		self.CPU_usg.append(jetson.cpu['CPU6']['val'])
		self.CPU_usg.append(jetson.cpu['CPU7']['val'])
		self.CPU_usg.append(jetson.cpu['CPU8']['val'])
		self.GPU_usg.append(jetson.gpu['val'])
		print(f"{jetson.gpu['val']}")

		#jetson.close()

    # for Nano with 4 cpu core and '5V' as a prefix for jetson key value
	def record_usage_nano(self):
		jetson.start()

		self.FPS.append(1.0/(time.time() - self.begin_time))
		self.CPU.append(self.jetson.power[1]['5V_CPU']['avg']/1000)
		self.GPU.append(self.jetson.power[1]['GPU']['avg']/1000)
		self.RAM.append(self.jetson.ram['use'] / 2.**20)
		self.CPU_usg.append(self.jetson.cpu['5V_CPU1']['val'])
		self.CPU_usg.append(self.jetson.cpu['5V_CPU2']['val'])
		self.CPU_usg.append(self.jetson.cpu['5V_CPU3']['val'])
		self.CPU_usg.append(self.jetson.cpu['5V_CPU4']['val'])
		self.GPU_usg.append(self.jetson.gpu['val'])

		jetson.close()

    # record the resource usage in idle state
    # jetson uses quiet big resource if it runs in a loop, so try to run once   
	def record_idle_state(self, name, file_address):
		CPU = []
		GPU = []
		RAM = []
		CPU_usg = []
		GPU_usg = []
			
		jetson = jtop()
		jetson.start()

		CPU.append(jetson.power[1]['CPU']['avg']/1000)
		#CPU.append(jetson.power[1]['5V_CPU']['avg']/1000)
		GPU.append(jetson.power[1]['GPU']['avg']/1000)
		RAM.append(jetson.ram['use'] / 2.**20)
		#for AGX
		CPU_usg.append(jetson.cpu['CPU1']['val'])
		CPU_usg.append(jetson.cpu['CPU2']['val'])
		CPU_usg.append(jetson.cpu['CPU3']['val'])
		CPU_usg.append(jetson.cpu['CPU4']['val'])
		CPU_usg.append(jetson.cpu['CPU5']['val'])
		CPU_usg.append(jetson.cpu['CPU6']['val'])
		CPU_usg.append(jetson.cpu['CPU7']['val'])
		CPU_usg.append(jetson.cpu['CPU8']['val'])
		#for nano
		#CPU_usg.append(jetson.cpu['5V_CPU1']['val'])
		#CPU_usg.append(jetson.cpu['5V_CPU2']['val'])
		#CPU_usg.append(jetson.cpu['5V_CPU3']['val'])
		#CPU_usg.append(jetson.cpu['5V_CPU4']['val'])

		GPU_usg.append(jetson.gpu['val'])

		jetson.close()

		p = open(f"{file_address}/{name}_idle_usg.txt", 'a')    
		p_data = f"{statistics.mean(CPU)} {statistics.mean(GPU)} {statistics.mean(RAM)} {statistics.mean(CPU_usg)} \
		{statistics.mean(GPU_usg)}\n"  
		p.write(p_data)
		p.close()

# for writing data to .txt files

	# write FPS(Hz), CPU(power/W), GPU(power/W), RAM(GB) per inference on datasets
	def write_usage_per_inference(self, board_name, file_address, iou_thresh):
		f = open(f"{file_address}/{board_name}_iou{iou_thresh}.txt", 'a')  
		for i in range(len(FPS)):
			A_data = f"{self.FPS[i]} {self.CPU[i]} {self.GPU[i]} {self.RAM[i]}\n" 
			f.write(A_data)
		f.close()

    # write mean values of FPS(Hz), CPU(power/W), GPU(power/W), RAM(GB), CPU_usg(%), GPU_usg(%) 
	# def write_mean_usage(self, board_name, file_address, iou_thresh):
	# 	p = open(f"{file_address}/{board_name}_mean_iou{iou_thresh}.txt", 'a')   
	# 	p_data = f"{statistics.mean(self.FPS)} {statistics.mean(self.CPU)} {statistics.mean(self.GPU)} {statistics.mean(self.RAM)} {statistics.mean(self.CPU_usg)} {statistics.mean(self.GPU_usg)}\n" 
	# 	p.write(p_data)  
	def write_mean_usage(self, board_name, file_address, iou_thresh,FPS,CPU,GPU,RAM,CPU_usg,GPU_usg):
			p = open(f"{file_address}/{board_name}_mean_iou{iou_thresh}.txt", 'a')   
			p_data = f"{statistics.mean(FPS)} {statistics.mean(CPU)} {statistics.mean(GPU)} {statistics.mean(RAM)} {statistics.mean(CPU_usg)} {statistics.mean(GPU_usg)}\n" 
			p.write(p_data)  