from jtop import jtop
import time

# jetson = jtop()
# jetson.start()
# stat = jetson.stats
# print(stat)
# print(jetson.ram)
# jetson.close()

if __name__ == "__main__":

    print("All accessible jtop properities")

    with jtop() as jetson:
        # boards
        # print('*** board ***')
        # print(jetson.board)
        # jetson.ok() will provide the proper update frequency
        while jetson.ok():
            # uptime
            # print('*** uptime ***')
            # print(jetson.uptime)
            # F_data = f"{jetson.uptime}.\n"
            # f.write(F_data)
            # CPU
            # print('*** CPUs ***')
            # print(jetson.cpu)
            # GPU
            # print('*** GPU ***')
            # print(jetson.gpu)
            # Power
            # print('*** power ***')
            # print(jetson.power)
            # IRAM
            print('*** ram ***')
            print(jetson.power[1]['CPU']['avg']/1000)
            print(jetson.power[1]['GPU']['avg']/1000)
            # IRAM
            # print('*** iram ***')
            # print(jetson.iram)
