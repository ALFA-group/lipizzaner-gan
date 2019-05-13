import sys
import os
import subprocess
import io
import pandas as pd
from datetime import datetime
import numpy as np
# from helpers.pytorch_helpers import to_pytorch_variable



file_to_analyze = "network_capture"

pcap_file = file_to_analyze + ".pcap"
argus_file = file_to_analyze + ".argus"
tmp_file = file_to_analyze + "_tmp.txt"
new_tmp_file = file_to_analyze + "_new_tmp.txt"


remove_old_argus_file = "rm %s"%(argus_file)
remove_old_tmp_file = "rm %s"%(tmp_file)
remove_old_new_tmp_file = "rm %s"%(new_tmp_file)

convert_to_argus_command = "argus -r %s -w %s"%(pcap_file, argus_file)
read_network_flow_command = "racluster -r %s > %s"%(argus_file,tmp_file)



os.system(remove_old_argus_file)
os.system(remove_old_tmp_file)
os.system(remove_old_new_tmp_file)
os.system(convert_to_argus_command)
os.system(read_network_flow_command)


f = open(tmp_file, 'r')
newfile = open(new_tmp_file, 'w')
first_line = f.readline()
first = True

current_address = 0
seen_addresses = {}

data = []

SEQUENCE_LENGTH = 15

sequence_data = []
sequence_num = 0
for line in f.readlines():

    split_line = line.strip().split()
    length = len(split_line)
    if length == 9:
        start_time = split_line[0]
        start_time_datetime = datetime.strptime(start_time, "%H:%M:%S.%f")
        if first:
            first_time = start_time_datetime
            last_time = start_time_datetime
            first = False

        else:
            elapsed_time = (last_time - start_time_datetime).total_seconds()
            last_time = start_time_datetime

            # Extract elements of the source address
            src_addr = split_line[3]
            src_addr_parts = src_addr.split('.')
            print(src_addr_parts)
            if len(src_addr_parts) != 5:
                continue
            src_addr_joined = ''.join(src_addr_parts[:4])
            print(split_line)

            if src_addr_joined in seen_addresses:
                src_addr_num = seen_addresses[src_addr_joined]
            else:
                src_addr_num = current_address
                seen_addresses[src_addr_joined] = src_addr_num
                current_address += 1

            # Extract elements of the destination address
            dest_addr = split_line[5]
            dest_addr_parts = dest_addr.split('.')
            if len(dest_addr_parts) != 5:
                continue
            dest_addr_joined = ''.join(dest_addr_parts[:4])

            if dest_addr_joined in seen_addresses:
                dest_addr_num = seen_addresses[src_addr_joined]
            else:
                dest_addr_num = current_address
                seen_addresses[dest_addr_joined] = dest_addr_num
                current_address += 1

            packet_size = int(split_line[7])

            new_line = [elapsed_time, src_addr_num, dest_addr_num, packet_size]
            sequence_data.append(new_line)
            sequence_num += 1
            if sequence_num >= SEQUENCE_LENGTH:
                data.append(sequence_data)
                sequence_data = []
                sequence_num = 0


data = np.array(data)
print(data)
print(data.shape)
np.save("small_network_data_clean", data)
# print(data[-1])
        # ['15:35:25.891875', 'e', 'udp', '18.21.171.215.57035', '->', '239.255.255.250.ssdp', '1', '217', 'INT']

        # print(line.strip().split())
        # print(len(line.strip().split()))

    # print(line)

# os.system(read_network_flow_command)
#
# data = pd.read_csv(tmp_file, sep='\s+')
# print(len(data["SrcAddr"].unique()))

# Data Format:
# list of (target, time_since_last_request, packet_size, num_packets)



# data['cleanSrcAddr'] =





# process = subprocess.Popen(read_network_flow_command.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()



# print(output)

# command = " tshark -r network_capture.pcap1 -z flow,tcp,standard"

# res = os.system(commansd)

# print(len(res))





# print(output)
