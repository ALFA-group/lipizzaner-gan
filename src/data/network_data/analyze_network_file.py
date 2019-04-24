import sys
import os
import subprocess
import io
import pandas as pd


file_to_analyze = "small_network"

pcap_file = file_to_analyze + ".pcap"
argus_file = file_to_analyze + ".argus"
tmp_file = file_to_analyze + "_tmp.txt"
new_tmp_file = file_to_analyze + "_new_tmp.txt"


remove_old_argus_file = "rm %s"%(argus_file)

convert_to_argus_command = "argus -r %s -w %s"%(pcap_file, argus_file)
read_network_flow_command = "ra -r %s > %s"%(argus_file,tmp_file)

os.system(remove_old_argus_file)
os.system(convert_to_argus_command)

f = open(tmp_file, 'r')
newfile = open(new_tmp_file, 'w')
first_line = f.readline()
for line in f.readlines():
    length = len(line.strip().split())
    if length == 9:
        new_line = []
        start_time = line[0]
        src_addr = line[3]

        print(line.strip().split())
        print(len(line.strip().split()))

    # print(line)

os.system(read_network_flow_command)

data = pd.read_csv(tmp_file, sep='\s+')
print(len(data["SrcAddr"].unique()))

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
