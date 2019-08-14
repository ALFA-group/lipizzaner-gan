import argparse
import os
from datetime import datetime

import numpy as np


def main(file_to_analyze):
    # Defining and creating new files to be created
    pcap_file = file_to_analyze + ".pcap"
    argus_file = file_to_analyze + ".argus"
    tmp_file = file_to_analyze + ".txt"

    # Remove previous versions of these files
    remove_old_argus_file = "rm %s"%(argus_file)
    remove_old_tmp_file = "rm %s"%(tmp_file)
    os.system(remove_old_argus_file)
    os.system(remove_old_tmp_file)
    # os.system("touch %s; chmod 777 %s"%(tmp_file,tmp_file))


    # Convert to argus and get the required fields
    convert_to_argus_command = "argus -r %s -w %s"%(pcap_file, argus_file)
    read_argus_command = "racluster -r %s -M rmon dsrs=\"-agr\" -m smac saddr -s stime dur:20 pkts bytes trans > %s"%(argus_file, tmp_file)
    os.system(convert_to_argus_command)
    os.system(read_argus_command)


    f = open(tmp_file, 'r')
    first_line = f.readline()

    first = True
    data = []

    # TODO: Get this as an argument in the future
    SEQUENCE_LENGTH = 30

    sequence_data = []
    sequence_num = 0

    # Read the data and create the dataset
    for line in f.readlines():
        split_line = line.strip().split()
        length = len(split_line)
        start_time = split_line[0]
        start_time_datetime = datetime.strptime(start_time, "%H:%M:%S.%f")
        if first:
            last_time = start_time_datetime
            first = False
            continue

        elapsed_time = (last_time - start_time_datetime).total_seconds()
        last_time = start_time_datetime

        flow_duration = float(split_line[1])
        num_packets = int(split_line[2])
        num_bytes = int(split_line[3])

        new_line = [elapsed_time, flow_duration, num_packets, num_bytes]
        sequence_data.append(new_line)
        sequence_num += 1
        if sequence_num >= SEQUENCE_LENGTH:
            data.append(sequence_data)
            sequence_data = []
            sequence_num = 0


    data = np.array(data)
    np.save(file_to_analyze + ".npy", data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze pcap file')
    parser.add_argument('--pcap_file',
                        type=str,
                        required=True,
                        help='Name of pcap file to extract the netflow data from; do not include the .pcap at the end.')
    args = parser.parse_args()
    main(args.pcap_file)
