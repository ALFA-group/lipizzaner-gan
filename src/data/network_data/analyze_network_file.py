import argparse
import os
from datetime import datetime

import numpy as np


def main(file_to_analyze, sequence_length):
    # Defining and creating new files to be created
    basename = os.path.splitext(file_to_analyze)[0]
    pcap_file = "{}.pcap".format(basename)
    argus_file = "{}.argus".format(basename)
    tmp_file = "{}.txt".format(basename)

    # Remove previous versions of these files
    if os.path.exists(argus_file):
        os.remove(argus_file)
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    # Convert to argus and get the required fields
    convert_to_argus_command = "argus -r {} -w {}".format(pcap_file, argus_file)
    read_argus_command = "racluster -r {} -M rmon dsrs=\"-agr\" -m smac saddr -s stime dur:20 pkts bytes trans > {}".format(argus_file, tmp_file)
    os.system(convert_to_argus_command)
    os.system(read_argus_command)


    with open(tmp_file, 'r') as f:
        first_line = f.readline()
        first = True
        data = []
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
            if sequence_num >= sequence_length:
                data.append(sequence_data)
                sequence_data = []
                sequence_num = 0

    data = np.array(data)
    np.save("{}.npy".format(basename), data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze pcap file')
    parser.add_argument('--pcap_file',
                        type=str,
                        required=True,
                        help='Name of pcap file to extract the netflow data from.')
    parser.add_argument('--sequence_length',
                        type=int,
                        required=True,
                        default=30,
                        help='Length of a sequence.')
    args = parser.parse_args()
    main(args.pcap_file, args.sequence_length)
