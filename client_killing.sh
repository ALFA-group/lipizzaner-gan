# parameters are [netstat_output_path, sleep_time, port_path]
echo "client killing script"

# store output of netstat command in file
netstat -ap | grep python &> ${1}

# parse output of netstat command and store ports in file
python ../parsePorts.py ${1} ${3}

sleep ${2}

echo "printing here"