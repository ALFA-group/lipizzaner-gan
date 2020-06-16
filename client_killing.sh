# parameters are [netstat_output_path, sleep_time, port_path]
echo "client killing script"

# store output of netstat command in file
netstat -ap | grep python &> ${1}

# parse output of netstat command and store ports in file
python ../parsePorts.py ${1} ${3}

sleep ${2}

echo "printing here"

python - << END 
import requests
print("python code inside bash script. currently not putting client to sleep \n")
# response = requests.get("http://127.0.0.1:5000/experiments/sleep") 
# print("Response is " + str(response.status_code))