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
import time
# print("python code inside bash script. not putting any clients to sleep \n")
print("python code inside bash script. putting one clients to sleep \n")
response = requests.get("http://127.0.0.1:5000/experiments/sleep") 
print("Response to putting to sleep is " + str(response.status_code))
time.sleep(35)
response = requests.get("http://127.0.0.1:5000/experiments/sleep") 
print("Response to waking up is " + str(response.status_code))
print("Should have woken up client")
