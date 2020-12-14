# parameters are [sleep_time]
echo "client killing script"

sleep ${1};

python - << END 
import requests
import time
# print("python code inside bash script. not putting any clients to sleep \n")
print("python code inside bash script. putting one client to sleep! \n")
response = requests.get("http://127.0.0.1:5018/experiments/sleep")


# response = requests.get("http://127.0.0.1:/experiments/sleep") 
# print("Response to waking up is " + str(response.status_code))
# print("Should have woken up client")
