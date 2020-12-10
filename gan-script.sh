cd src/

PID_FILE="gan_pids"
rm -f ${PID_FILE}

for ((i=0; i<6; i++))
do 
    echo "Start client on GPU ${i}"
    export CUDA_VISIBLE_DEVICES=${i}; 
    python main.py train --distributed --client &
    echo $! >> ${PID_FILE}
    sleep 2;
done

echo "Client PIDS:"
cat ${PID_FILE}

# call the client killing script
sleep_time=15
port_to_kill=5000
bash -x ../client_killing.sh ${sleep_time} ${port_to_kill} &

echo "Start master on GPU 3"
export CUDA_VISIBLE_DEVICES=3; 
python main.py train --distributed --master -f configuration/quickstart/mnist.yml #toy-1d-gaussian.yml 

echo "Begin kill clients"
cat ${PID_FILE} | xargs -I {} kill -9 {}

echo "Done killing clients"
