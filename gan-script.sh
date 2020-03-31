cd src/

PID_FILE="gan_pids"
rm -f ${PID_FILE}

for ((i=0; i<4; i++))
do 
    echo "Start client on GPU ${i}"
    export CUDA_VISIBLE_DEVICES=${i}; 
    python main.py train --distributed --client &
    echo $! >> ${PID_FILE}
    sleep 5;
done

echo "Client PIDS:"
cat ${PID_FILE}
sleep 5

# kill number of clients
bash client_failure_exp.sh 4

echo "Start master on GPU 4"
# export CUDA_VISIBLE_DEVICES=4; 
python main.py train --distributed --master -f configuration/quickstart/mnist.yml

echo "Begin kill clients"
cat ${PID_FILE} | xargs -I {} kill -9 {}

echo "Done killing clients"
