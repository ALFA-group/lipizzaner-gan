cd src/
PID_FILE="gan_pids"
rm -f ${PID_FILE}

current_port=5000

for ((i=0; i<4; i++))
do
    for ((j=0; j<6; j++))
    do
      echo "Start client ${j} on GPU ${i}"
      export CUDA_VISIBLE_DEVICES=${i};
      python main.py train --distributed --client &
      echo $! >> ${PID_FILE}
      sleep 2;
    done

    # call the client killing script
    sleep_time=$((RANDOM % 2100))
    previous_port=current_port
    current_port+=4*i
    sed -i 's/previous_port/current_port/g' ../client_killing.sh
    bash -x ../client_killing.sh ${sleep_time} ${port} &

    echo "Start master on GPU ${i}"
    export CUDA_VISIBLE_DEVICES=${i};
    python main.py train --distributed --master -f configuration/quickstart/mnist-${i}.yml #toy-1d-gaussian.yml

    sleep 90

done
echo "Client PIDS:"
cat ${PID_FILE}





# sleep_time=15
# port = 5004
# bash -x ../client_killing.sh ${sleep_time} ${port} &

# echo "Start master on GPU 1"
# export CUDA_VISIBLE_DEVICES=1;
# python main.py train --distributed --master -f configuration/quickstart/mnist-1.yml #toy-1d-gaussian.yml

# sleep 100

# echo "Start master on GPU 0"
# export CUDA_VISIBLE_DEVICES=0;
# python main.py train --distributed --master -f configuration/quickstart/mnist-0.yml #toy-1d-gaussian.yml
# sleep 60;

# echo "Start master on GPU 1"
# export CUDA_VISIBLE_DEVICES=1;
# python main.py train --distributed --master -f configuration/quickstart/mnist-1.yml #toy-1d-gaussian.yml

# sleep 60;
# echo "Start master on GPU 2"
# export CUDA_VISIBLE_DEVICES=2;
# python main.py train --distributed --master -f configuration/quickstart/mnist-2.yml #toy-1d-gaussian.yml

# sleep 60;
# echo "Start master on GPU 3"
# export CUDA_VISIBLE_DEVICES=3;
# python main.py train --distributed --master -f configuration/quickstart/mnist-3.yml #toy-1d-gaussian.yml


echo "Begin kill clients"
cat ${PID_FILE} | xargs -I {} kill -9 {}
echo "Done killing clients"

