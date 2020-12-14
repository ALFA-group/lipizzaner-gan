cd src/
PID_FILE="gan_pids"
rm -f ${PID_FILE}

current_port=5000
previous_port=${current_port}

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
    sleep_time=15 #$(((RANDOM % 2100)))
 
    echo "previous port is ${previous_port} current port is ${current_port}"
    sed -i "s/${previous_port}/${current_port}/g" ../client_killing.sh 
    bash -x ../client_killing.sh ${sleep_time} &
    previous_port=${current_port}
    let "current_port += 6"
    cat ../client_killing.sh 
    
    echo "Start master on GPU ${i}"
    export CUDA_VISIBLE_DEVICES=${i};
    
    if (( ${i} < 3 ))
    then
      python main.py train --distributed --master -f configuration/quickstart/mnist-${i}.yml &
      echo "master with ampersand"
    else
      python main.py train --distributed --master -f configuration/quickstart/mnist-${i}.yml
      echo "master without"
    fi
    sleep 90 
done

echo "Client PIDS:"
cat ${PID_FILE}

echo "Begin kill clients"
cat ${PID_FILE} | xargs -I {} kill -9 {}
echo "Done killing clients"

