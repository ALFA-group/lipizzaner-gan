#!/bin/bash

killall python

# 1st argument specifies the number of clients to create
num_processes=$1

sleep_time=3

array=()

half_processes=$((num_processes / 2))

echo "Starting $num_processes clients"
# Start the silent client processes
for ((i=0;i<$half_processes-1;i++))
do
  sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client > /dev/null 2>&1 &
  array+=($!)
  echo $i
done

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client & 
array+=($!)
echo $(($num_processes - 1)) 

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/quickstart/mnist.yml &

# Start the silent client processes
for ((i=$half_processes;i<$num_processes-1;i++))
do
  sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client > /dev/null 2>&1 &
  array+=($!)
  echo $i
done

# Open up a single process to see the output from
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client &
array+=($!)
echo $(($num_processes - 1))

# Kill all the processes once a key is entered into the terminal
# read -p "Press Enter to Quit. "
# for i in "${array[@]}"; do
#   kill -9 $i
# done

#sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/quickstart/mnist.yml &

sleep 100

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/quickstart/mnist-2.yml


sleep 121
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/quickstart/mnist.yml &  
sleep 120  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/quickstart/mnist-2.yml 

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/quickstart/mnist.yml &                                    
sleep 120                                                                                                                                                               sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/quickstart/mnist-2.yml       

sleep 300  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/quickstart/mnist.yml & 
sleep 120  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/quickstart/mnist-2.yml

sleep 300  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/quickstart/mnist.yml & 
sleep 120  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/quickstart/mnist-2.yml

sleep 300  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/quickstart/mnist.yml & 
sleep 120  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/quickstart/mnist-2.yml

sleep 300  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/quickstart/mnist.yml & 
sleep 120  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/quickstart/mnist-2.yml

sleep 300  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/quickstart/mnist.yml & 
sleep 120  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/quickstart/mnist-2.yml

sleep 300  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/quickstart/mnist.yml & 
sleep 120  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/quickstart/mnist-2.yml

sleep 300  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/quickstart/mnist.yml & 
sleep 120  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/quickstart/mnist-2.yml

sleep 300  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/quickstart/mnist.yml &
sleep 120  
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/quickstart/mnist-2.yml
