#!/bin/bash

killall python

# 1st argument specifies the number of clients to create
num_processes=16

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

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-8.yml &

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

sleep 100

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-8-b.yml


sleep 60
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-8.yml &
sleep 120
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-8-b.yml

sleep 60
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-8.yml &
sleep 120
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-8-b.yml

sleep 60
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-8.yml &
sleep 120
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-8-b.yml


sleep 60
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-7.yml &
sleep 120
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-7-b.yml

sleep 60
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-7.yml &
sleep 120
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-7-b.yml

sleep 60
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-7.yml &
sleep 120
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-7-b.yml

sleep 60
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-7.yml &
sleep 120
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-7-b.yml
