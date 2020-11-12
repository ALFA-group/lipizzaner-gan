#!/bin/bash

killall python

# 1st argument specifies the number of clients to create
num_processes=18

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

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-6.yml &

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

sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-6-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-6-c.yml


sleep 60
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-6.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-6-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-6-c.yml

sleep 60 ;CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-6.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-6-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-6-c.yml

sleep 60 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-6-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5-b.yml

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client > /dev/null 2>&1 &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client > /dev/null 2>&1 &

sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5-d.yml &

sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5-d.yml

sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-d.yml

sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-d.yml

sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-d.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3-b.yml

sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3-d.yml

sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3-d.yml

sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2-d.yml

sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2-d.yml

sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2-b.yml
