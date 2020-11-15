#!/bin/bash

killall python

sleep_time=3

array=()

half_processes=$((num_processes / 2))

echo "Starting $num_processes clients"
# Start the silent client processes
for ((i=0;i<7;i++))
do
  sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client > /dev/null 2>&1 &
  array+=($!)
  echo $i
done

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-8.yml &

# Start the silent client processes
for ((i=0;i<7;i++))
do
  sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client > /dev/null 2>&1 &
  array+=($!)
  echo $i
done
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client &
sleep 60 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-8-b.yml &

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client > /dev/null 2>&1 &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client &
sleep 60 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2_16-17.yml &

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client > /dev/null 2>&1 &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client &
sleep 60 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2_18-19.yml
sleep 600

sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5-d.yml
sleep 600

sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-5-d.yml
sleep 600

sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-d.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-e.yml
sleep 600

sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-d.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-4-e.yml
sleep 600

sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3-d.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2_16-17.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2_18-19.yml
sleep 600

sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-3-d.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2_16-17.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2_18-19.yml
sleep 600

sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2-d.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2_16-17.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2_18-19.yml
sleep 600

sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2-b.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2-c.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2-d.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2_16-17.yml &
sleep 100 ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf-mnist-scaling/mnist-2_18-19.yml
