#!/bin/bash

# 1st argument specifies the number of clients to create
num_processes=5
sleep_time=2
array=()

# Start the silent client processes
for ((i=0;i<$num_processes-1;i++))
do
  sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client > /dev/null 2>&1 &
  array+=($!)
  echo $i
done

num_processes=4
# Start the silent client processes
for ((i=0;i<$num_processes-1;i++))
do
  sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client > /dev/null 2>&1 &
  array+=($!)
  echo $i
done

# Open up a single process to see the output from
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client &
array+=($!)
echo $(($num_processes - 1))

