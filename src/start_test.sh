#!/bin/bash

# 1st argument specifies the number of clients to create
num_processes=$1
sleep_time=5
array=()

# Start the silent client processes
for ((i=0;i<$num_processes-1;i++))
do
  sleep $sleep_time ; python main.py train --distributed --client > /dev/null 2>&1 &
  array+=($!)
  echo $i
done

# Open up a single process to see the output from
sleep $sleep_time ; python main.py train --distributed --client &
array+=($!)
echo $(($num_processes - 1))

# Kill all the processes once a key is entered into the terminal
read -p "Press Enter to Quit. "
for i in "${array[@]}"; do
  kill -9 $i
done
