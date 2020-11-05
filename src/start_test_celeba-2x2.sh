#!/bin/bash

killall python

# 1st argument specifies the number of clients to create

sleep_time=3

killall python

CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client > /dev/null 2>&1 &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client > /dev/null 2>&1 &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client > /dev/null 2>&1 &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client &



sleep $sleep_time ;CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf/celeba-2x2.sh
sleep $sleep_time ;CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf/celeba-2x2.sh
sleep $sleep_time ;CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf/celeba-2x2.sh
sleep $sleep_time ;CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf/celeba-2x2.sh
sleep $sleep_time ;CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf/celeba-2x2.sh
sleep $sleep_time ;CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf/celeba-2x2.sh
sleep $sleep_time ;CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf/celeba-2x2.sh
sleep $sleep_time ;CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf/celeba-2x2.sh
sleep $sleep_time ;CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf/celeba-2x2.sh
sleep $sleep_time ;CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/golf/celeba-2x2.sh
