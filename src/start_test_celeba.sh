#!/bin/bash

killall python

sleep_time=3

CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/celeba-mustang/celeba-1.yml &
sleep 100

CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/celeba-mustang/celeba-2.yml &
sleep 100

CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/celeba-mustang/celeba-3.yml &
sleep 100

CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/celeba-mustang/celeba-4.yml &
sleep 100

CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/celeba-mustang/celeba-5.yml &
sleep 100

CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/celeba-mustang/celeba-6.yml &


