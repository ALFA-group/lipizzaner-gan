#!/bin/bash


sleep_time=3

CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/celeba-mustang/celeba-7.yml &
sleep 100

CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/celeba-mustang/celeba-8.yml &
sleep 100

CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/celeba-mustang/celeba-9.yml &
sleep 100

CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/celeba-mustang/celeba-10.yml &
sleep 100

CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/celeba-mustang/celeba-11.yml &
sleep 100

CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/celeba-mustang/celeba-12.yml &


