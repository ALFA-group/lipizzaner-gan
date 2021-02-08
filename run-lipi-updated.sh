#!/bin/bash
#SBATCH --job-name=lipi7030
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=8196
#SBATCH --time=6:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

export PATH=$PATH:/usr/local/cuda-9.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.2/lib64

cd lipizzaner-gan/src

DATE=$(date "+%F-%T")
export OUTPUT_FOLDER=../../job_"$SLURM_JOBID"_run_"$DATE"_"$SLURM_JOB_NAME"
CONFIG="configuration/lipizzaner-gan/mnist_diverse.yml"

mkdir $OUTPUT_FOLDER
cp -r ./output/data $OUTPUT_FOLDER
cp -r ./output/networks $OUTPUT_FOLDER
cp -r ./output/log $OUTPUT_FOLDER
cp -r ./output/lipizzaner_gan $OUTPUT_FOLDER
cp $CONFIG $OUTPUT_FOLDER/config.yml

nvidia-smi -l 60 2>&1 | tee $OUTPUT_FOLDER/nvidiasmi_output.txt &


python main.py train --distributed --client --port 5000 2>&1 | tee $OUTPUT_FOLDER/client1_output.txt & sleep 10;
python main.py train --distributed --client --port 5001 2>&1 | tee $OUTPUT_FOLDER/client2_output.txt & sleep 10;
python main.py train --distributed --client --port 5002 2>&1 | tee $OUTPUT_FOLDER/client3_output.txt & sleep 10;
python main.py train --distributed --client --port 5003 2>&1 | tee $OUTPUT_FOLDER/client4_output.txt & sleep 20;
python main.py train --distributed --master -f $CONFIG 2>&1 | tee $OUTPUT_FOLDER/master_output.txt
