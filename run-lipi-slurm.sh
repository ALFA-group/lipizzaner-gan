#!/bin/bash
#SBATCH --job-name=lipi_vgg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16384
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
mkdir $OUTPUT_FOLDER/log
mkdir $OUTPUT_FOLDER/lipizzaner_gan
cp $CONFIG $OUTPUT_FOLDER/config.yml

nvidia-smi -l 60 2>&1 | tee $OUTPUT_FOLDER/nvidiasmi_output.txt &

for i in {0..3}
  do
     python main.py train --distributed --client --port 500$i 2>&1 | tee $OUTPUT_FOLDER/client"$i"_output.txt & sleep 10;
 done

python main.py train --distributed --master -f $CONFIG 2>&1 | tee $OUTPUT_FOLDER/master_output.txt

rm -r $OUTPUT_FOLDER/data
rm -r $OUTPUT_FOLDER/networks
