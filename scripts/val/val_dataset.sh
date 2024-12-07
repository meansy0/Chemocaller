#!/bin/bash				
#SBATCH -J val_dataset
#SBATCH -N 1							
#SBATCH -p gpu		
#SBATCH --mem 128g
#SBATCH --gres=gpu:2
#SBATCH -o %x.out						## 作业stdout输出文件为: 作业名_作业id.out
#SBATCH -e %x.err						## 作业stderr 输出文件为: 作业名_作业id.err


# cd 
file_folder=/public/home/xiayini/project/NewL/Decoder/data/merge
t1=C
cd /public/home/xiayini/project/NewL/1_Classification/scripts/chemoClassifier
python main.py \
  dataset make_config \
  $file_folder/${t1}_train_dataset.jsn \
  /public/home/xiayini/project/NewL/Decoder/data/canonical/prepare/${t1}_chuncks \
  --dataset-weights 1 \
  --log-filename $file_folder/${t1}_train_dataset.log