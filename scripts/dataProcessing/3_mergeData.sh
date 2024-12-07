#!/bin/bash				
#SBATCH -J 3_mergeData
#SBATCH -N 1							
#SBATCH -p gpu		
#SBATCH --mem 128g
#SBATCH --gres=gpu:2
#SBATCH -o %x.out						## 作业stdout输出文件为: 作业名_作业id.out
#SBATCH -e %x.err						## 作业stderr 输出文件为: 作业名_作业id.err



# dataset_name=m5c
dataset_name=canonical
# dataset_name=m5u
# type1=C
# type2=T

# type1=A
# type2=G

# type1=control
# type2=m5u

# type1=control
# type2=m5c

folder=/public/home/xiayini/project/NewL/Decoder/data/$dataset_name/prepare

# cd /public/home/xiayini/project/NewL/1_Classification/scripts/chemoClassifier
# python main.py \
#   dataset make_config \
#   $folder/${type1}_${type2}_train_dataset.jsn \
#   $folder/${type1}_chuncks \
#   $folder/${type2}_chuncks \
#   --dataset-weights 1 1 \
#   --log-filename $folder/${type1}_${type2}_train_dataset.log


# cd 
file_folder=/public/home/xiayini/project/NewL/Decoder/data/merge
t1=T
t2=5mC
cd /public/home/xiayini/project/NewL/1_Classification/scripts/chemoClassifier
python main.py \
  dataset make_config \
  $file_folder/${t1}_${t2}_train_dataset.jsn \
  /public/home/xiayini/project/NewL/Decoder/data/canonical/prepare/${t1}_chuncks \
  /public/home/xiayini/project/NewL/Decoder/data/m5c/prepare/${t2}_chuncks \
  --dataset-weights 1 1 \
  --log-filename $file_folder/${t1}_${t2}_train_dataset.log

# # cd 
# file_folder=/public/home/xiayini/project/NewL/Decoder/data/merge
# t1=C
# t2=5mC
# t3=T
# cd /public/home/xiayini/project/NewL/1_Classification/scripts/chemoClassifier
# python main.py \
#   dataset make_config \
#   $file_folder/${t1}_${t2}_${t3}_train_dataset.jsn \
#   /public/home/xiayini/project/NewL/Decoder/data/canonical/prepare/${t1}_chuncks \
#   /public/home/xiayini/project/NewL/Decoder/data/m5c/prepare/${t2}_chuncks \
#   /public/home/xiayini/project/NewL/Decoder/data/canonical/prepare/${t3}_chuncks \
#   --dataset-weights 1 1 1 \
#   --log-filename $file_folder/${t1}_${t2}_${t3}_train_dataset.log



# cd 
file_folder=/public/home/xiayini/project/NewL/Decoder/data/motif
t1=C
# t2=5mC
# t3=T

merge_name=C
cd /public/home/xiayini/project/NewL/1_Classification/scripts/chemoClassifier
python main.py \
  dataset make_config \
  $file_folder/${merge_name}_train_dataset.jsn \
  $file_folder/${t1}_chuncks \
  --dataset-weights 1 \
  --log-filename $file_folder/${merge_name}_train_dataset.log