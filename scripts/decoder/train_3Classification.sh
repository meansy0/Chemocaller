#!/bin/bash				
#SBATCH -J train_3Classification
#SBATCH -N 1							
#SBATCH -p gpu		
#SBATCH --mem 128g
#SBATCH --gres=gpu:1
#SBATCH -o %x.out						## 作业stdout输出文件为: 作业名_作业id.out
#SBATCH -e %x.err						## 作业stderr 输出文件为: 作业名_作业id.err


input_folder=/public/home/xiayini/project/NewL/Decoder/data/merge
epochs_num=50

# type1=can
# type2=mod

type1=C
type2=5mC
type3=T

save_folder_path=/public/home/xiayini/project/NewL/Decoder/results/${type1}_${type2}_${type3}/epoch${epochs_num}
mkdir -p $save_folder_path
rm -r $save_folder_path

# Purine-腺嘌呤（A）、鸟嘌呤（G）
# Pyrimidine-胞嘧啶（C）、胸腺嘧啶（T）、尿嘧啶（U）

folder=/public/home/xiayini/project/NewL/Decoder
cd $folder/scripts/decoder
python main.py \
  model train \
  $input_folder/${type1}_${type2}_${type3}_train_dataset.jsn \
  --model model_Pyrimidine.py \
  --device 0 \
  --epochs $epochs_num \
  --chunk-context 50 50 \
  --chunks-per-epoch 1000000 \
  --output-path $save_folder_path

