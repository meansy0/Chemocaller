#!/bin/bash				
#SBATCH -J train100
#SBATCH -N 1							
#SBATCH -p gpu		
#SBATCH --mem 128g
#SBATCH --gres=gpu:1
#SBATCH -o %x.out						## 作业stdout输出文件为: 作业名_作业id.out
#SBATCH -e %x.err						## 作业stderr 输出文件为: 作业名_作业id.err

epochs_num=100

# model_noseqs_lstm3 model_noseqs_lstm2 model_seqs_lstm2
model_type=model_noseqs_lstm2
j_filename=train_${epochs_num}

# input_folder=/public/home/xiayini/project/NewL/Decoder/data/merge
input_folder=/public/home/xiayini/project/NewL/Decoder/newdata_Process3.0/merge/train


type1=C_5mC
type2=T

# type1=T
# type2=5mC

folder1=/public/home/xiayini/project/NewL/Decoder/results_newdata_Process3.0
save_folder_path=$folder1/${type1}_${type2}/epoch${epochs_num}_${model_type}
mkdir -p $save_folder_path
rm -r $save_folder_path

# Purine-腺嘌呤（A）、鸟嘌呤（G）
# Pyrimidine-胞嘧啶（C）、胸腺嘧啶（T）、尿嘧啶（U）

model_file=/public/home/xiayini/project/NewL/Decoder/scripts/model/Pyrimidine/${model_type}.py
folder=/public/home/xiayini/project/NewL/Decoder
cd $folder/scripts/decoder
python main.py \
  model train \
  $input_folder/${type1}_${type2}_dataset.jsn \
  --model $model_file \
  --device 0 \
  --epochs $epochs_num \
  --chunk-context 50 50 \
  --chunks-per-epoch 10000000 \
  --num-test-chunks 10 \
  --output-path ${save_folder_path} \
  --batch-size 4096


python plot_batch.py \
  $epochs_num \
  $save_folder_path

cp $folder1/${type1}_${type2}/$j_filename.err $save_folder_path
cp $folder1/${type1}_${type2}/$j_filename.out $save_folder_path
cp /public/home/xiayini/project/NewL/Decoder/scripts/decoder/train.sh $save_folder_path

# mv $save_folder_path/${model_type}.py $save_folder_path/model_Pyrimidine.py 
# /public/home/xiayini/project/NewL/Decoder/data/merge/T_5mC_train_dataset.jsn
