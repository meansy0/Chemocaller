#!/bin/bash				
#SBATCH -J val_new
#SBATCH -N 1							
#SBATCH -p gpu		
#SBATCH --mem 128g
#SBATCH --gres=gpu:1
#SBATCH -o %x.out						## 作业stdout输出文件为: 作业名_作业id.out
#SBATCH -e %x.err						## 作业stderr 输出文件为: 作业名_作业id.err

# nvidia-smi

python /public/home/xiayini/project/NewL/Decoder/scripts/val/val_new.py