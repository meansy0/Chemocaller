#!/bin/bash				
#SBATCH -J 1_basecaller_m5c
#SBATCH -N 1							
#SBATCH -p gpu		
#SBATCH --mem 128g
#SBATCH --gres=gpu:2
#SBATCH -o %x.out						## 作业stdout输出文件为: 作业名_作业id.out
#SBATCH -e %x.err						## 作业stderr 输出文件为: 作业名_作业id.err

# bonito  basecaller rna002_70bps_hac@v3  --modified-bases 5mU --rna --save-ctc --reference $fasta_file $input_file > $output_file

# conda deactivate
# conda activate nanopore

reference_file=/public/home/xiayini/project/NewL/1_Classification/reference/rna.fasta

dataset_name=m5c
# dataset_name=canonical
# dataset_name=m5u
folder_path=/public/home/xiayini/project/NewL/Decoder

input_file=$folder_path/newdata/${dataset_name}/pod5

new_trainModel=/public/home/xiayini/project/NewL/Decoder/scripts/train

out_folder=$folder_path/newdata/${dataset_name}/bam
output_file=$out_folder/bonito_basecaller.bam
mkdir -p $out_folder
# bonito basecaller rna002_70bps_sup@v3 --rna --reference $reference_file $input_file > $output_file
bonito basecaller $new_trainModel --rna --reference $reference_file $input_file > $output_file

# bonito basecaller rna002_70bps_hac@v3  --rna --save-ctc --reference $fasta_file $input_file > $output_file
# $ bonito train --directory /data/training/ctc-data /data/training/model-dir