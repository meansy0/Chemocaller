#!/bin/bash				
#SBATCH -J 2_remora_Prepare
#SBATCH -N 1							
#SBATCH -p gpu		
#SBATCH --mem 128g
#SBATCH --gres=gpu:2
#SBATCH -o %x.out						## 作业stdout输出文件为: 作业名_作业id.out
#SBATCH -e %x.err						## 作业stderr 输出文件为: 作业名_作业id.err

# conda deactivate
# conda activate nanopore

# dataset_name=m5c
dataset_name=canonical
# dataset_name=m5u

pre_function=bonito_basecaller


folder_path=/public/home/xiayini/project/NewL/Decoder/data/${dataset_name}
input_file=$folder_path/pod5

sam_folder=$folder_path/bam
bamfile=$sam_folder/$pre_function.bam
sorted_bamfile=$sam_folder/sorted_$pre_function.bam

reference_levels_txt=/public/home/xiayini/project/NewL/1_Classification/reference/5mer_levels_v1.txt
reference_bed=/public/home/xiayini/project/NewL/Decoder/reference/motif0.bed
filtered_bam=$sam_folder/filter_$pre_function.bam
# sam2bam file
module load apps/samtools/1.16.1-gnu485
# samtools sort $bamfile -o $sorted_bamfile

# 使用 samtools view 提取读取记录并筛选出包含 MD 标签的记录

header_sam=/public/home/xiayini/project/NewL/1_Classification/data/canonical_sam/header.sam
filtered_records_sam=/public/home/xiayini/project/NewL/1_Classification/data/canonical_sam/filtered_records.sam
samtools view -H $bamfile > $header_sam
# samtools view $bamfile | awk '$0 ~ /MD:Z:/' > $filtered_records_sam
samtools view $bamfile | awk '$0 ~ /MD:/' > $filtered_records_sam
cat $header_sam $filtered_records_sam | samtools view -Sb - > $filtered_bam
samtools sort $filtered_bam -o $sorted_bamfile
rm $filtered_records_sam
rm $header_sam
# /public/home/xiayini/anaconda3/envs/nanopore/bin/python /public/home/xiayini/project/NewL/1_Classification/scripts/2_0_bam_deal.py $sorted_bamfile $out_sorted_bamfile_0


output_path=$folder_path/prepare
mkdir -p $output_path
cd $output_path


# remora \
#   dataset prepare \
#   $input_file \
#   $sorted_bamfile \
#   --output-path /public/home/xiayini/project/NewL/Decoder/data/motif/C_chuncks \
#   --refine-kmer-level-table $reference_levels_txt \
#   --refine-rough-rescale \
#   --reverse-signal \
#   --motif C 0 \
#   --mod-base  C C 


# remora \
#   dataset prepare \
#   $input_file \
#   $sorted_bamfile \
#   --output-path G_chuncks \
#   --refine-kmer-level-table $reference_levels_txt \
#   --refine-rough-rescale \
#   --reverse-signal \
#   --motif G 0 \
#   --mod-base  G G

remora \
  dataset prepare \
  $input_file \
  $sorted_bamfile \
  --output-path /public/home/xiayini/project/NewL/Decoder/data/motif/T_chuncks \
  --refine-kmer-level-table $reference_levels_txt \
  --refine-rough-rescale \
  --reverse-signal \
  --motif T 0 \
  --mod-base  T T 
  # --focus-reference-positions $reference_bed

# remora \
#   dataset prepare \
#   $input_file \
#   $sorted_bamfile \
#   --output-path A_chuncks \
#   --refine-kmer-level-table $reference_levels_txt \
#   --refine-rough-rescale \
#   --reverse-signal \
#   --motif A 0 \
#   --mod-base  A A

# # # 5mc
# remora \
#   dataset prepare \
#   $input_file \
#   $sorted_bamfile \
#   --output-path control_chuncks \
#   --refine-kmer-level-table $reference_levels_txt \
#   --refine-rough-rescale \
#   --motif C 0 \
#   --mod-base-control

remora \
  dataset prepare \
  $input_file \
  $sorted_bamfile \
  --output-path 5mC_chuncks \
  --refine-kmer-level-table $reference_levels_txt \
  --refine-rough-rescale \
  --reverse-signal \
  --motif C 0 \
  --mod-base  m 5mC
  # --focus-reference-positions $reference_bed

# # 5mu
# remora \
#   dataset prepare \
#   $input_file \
#   $sorted_bamfile \
#   --output-path control_chuncks \
#   --refine-kmer-level-table $reference_levels_txt \
#   --refine-rough-rescale \
#   --motif T 0 \
#   --mod-base-control
# remora \
#   dataset prepare \
#   $input_file \
#   $sorted_bamfile \
#   --output-path m5u_chunck \
#   --refine-kmer-level-table $reference_levels_txt \
#   --refine-rough-rescale \
#   --motif TG 0 \
#   --mod-base  u 5mT


# # 5mu
# remora \
#   dataset prepare \
#   /public/home/xiayini/project/NewL/Decoder/data/canonical/pod5 \
#   /public/home/xiayini/project/NewL/Decoder/data/canonical/bam/sorted_bonito_basecaller.bam \
#   --output-path can_chunks \
#   --refine-kmer-level-table $reference_levels_txt \
#   --refine-rough-rescale \
#   --motif T 0 \
#   --mod-base-control

# remora \
#   dataset prepare \
#   $input_file \
#   $sorted_bamfile \
#   --output-path mod_chunks \
#   --refine-kmer-level-table $reference_levels_txt \
#   --refine-rough-rescale \
#   --motif T 0 \
#   --mod-base t 5mU